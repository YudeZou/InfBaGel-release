import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import Adam
from utils import *
from constants import *
import os
from torch.utils.tensorboard import SummaryWriter
import datetime
from datasets.infbagel_mix import InfBaGelMixDataset

os.environ['ROOT_DIR'] = '..'
os.environ['HYDRA_FULL_ERROR'] = '1'
os.environ['CURRENT_TIME'] = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['NCCL_P2P_DISABLE'] = '0'
os.environ['NCCL_IB_DISABLE'] = '0'

import sys
sys.path.append(os.path.join(os.environ['ROOT_DIR'], 'code'))

@hydra.main(version_base=None, config_path="config", config_name="config_train_infbagel_mix")
def train(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = find_free_port()
    world_size = cfg.num_gpus
    print('Usable GPUS: ', torch.cuda.device_count(), flush=True)
    torch.multiprocessing.spawn(train_ddp,
                                args=(world_size, cfg),
                                nprocs=world_size,
                                join=True)

def train_ddp(rank, world_size, cfg):

    OmegaConf.register_new_resolver("times", lambda x, y: int(x) * int(y))

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    cfg.device = f"cuda:{rank}"
    print(f'Training on {device}', flush=True)
    print('Initializing Distributed', flush=True)
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

    model = init_model(list(cfg.model.values())[0], device=rank, eval=False, load_state_dict=cfg.load_state_dict)
    
    optimizer = Adam(model.parameters(), lr=cfg.lr)

    infbagel_dataset = InfBaGelMixDataset(**cfg.dataset)

    sampler = DistributedSampler(infbagel_dataset)
    dataloader = DataLoader(infbagel_dataset, batch_size=cfg.batch_size, drop_last=True, num_workers=cfg.num_workers,
                            sampler=sampler, pin_memory=True, persistent_workers=True)

    trainer = hydra.utils.instantiate(list(cfg.sampler.values())[0])
    trainer.set_dataset_and_model(infbagel_dataset, model)

    if cfg.use_tensorboard and rank == 0:
        writer = SummaryWriter(log_dir=os.path.join(cfg.exp_dir, 'tensorboard_logs'))

    for epoch in range(cfg.start_epoch, cfg.epochs):
        print(f'Start epoch {epoch}', flush=True)
        sampler.set_epoch(epoch)

        step = 0
        for batch in dataloader:
            step += 1
            optimizer.zero_grad()

            joints, mat, object_trans, object_rot_mat, scene_flag, text_clip_embedding, pelvis_goal, hand_goal, object_goal,\
                is_pick, need_scene, need_pelvis_dir, pi, need_pi, is_loco, is_object, obj_bps_data, obj_rot_mat_ref, rest_pose_obj_nn_pts, transformed_obj_verts, object_points = \
                batch['joints'].to(device), batch['mat'].to(device), batch['object_trans'].to(device), batch['object_rot_mat'].to(device), batch['scene_flag'].to(device), \
                batch['text_clip_embedding'].to(device), batch['pelvis_goal'].to(device), batch['hand_goal'].to(device), batch['object_goal'].to(device), \
                batch['is_pick'].to(device), batch['need_scene'].to(device), batch['need_pelvis_dir'].to(device), batch['pi'].to(device), batch['need_pi'].to(device), batch['is_loco'].to(device), \
                batch['is_object'].to(device), batch['obj_bps_data'].to(device), batch['obj_rot_mat_ref'].to(device), batch['rest_pose_obj_nn_pts'].to(device), batch['transformed_obj_verts'].to(device), batch['object_points'].to(device)

            global_rot_6d = batch['global_rot_6d'].to(device).reshape(batch['global_rot_6d'].shape[0], batch['global_rot_6d'].shape[1], -1)
            contact_label = batch['contact_label'].to(device)
            rest_human_offsets = batch['rest_human_offsets'].to(device)
            seg_len = batch['seg_len'].to(device)
            end_pi = batch['end_pi'].to(device)

            t = torch.randint(0, trainer.timesteps, (cfg.batch_size,), device=device).long()
            x_start = torch.cat([joints, global_rot_6d, object_trans, object_rot_mat.reshape(object_rot_mat.shape[0], object_rot_mat.shape[1], -1), contact_label], dim=-1) # 84 + 132 + 3 + 9 + 4
            with torch.no_grad():
                mask, _, _ = get_mask(x_start, -1, p=1., fixed_frame=cfg.auto_regre_num)
                
            loss, loss_fk, loss_object, _ = trainer.p_losses(x_start, joints, mat, scene_flag, mask, t, text_clip_embedding, pelvis_goal, hand_goal, object_goal, \
                is_pick, need_scene, need_pelvis_dir, pi, end_pi, seg_len, need_pi, is_loco, is_object, obj_bps_data, obj_rot_mat_ref, rest_pose_obj_nn_pts, transformed_obj_verts, rest_human_offsets, object_points)

            if loss_object is not None:
                loss = loss + cfg.loss_w_obj_pts * loss_object + cfg.loss_w_fk * loss_fk

            if step % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                if loss_object is not None:
                    print(f"Epoch: {epoch}, Step: {step} / {len(dataloader)}   Loss: {loss.item()}, Loss_fk: {loss_fk.item()}, Loss_object: {loss_object.item()}, LR: {current_lr:.6f}", flush=True)
                else:
                    print(f"Epoch: {epoch}, Step: {step} / {len(dataloader)}   Loss: {loss.item()}, LR: {current_lr:.6f}", flush=True)
                if cfg.use_tensorboard and rank == 0:
                    writer.add_scalar('Loss', loss.item(), epoch * len(dataloader) + step)
                    writer.add_scalar('Loss_fk', loss_fk.item(), epoch * len(dataloader) + step)
                    if loss_object is not None:
                        writer.add_scalar('Loss_object', loss_object.item(), epoch * len(dataloader) + step)
                    writer.add_scalar('Learning Rate', current_lr, epoch * len(dataloader) + step)

            loss.backward()
            optimizer.step()


        if rank == 0 and epoch % cfg.ckpt_interval == 0:
            print(f'Saving checkpoint', flush=True)
            ckpt_folder = os.path.join(cfg.exp_dir, 'checkpoints')
            os.makedirs(ckpt_folder, exist_ok=True)
            torch.save(model.module.state_dict(), os.path.join(ckpt_folder, f"{cfg.exp_name}_epoch{epoch:03d}.pth"))

        torch.distributed.barrier()

        print('Clearing cache', flush=True)
        torch.cuda.empty_cache()


def get_mask(x_start, ind, p, fixed_frame=0, mask_y=True):
    '''
    get mask for the input sequence of pre frames and final goal frame
    '''
    mask_frame = torch.zeros_like(x_start).to(dtype=torch.bool, device=x_start.device)
    mask_goal = torch.zeros_like(x_start).to(dtype=torch.bool, device=x_start.device)

    # goal mask
    if ind != -1:
        rand_batch = torch.rand(x_start.shape[0]).to(x_start.device) < p
        mask_goal[rand_batch, -1, ind * 3: ind * 3 + 3] = True
        if not mask_y:
            mask_goal[rand_batch, -1, ind * 3 + 1] = False

    # prefix frame mask
    if fixed_frame > 0:
        rand_batch = torch.rand(x_start.shape[0]).to(x_start.device) < p
        mask_frame[rand_batch, :fixed_frame, :] = True
    mask = torch.logical_or(mask_frame, mask_goal)
    return mask, mask_frame, mask_goal


if __name__ == '__main__':
    train()
