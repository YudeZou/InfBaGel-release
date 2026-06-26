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

os.environ['ROOT_DIR'] = '..'
os.environ['HYDRA_FULL_ERROR'] = '1'
os.environ['CURRENT_TIME'] = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['NCCL_P2P_DISABLE'] = '0'
os.environ['NCCL_IB_DISABLE'] = '0'

import sys
sys.path.append(os.path.join(os.environ['ROOT_DIR'], 'code'))

# batch fields consumed by the training step (the only tensors moved to GPU each iteration;
# GT / metadata fields returned by the dataset are intentionally left on CPU)
TRAIN_BATCH_KEYS = (
    'joints', 'mat', 'object_trans', 'object_rot_mat', 'scene_flag',
    'text_clip_embedding', 'pelvis_goal', 'scene_goal', 'object_goal',
    'need_scene', 'need_pelvis_dir', 'pi', 'need_pi', 'is_loco', 'is_object',
    'obj_bps_data', 'obj_rot_mat_ref', 'rest_pose_obj_nn_pts', 'transformed_obj_verts', 'object_points',
    'global_rot_6d', 'contact_label', 'rest_human_offsets', 'seg_len', 'end_pi',
)

@hydra.main(version_base=None, config_path="config", config_name="config_train_infbagel")
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

    # cfg.sample_type selects the training objective:
    #   diffusion   -> standard diffusion training via trainer.p_losses
    #   consistency -> consistency-model distillation via trainer.consistency_loss
    is_consistency = cfg.sample_type == 'consistency'

    if is_consistency:
        teacher_model = init_model(list(cfg.model.values())[0], device=rank, eval=False, load_state_dict=cfg.load_state_dict)
        teacher_model.requires_grad_(False)

        student_model = init_model(list(cfg.model.values())[0], device=rank, eval=False, load_state_dict=cfg.load_state_dict)
        student_model.requires_grad_(False)
        student_model.module.embedding_input.requires_grad_(True)
        student_model.module.embedding_output.requires_grad_(True)
        student_model.module.transformer.requires_grad_(True)
        student_model.module.out.requires_grad_(True)

        target_model = init_model(list(cfg.model.values())[0], device=rank, eval=False, load_state_dict=cfg.load_state_dict)
        target_model.requires_grad_(False)

        model = student_model
        optimizer = Adam(student_model.parameters(), lr=cfg.lr)
    else:
        model = init_model(list(cfg.model.values())[0], device=rank, eval=False, load_state_dict=cfg.load_state_dict)
        optimizer = Adam(model.parameters(), lr=cfg.lr)

    infbagel_dataset = hydra.utils.instantiate(cfg.dataset)

    sampler = DistributedSampler(infbagel_dataset)
    dataloader = DataLoader(infbagel_dataset, batch_size=cfg.batch_size, drop_last=True, num_workers=cfg.num_workers,
                            sampler=sampler, pin_memory=True, persistent_workers=True)

    trainer = hydra.utils.instantiate(list(cfg.sampler.values())[0])
    if is_consistency:
        trainer.set_dataset_and_model(infbagel_dataset, student_model, teacher_model, target_model)
    else:
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

            # async H2D copy for the training tensors (DataLoader sets pin_memory=True)
            b = {k: batch[k].to(device, non_blocking=True) for k in TRAIN_BATCH_KEYS}

            joints, mat, object_trans, object_rot_mat, scene_flag = \
                b['joints'], b['mat'], b['object_trans'], b['object_rot_mat'], b['scene_flag']
            text_clip_embedding, pelvis_goal, scene_goal, object_goal = \
                b['text_clip_embedding'], b['pelvis_goal'], b['scene_goal'], b['object_goal']
            need_scene, need_pelvis_dir, pi, need_pi, is_loco, is_object = \
                b['need_scene'], b['need_pelvis_dir'], b['pi'], b['need_pi'], b['is_loco'], b['is_object']
            obj_bps_data, obj_rot_mat_ref, rest_pose_obj_nn_pts, transformed_obj_verts, object_points = \
                b['obj_bps_data'], b['obj_rot_mat_ref'], b['rest_pose_obj_nn_pts'], b['transformed_obj_verts'], b['object_points']
            contact_label, rest_human_offsets, seg_len, end_pi = \
                b['contact_label'], b['rest_human_offsets'], b['seg_len'], b['end_pi']

            global_rot_6d = b['global_rot_6d'].reshape(b['global_rot_6d'].shape[0], b['global_rot_6d'].shape[1], -1)

            t = torch.randint(0, trainer.timesteps, (cfg.batch_size,), device=device).long()
            x_start = torch.cat([joints, global_rot_6d, object_trans, object_rot_mat.reshape(object_rot_mat.shape[0], object_rot_mat.shape[1], -1), contact_label], dim=-1) # 84 + 132 + 3 + 9 + 4
            with torch.no_grad():
                mask, _, _ = get_mask(x_start, -1, p=1., fixed_frame=cfg.auto_regre_num)

            if is_consistency:
                loss_dict = trainer.consistency_loss(x_start, joints, mat, scene_flag, mask, t, text_clip_embedding, pelvis_goal, scene_goal, object_goal, \
                    need_scene, need_pelvis_dir, pi, end_pi, seg_len, need_pi, is_loco, is_object, obj_bps_data, obj_rot_mat_ref, rest_pose_obj_nn_pts, transformed_obj_verts, rest_human_offsets, object_points)

                loss_consistency, loss_object, loss_fk = \
                    loss_dict['loss_consistency'], loss_dict['loss_object'], loss_dict['loss_fk']

                if loss_object is not None:
                    loss = loss_consistency + cfg.loss_w_obj_pts * loss_object + cfg.loss_w_fk * loss_fk
                else:
                    loss = loss_consistency

                if step % 10 == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"Epoch: {epoch}, Step: {step} / {len(dataloader)}   Loss: {loss.item()}, LR: {current_lr:.6f}", flush=True)
                    if cfg.use_tensorboard and rank == 0:
                        writer.add_scalar('Loss', loss.item(), epoch * len(dataloader) + step)
                        writer.add_scalar('Loss_consistency', loss_consistency.item(), epoch * len(dataloader) + step)
                        if loss_object is not None:
                            writer.add_scalar('Loss_object', loss_object.item(), epoch * len(dataloader) + step)
                            writer.add_scalar('Loss_fk', loss_fk.item(), epoch * len(dataloader) + step)
            else:
                loss_dict = trainer.p_losses(x_start, joints, mat, scene_flag, mask, t, text_clip_embedding, pelvis_goal, scene_goal, object_goal, \
                    need_scene, need_pelvis_dir, pi, end_pi, seg_len, need_pi, is_loco, is_object, obj_bps_data, obj_rot_mat_ref, rest_pose_obj_nn_pts, transformed_obj_verts, rest_human_offsets, object_points)

                loss, loss_object, loss_fk = \
                    loss_dict['loss'], loss_dict['loss_object'], loss_dict['loss_fk']
                    
                if loss_object is not None:
                    loss = loss + cfg.loss_w_obj_pts * loss_object + cfg.loss_w_fk * loss_fk

                if step % 10 == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"Epoch: {epoch}, Step: {step} / {len(dataloader)}   Loss: {loss.item()}, LR: {current_lr:.6f}", flush=True)
                    if cfg.use_tensorboard and rank == 0:
                        writer.add_scalar('Loss', loss.item(), epoch * len(dataloader) + step)
                        if loss_object is not None:
                            writer.add_scalar('Loss_object', loss_object.item(), epoch * len(dataloader) + step)
                            writer.add_scalar('Loss_fk', loss_fk.item(), epoch * len(dataloader) + step)

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
