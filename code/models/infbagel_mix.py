import math
import torch
from torch import nn
import torch.nn.functional as F
from vit_pytorch import ViT
from tqdm import tqdm
from utils import *
import pytorch3d.transforms as transforms

@torch.no_grad()
def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.
    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)

class Sampler:
    def __init__(self, device, mask_ind, emb_f, batch_size, channel, auto_regre_num, timesteps, ddim_timesteps, cm_timesteps, **kwargs):
        self.device = device
        self.mask_ind = mask_ind
        self.emb_f = emb_f
        self.batch_size = batch_size
        self.channel = channel
        self.auto_regre_num = auto_regre_num
        self.timesteps = timesteps
        self.ddim_timesteps = ddim_timesteps
        self.motion_len = kwargs.get('motion_len', None)
        self.scene_type = kwargs.get('scene_type', None)
        self.temp_voxel_num = kwargs.get('temp_voxel_num', 3)  # new param, controls number of temporal voxels, default 3 for backward compatibility
        self.get_scheduler()
        self.solver = DDIMSolver(self.alpha_cumprod.numpy(), self.timesteps, self.ddim_timesteps).to(self.device)
        self.cm_timesteps = cm_timesteps
        self.is_o0 = kwargs.get('is_o0', False)  # whether to use o0
        self.w = kwargs.get('w', 0)
        
    def set_dataset_and_model(self, dataset, student_model, teacher_model=None, target_model=None):
        self.dataset = dataset
        if dataset.load_scene:
            self.grid = dataset.create_meshgrid(batch_size=self.batch_size).to(self.device)
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.target_model = target_model
        nb_voxels = dataset.nb_voxels
        self.occ_idx = torch.arange(0, nb_voxels[1], 1).to(self.device)

    def get_scheduler(self):
        betas = linear_beta_schedule(timesteps=self.timesteps)

        # define alphas
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.betas = betas

        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))
        self.posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_mean_coef2 = (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)

        self.alpha_cumprod = alphas_cumprod
    
    def sample_cfg_scale_mixed(self, batch_size, device, uncond_prob=0.1, w_max=2.0):
        """Mixed sampling strategy: with 10% probability sample w=-1, with 90% probability sample uniformly from [0, w_max]
        
        Args:
            batch_size: batch size
            device: device
            uncond_prob: probability of unconditional generation
            w_max: maximum CFG scale for conditional generation
            
        Returns:
            w: CFG scale [batch_size, 1]
            is_uncond: flag for unconditional generation [batch_size]
        """
        is_uncond = torch.rand(batch_size) < uncond_prob
        
        # generate w values
        w = torch.zeros(batch_size, 1, device=device)
        w[is_uncond] = -1.0
        if (~is_uncond).any():
            w[~is_uncond] = torch.rand((~is_uncond).sum(), 1, device=device) * w_max
        
        return w, is_uncond
        
    def q_sample(self, x_start, t, noise):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def _get_temp_frame_indices(self, temp_voxel_num, seq_length=15):
        """
        Dynamically generate temporal frame indices based on temp_voxel_num
        
        Args:
            temp_voxel_num: required number of temporal voxels
            seq_length: total sequence length, default 15
        
        Returns:
            List[int]: list of temporal frame indices
        """
        if temp_voxel_num == 0:
            return []
        elif temp_voxel_num == 1:
            return [seq_length // 2]  # middle frame, i.e. [8]
        elif temp_voxel_num == 2:
            return [8, 15]
        elif temp_voxel_num == 3:
            return [5, 10, 15]  # original implementation, kept for backward compatibility
        else:
            # for other counts, distribute uniformly
            if temp_voxel_num > seq_length - 1:
                temp_voxel_num = seq_length - 1
            indices = []
            for i in range(temp_voxel_num):
                idx = (i + 1) * seq_length // (temp_voxel_num + 1)
                indices.append(min(idx, seq_length - 1))
            return indices

    def consistency_loss(self, x_start, joints, mat, scene_flag, mask, t, text_emb, pelvis_goal, hand_goal, object_goal, is_pick, need_scene, need_pelvis_dir, pi, end_pi, seq_length, need_pi, is_loco, is_object, obj_bps_data, obj_rot_mat_ref, rest_pose_obj_nn_pts, transformed_obj_verts, rest_human_offsets, object_points=None, noise=None, loss_type='l2'):
        update_ema(self.target_model.parameters(), self.student_model.parameters(), 0.95)

        if noise is None:
            noise = torch.randn_like(x_start)
        
        noise = noise.to(x_start.device, dtype=torch.float32)
        noise[mask] = 0.

        # Sample a random timestep for each image t_n ~ U[0, N - k - 1] without bias.
        topk = (self.timesteps // self.ddim_timesteps)
        index = torch.randint(0, self.ddim_timesteps, (x_start.shape[0],), device=x_start.device).long()
        
        # test
        # index = torch.zeros_like(index)
        # test
        
        start_timestep = self.solver.ddim_timesteps[index]
        timesteps = start_timestep - topk
        timesteps = torch.where(timesteps < 0, torch.zeros_like(timesteps), timesteps)

        inference_indices = np.linspace(
                    0, len(self.solver.ddim_timesteps), num=self.cm_timesteps, endpoint=False
                )
        inference_indices = np.floor(inference_indices).astype(np.int64)
        inference_indices = (
            torch.from_numpy(inference_indices).long().to(timesteps.device)
        )
        
        # Get boundary scalings for start_timesteps and (end) timesteps.
        c_skip_start, c_out_start = scalings_for_boundary_conditions(start_timestep)
        c_skip_start, c_out_start = [append_dims(x, x_start.ndim) for x in [c_skip_start, c_out_start]]
        
        c_skip, c_out = scalings_for_boundary_conditions(timesteps)
        c_skip, c_out = [append_dims(x, x_start.ndim) for x in [c_skip, c_out]]

        # Add noise to the latents according to the noise magnitude at each timestep
        x_start_noisy = self.q_sample(x_start=x_start, t=start_timestep, noise=noise)
        x_start_noisy[mask] = x_start[mask]
        x_start_noisy[torch.logical_not(is_object), :, 216:] = x_start[torch.logical_not(is_object), :, 216:]

        if self.dataset.load_scene:
            with torch.no_grad():
                # print(x_noisy.shape, joints.shape, mat.shape)
                x_orig = transform_points(self.dataset.denormalize_torch(x_start_noisy[:, :, :joints.shape[-1]]), mat)
                mat_for_query = mat.clone()
                target_ind = self.mask_ind if self.mask_ind != -1 else 0
                mat_for_query[:, :3, 3] = x_orig[:, self.emb_f, target_ind * 3: target_ind * 3 + 3]
                mat_for_query[:, 1, 3] = 0
                query_points = transform_points(self.grid, mat_for_query)
                occ = self.dataset.get_occ_for_points(query_points, object_points, scene_flag).float()

                nb_voxels = self.dataset.nb_voxels
                occ = occ.reshape(-1, nb_voxels[0], nb_voxels[1], nb_voxels[2]).float()

                if self.scene_type in ['plane_two', 'occ_two', 'occ_temp']:
                    mat_for_query_goal = mat.clone()
                    
                    # handle pelvis goal in the need_pelvis_dir case
                    pelvis_goal_copy = pelvis_goal.clone()
                    # handle pelvis goal in the not-is_loco case (static scene interaction), where pelvis is replaced by hand goal
                    hand_goal_copy = hand_goal.clone()

                    pelvis_goal_copy[torch.logical_not(is_loco)] = hand_goal_copy[torch.logical_not(is_loco)]
                    # handle pelvis goal in the is_loco case
                    # pelvis_goal_copy[is_loco] = pelvis_goal_copy[is_loco] / (torch.norm(pelvis_goal_copy[is_loco], dim=-1, keepdim=True) + 1e-6) * 0.8
                    pelvis_goal_orig = transform_points(pelvis_goal_copy.unsqueeze(1), mat).squeeze(1)
                    
                    # handle object goal in the is_object case - no rotation needed
                    object_goal_copy = object_goal.clone()
                    # object_goal_copy[is_object] = object_goal_copy[is_object] / (torch.norm(object_goal_copy[is_object], dim=-1, keepdim=True) + 1e-6) * 0.8
                    object_goal_orig = transform_points(object_goal_copy.unsqueeze(1), mat).squeeze(1)

                    # set goal position based on need_pelvis_dir and is_object
                    mat_for_query_goal[need_pelvis_dir, :3, 3] = pelvis_goal_orig[need_pelvis_dir] # need_pelvis_dir: inter_scene, is_loco, is_object
                    mat_for_query_goal[is_object, :3, 3] = object_goal_orig[is_object] # is_object: inter_object
                    mat_for_query_goal[torch.logical_not(torch.logical_or(need_pelvis_dir, is_object)), :3, 3] = mat_for_query[torch.logical_not(torch.logical_or(need_pelvis_dir, is_object)), :3, 3].clone()
                    mat_for_query_goal[:, 1, 3] = 0.
                    
                    query_points = transform_points(self.grid, mat_for_query_goal)
                    occ_goal = self.dataset.get_occ_for_points(query_points, None, scene_flag)
                    nb_voxels = self.dataset.nb_voxels
                    occ_goal = occ_goal.reshape(-1, nb_voxels[0], nb_voxels[1], nb_voxels[2]).float()

                    end_goal_pos = torch.zeros(self.batch_size, 2).to(self.device)
                    end_goal_pos[need_pelvis_dir] = pelvis_goal_copy[need_pelvis_dir].reshape(-1, 3)[:, [0, 2]]
                    end_goal_pos[is_object] = object_goal_copy[is_object].reshape(-1, 3)[:, [0, 2]]
                
                occ_pos = torch.zeros(0, self.batch_size, 2).to(self.device)
                occ_pos = torch.cat([occ_pos, end_goal_pos[None]], dim=0)
                
                occ_list = torch.zeros(0, nb_voxels[1], nb_voxels[0], nb_voxels[2]).to(self.device)
                occ_list = torch.cat([occ_list, occ], dim=0)
                occ_temp = None
                if self.scene_type == 'occ_temp':
                    object_points_temp = object_points.clone()
                    if self.is_o0 == False:
                        pred_obj_rot_mat_rel = x_start_noisy[:, :, 219:228].reshape(joints.shape[0], -1, 3, 3)
                        
                        pred_obj_rot_mat_rel_aa = transforms.matrix_to_axis_angle(pred_obj_rot_mat_rel) # [b, t, 3]
                        # std_per_dim = torch.tensor([0.5, 1.5, 0.5], device=pred_obj_rot_mat_rel_aa.device).view(1, 1, 3)
                        # perturb = torch.randn_like(pred_obj_rot_mat_rel_aa) * std_per_dim
                        # pred_obj_rot_mat_rel_aa = pred_obj_rot_mat_rel_aa + perturb
                        pred_obj_rot_mat_rel = transforms.axis_angle_to_matrix(pred_obj_rot_mat_rel_aa)
                        
                        obj_rot_mat_ref_temp = obj_rot_mat_ref.unsqueeze(1).repeat(1, pred_obj_rot_mat_rel.shape[1], 1, 1)
                        pred_obj_rot_mat = pred_obj_rot_mat_rel @ obj_rot_mat_ref_temp # [b, t, 3, 3]
                        pred_obj_rot_mat = pred_obj_rot_mat @ pred_obj_rot_mat[:, 0:1, :, :].transpose(2, 3)

                        pred_obj_trans = x_start_noisy[:, :, 216:219] # [b, t, 3]
                        pred_obj_trans = transform_points(self.dataset.denormalize_torch(pred_obj_trans, is_object=True), mat)
                        pred_obj_trans = pred_obj_trans - pred_obj_trans[:, 0:1, :]

                        # perturb = (torch.rand_like(pred_obj_trans) - 0.5) * 0.4  # ∈ [-0.2, 0.2]
                        # pred_obj_trans = pred_obj_trans + perturb
                    else:
                        pred_obj_rot_mat_rel = x_start[:, :, 219:228].reshape(joints.shape[0], -1, 3, 3)
                        
                        pred_obj_rot_mat_rel_aa = transforms.matrix_to_axis_angle(pred_obj_rot_mat_rel) # [b, t, 3]
                        std_per_dim = torch.tensor([0.5, 1.5, 0.5], device=pred_obj_rot_mat_rel_aa.device).view(1, 1, 3)
                        perturb = torch.randn_like(pred_obj_rot_mat_rel_aa) * std_per_dim
                        pred_obj_rot_mat_rel_aa = pred_obj_rot_mat_rel_aa + perturb
                        pred_obj_rot_mat_rel = transforms.axis_angle_to_matrix(pred_obj_rot_mat_rel_aa)
                        
                        obj_rot_mat_ref_temp = obj_rot_mat_ref.unsqueeze(1).repeat(1, pred_obj_rot_mat_rel.shape[1], 1, 1)
                        pred_obj_rot_mat = pred_obj_rot_mat_rel @ obj_rot_mat_ref_temp # [b, t, 3, 3]
                        pred_obj_rot_mat = pred_obj_rot_mat @ pred_obj_rot_mat[:, 0:1, :, :].transpose(2, 3)

                        pred_obj_trans = x_start[:, :, 216:219] # [b, t, 3]
                        pred_obj_trans = transform_points(self.dataset.denormalize_torch(pred_obj_trans, is_object=True), mat)
                        pred_obj_trans = pred_obj_trans - pred_obj_trans[:, 0:1, :]

                        perturb = (torch.rand_like(pred_obj_trans) - 0.5) * 0.4  # ∈ [-0.2, 0.2]
                        pred_obj_trans = pred_obj_trans + perturb

                    object_points_temp = object_points_temp.unsqueeze(1).repeat(1, pred_obj_rot_mat.shape[1], 1, 1) # [b, t, 1024, 3]
                    object_points_temp = torch.matmul(pred_obj_rot_mat, object_points_temp.transpose(-2,-1)).transpose(-2,-1) + pred_obj_trans.unsqueeze(-2) # [b, t, 1024, 3]

                    x_denorm = self.dataset.denormalize_torch(x_start[:, :, :joints.shape[-1]])
                    perturb = (torch.rand_like(x_denorm) - 0.5) * 0.2  # ∈ [-0.1, 0.1]
                    x_denorm = x_denorm + perturb

                    # dynamically obtain temporal frame indices
                    temp_indices = self._get_temp_frame_indices(self.temp_voxel_num)
                    
                    # only loop when temporal voxels exist
                    for i in temp_indices:
                        x0_orig = transform_points(x_denorm, mat)
                        mat_for_query = mat.clone()
                        target_ind = self.mask_ind if self.mask_ind != -1 else 0
                        
                        mat_for_query[:, :3, 3] = x0_orig[:, i, target_ind * 3: target_ind * 3 + 3]
                        mat_for_query[:, 1, 3] = 0
                        query_points = transform_points(self.grid, mat_for_query)
                        
                        occ_pos = torch.cat([occ_pos, x_denorm[:, i, [0, 2]][None]], dim=0)
                    
                        occ_temp = self.dataset.get_occ_for_points(query_points, object_points_temp[:, i, :, :], scene_flag)
                        nb_voxels = self.dataset.nb_voxels
                        occ_temp = occ_temp.reshape(-1, nb_voxels[0], nb_voxels[1], nb_voxels[2]).float()
                        occ_temp = occ_temp.permute(0, 2, 1, 3)

                        occ_list = torch.cat([occ_list, occ_temp], dim=0)

                if self.scene_type == 'occ':
                    occ = occ.permute(0, 2, 1, 3)
                elif self.scene_type == 'plane':
                    occ = occ.permute(0, 1, 3, 2)
                    occ_cnt = occ * self.occ_idx
                    occ = torch.argmax(occ_cnt, dim=-1).unsqueeze(1).float() / nb_voxels[1]
                elif self.scene_type == 'plane_two':
                    occ = occ.permute(0, 1, 3, 2)
                    occ_cnt = occ * self.occ_idx
                    occ = torch.argmax(occ_cnt, dim=-1).unsqueeze(1).float() / nb_voxels[1]

                    occ_goal = occ_goal.permute(0, 1, 3, 2)
                    occ_goal_cnt = occ_goal * self.occ_idx
                    occ_goal = torch.argmax(occ_goal_cnt, dim=-1).unsqueeze(1).float() / nb_voxels[1]
                    occ = torch.cat([occ, occ_goal], dim=1)
                elif self.scene_type == 'occ_two':
                    occ = occ.permute(0, 2, 1, 3)
                    occ_goal = occ_goal.permute(0, 2, 1, 3)
                    occ = torch.cat([occ, occ_goal], dim=1)
                elif self.scene_type == 'occ_temp':
                    occ = occ_goal.permute(0, 2, 1, 3)

        else:
            occ = None
        
        # sample CFG scale
        w, is_uncond = self.sample_cfg_scale_mixed(x_start.shape[0], x_start.device)
        
        # Student model prediction (with CFG scale)
        pred_x_0 = self.student_model(x_start_noisy, occ, start_timestep, text_emb, pelvis_goal, hand_goal, is_loco, need_scene, need_pelvis_dir, pi, end_pi, seq_length, need_pi, object_goal, is_object, obj_bps_data, occ_list, occ_pos, cfg_scale=w)

        sqrt_one_minus_alphas_cumprod_t = extract(
                self.sqrt_one_minus_alphas_cumprod, start_timestep, x_start_noisy.shape
            )
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, start_timestep, x_start_noisy.shape)
        
        noise_pred = (x_start_noisy - sqrt_alphas_cumprod_t * pred_x_0) / sqrt_one_minus_alphas_cumprod_t

        model_pred = pred_x_0

        model_pred = c_skip_start * x_start_noisy + c_out_start * model_pred

        # print('student pred_x_0', torch.norm(pred_x_0 - x_start))
        # print('student model_pred', torch.norm(model_pred - x_start))
        
        # print('self.solver.ddim_timesteps[inference_indices]', self.solver.ddim_timesteps[inference_indices])
        # print('start_timestep', start_timestep)
        # print('end_timesteps', end_timesteps)
        # print('timesteps', timesteps)

        # Use the ODE solver to predict the kth step in the augmented PF-ODE trajectory after
        with torch.no_grad():
            # conditional prediction
            cond_pred = self.teacher_model(x_start_noisy, occ, start_timestep, text_emb, pelvis_goal, hand_goal, 
                                          is_loco, need_scene, need_pelvis_dir, pi, end_pi, seq_length, need_pi, 
                                          object_goal, is_object, obj_bps_data, occ_list, occ_pos, 
                                          is_sample=True, is_uncondition=False)
            
            # unconditional prediction
            uncond_pred = self.teacher_model(x_start_noisy, occ, start_timestep, text_emb, pelvis_goal, hand_goal, 
                                            is_loco, need_scene, need_pelvis_dir, pi, end_pi, seq_length, need_pi, 
                                            object_goal, is_object, obj_bps_data, occ_list, occ_pos, 
                                            is_sample=True, is_uncondition=True)
            
            # unified CFG formula, automatically handles all w values
            # w = -1: teacher_pred_x0 = cond_pred + (-1) * (cond_pred - uncond_pred) = uncond_pred
            # w = 0:  teacher_pred_x0 = cond_pred + 0 * (cond_pred - uncond_pred) = cond_pred
            # w > 0:  teacher_pred_x0 = cond_pred + w * (cond_pred - uncond_pred) = CFG enhancement
            teacher_pred_x0 = cond_pred + w.unsqueeze(-1) * (cond_pred - uncond_pred)
            
            teacher_noise_pred = (x_start_noisy - sqrt_alphas_cumprod_t * teacher_pred_x0) / sqrt_one_minus_alphas_cumprod_t
            x_prev = self.solver.ddim_step(teacher_pred_x0, teacher_noise_pred, index)
            
            x_prev[mask] = x_start[mask].to(x_prev.dtype)
            x_prev[torch.logical_not(is_object), :, 216:] = x_start[torch.logical_not(is_object), :, 216:].to(x_prev.dtype)

            # Get target LCM prediction on x_prev, w, c, t_n
            target_pred_x0 = self.target_model(x_prev, occ, timesteps, text_emb, pelvis_goal, hand_goal, is_loco, need_scene, need_pelvis_dir, pi, end_pi, seq_length, need_pi, object_goal, is_object, obj_bps_data, occ_list, occ_pos)
            
            sqrt_one_minus_alphas_cumprod_t = extract(
                    self.sqrt_one_minus_alphas_cumprod, timesteps, x_prev.shape
                )
            sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, timesteps, x_prev.shape)
            
            target_noise_pred = (x_prev - sqrt_alphas_cumprod_t * target_pred_x0) / sqrt_one_minus_alphas_cumprod_t

            target = target_pred_x0

            target = c_skip * x_prev + c_out * target

        # Calculate loss
        mask_inv = torch.logical_not(mask)
        mask_inv[torch.logical_not(is_object), :, 216:] = False
        if loss_type == 'l1':
            loss = F.l1_loss(model_pred[mask_inv].float(), target[mask_inv].float())
        elif loss_type == 'l2':
            loss = F.mse_loss(model_pred[mask_inv].float(), target[mask_inv].float())
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(model_pred[mask_inv].float(), target[mask_inv].float())
        else:
            raise NotImplementedError()

        # add object loss (obj_rot_mat_ref, rest_pose_obj_nn_pts, transformed_obj_verts)
        if self.dataset.use_object_keypoints:
            hand_idx_28 = [20, 21, 25, 27]
            hand_idx_24 = [20, 21, 22, 23]
            foot_idx = [7, 8, 10, 11]
            
            gt_global_jpos = transform_points(self.dataset.denormalize_torch(joints), mat).reshape(joints.shape[0], -1, 28, 3)
            gt_global_hand_jpos = gt_global_jpos[:, :, hand_idx_28, :]
            gt_global_foot_jpos = gt_global_jpos[:, :, foot_idx, :]

            model_pred[mask] = x_start[mask]
            
            global_jpos = transform_points(self.dataset.denormalize_torch(model_pred[:, :, :84]), mat).reshape(joints.shape[0], -1, 28, 3)

            # FK to get joint positions.
            curr_seq_local_jpos = rest_human_offsets[:, None].repeat(1, global_jpos.shape[1], 1, 1) # [b, t, 24, 3]
            curr_seq_local_jpos = curr_seq_local_jpos.reshape(-1, 24, 3) # [b*t, 24, 3]
            curr_seq_local_jpos[:, 0, :] = global_jpos.reshape(-1, 28, 3)[:, 0, :]

            global_jrot_6d = model_pred[:, :, 84:216].reshape(joints.shape[0], -1, 22, 6)
            global_jrot_mat = transforms.rotation_6d_to_matrix(global_jrot_6d) # [b, t, 22, 3, 3]
            global_jrot_mat = mat[:, None, None, :3, :3] @ global_jrot_mat
            
            local_jrot_mat = self.dataset.quat_ik_torch(global_jrot_mat.reshape(-1, 22, 3, 3)) # [b*t, 22, 3, 3]
            _, human_jnts = self.dataset.quat_fk_torch(local_jrot_mat, curr_seq_local_jpos) # [b*t, 24, 3]
            human_jnts = human_jnts.reshape(joints.shape[0], -1, 24, 3) # [b, t, 24, 3]

            pred_global_hand_jpos = human_jnts[:, :, hand_idx_24, :]
            pred_global_foot_jpos = human_jnts[:, :, foot_idx, :] # [b, t, 4, 3]

            mask_fk = torch.ones(mask_inv.shape[0], self.dataset.max_window_size, 4, 3, dtype=torch.bool).to(mask_inv.device)
            mask_fk[:, :self.auto_regre_num, :, :] = False
            # print(torch.equal(mask_fk, mask_inv[:, :, :3*4].reshape(mask_inv.shape[0], -1, 4, 3)))
            fk_hand_loss = F.mse_loss(pred_global_hand_jpos[mask_fk], gt_global_hand_jpos[mask_fk])
            fk_foot_loss = F.mse_loss(pred_global_foot_jpos[mask_fk], gt_global_foot_jpos[mask_fk])
            loss_fk = fk_hand_loss + fk_foot_loss
            
            model_mean = model_pred # x_start
            pred_obj_rot_mat_rel = model_mean[:, :, 219:228].reshape(joints.shape[0], -1, 3, 3)
            obj_rot_mat_ref = obj_rot_mat_ref.unsqueeze(1).repeat(1, pred_obj_rot_mat_rel.shape[1], 1, 1)
            pred_obj_rot_mat = pred_obj_rot_mat_rel @ obj_rot_mat_ref # [b, t, 3, 3]

            pred_obj_trans = model_mean[:, :, 216:219] # [b, t, 3]
            pred_obj_trans = transform_points(self.dataset.denormalize_torch(pred_obj_trans, is_object=True), mat)

            rest_pose_obj_nn_pts = rest_pose_obj_nn_pts.unsqueeze(1).repeat(1, pred_obj_rot_mat.shape[1], 1, 1) # [b, t, 100, 3]
            pred_seq_obj_kpts = torch.matmul(pred_obj_rot_mat, rest_pose_obj_nn_pts.transpose(-2,-1)).transpose(-2,-1) + pred_obj_trans.unsqueeze(-2) # [b, t, 100, 3]
            
            # rest_pose_obj_normals = rest_pose_obj_normals.unsqueeze(1).repeat(1, pred_obj_rot_mat.shape[1], 1, 1) # [b, t, 100, 3]
            # pred_seq_obj_normals = torch.matmul(pred_obj_rot_mat, rest_pose_obj_normals.transpose(-2,-1)).transpose(-2,-1) # [b, t, 100, 3]
            
            # transformed_obj_verts = self.dataset.normalize_torch(transformed_obj_verts, is_object=True)
            # pred_seq_obj_kpts = self.dataset.normalize_torch(pred_seq_obj_kpts, is_object=True)
            
            mask_points = torch.ones(mask_inv.shape[0], self.dataset.max_window_size, 100, 3, dtype=torch.bool).to(mask_inv.device)
            mask_points[:, :self.auto_regre_num, :, :] = False
            mask_points[torch.logical_not(is_object)] = False
            
            if loss_type == 'l1':
                loss_object = F.l1_loss(transformed_obj_verts[mask_points], pred_seq_obj_kpts[mask_points])
            elif loss_type == 'l2':
                loss_object = F.mse_loss(transformed_obj_verts[mask_points], pred_seq_obj_kpts[mask_points])
            elif loss_type == "huber":
                loss_object = F.smooth_l1_loss(transformed_obj_verts[mask_points], pred_seq_obj_kpts[mask_points])
            else:
                raise NotImplementedError()
            
            # --- Velocity Loss Calculation ---
            
            # 1. Human Velocity Loss
            # First, get ground truth 3D joint positions by applying FK to x_start to ensure consistency
            gt_global_jpos_for_fk = transform_points(self.dataset.denormalize_torch(x_start[:, :, :84]), mat).reshape(joints.shape[0], -1, 28, 3)
            gt_curr_seq_local_jpos = rest_human_offsets[:, None].repeat(1, gt_global_jpos_for_fk.shape[1], 1, 1)
            gt_curr_seq_local_jpos = gt_curr_seq_local_jpos.reshape(-1, 24, 3)
            gt_curr_seq_local_jpos[:, 0, :] = gt_global_jpos_for_fk.reshape(-1, 28, 3)[:, 0, :]
            
            gt_global_jrot_6d = x_start[:, :, 84:216].reshape(joints.shape[0], -1, 22, 6)
            gt_global_jrot_mat = transforms.rotation_6d_to_matrix(gt_global_jrot_6d)
            gt_global_jrot_mat = mat[:, None, None, :3, :3] @ gt_global_jrot_mat
            
            gt_local_jrot_mat = self.dataset.quat_ik_torch(gt_global_jrot_mat.reshape(-1, 22, 3, 3))
            _, gt_human_jnts = self.dataset.quat_fk_torch(gt_local_jrot_mat, gt_curr_seq_local_jpos)
            gt_human_jnts = gt_human_jnts.reshape(joints.shape[0], -1, 24, 3)

            # Calculate velocity for predicted (human_jnts) and ground truth (gt_human_jnts)
            vel_human_pred = human_jnts[:, 1:] - human_jnts[:, :-1]
            vel_human_gt = gt_human_jnts[:, 1:] - gt_human_jnts[:, :-1]
            loss_vel_human = F.mse_loss(vel_human_pred, vel_human_gt)

            # 2. Object Velocity Loss
            # Calculate velocity for predicted (pred_seq_obj_kpts) and ground truth (transformed_obj_verts)
            vel_obj_pred = pred_seq_obj_kpts[:, 1:] - pred_seq_obj_kpts[:, :-1]
            vel_obj_gt = transformed_obj_verts[:, 1:] - transformed_obj_verts[:, :-1]
            loss_vel_obj = F.mse_loss(vel_obj_pred, vel_obj_gt)
            
            # 3. Total Velocity Loss
            loss_vel = loss_vel_human + loss_vel_obj

        else: 
            loss_object = None
            loss_fk = None
            loss_vel = None

        return dict(loss_consistency=loss, loss_object=loss_object, loss_fk=loss_fk, loss_vel=loss_vel)

    @torch.no_grad()
    def cm_sample_loop(self, fixed_points, mat, scene_flag, text_emb, pelvis_goal, hand_goal, object_goal, \
                    is_pick, need_scene, need_pelvis_dir, pi, end_pi, seq_length, need_pi, is_loco, is_object, obj_bps_data, object_points, obj_rot_mat_ref, obj_rest_verts, obj_vert_normals, seq_name_dict, human_dict, guidance_fn, guidance_scale, object_only=False, w=None, obj_rot_mat_prefix=None):
        self.batch_size = fixed_points.shape[0]
        device = next(self.student_model.parameters()).device
        shape = (self.batch_size, self.dataset.max_window_size, self.channel)
        points = torch.randn(shape, device=device, dtype=torch.float32)

        if self.auto_regre_num > 0:
            self.set_fixed_points(points, None, fixed_points, mat, joint_id=self.mask_ind, fix_mode=True, fix_goal=False)
        imgs = []
        occs = []
        x0 = []
        inference_indices = np.linspace(-1, len(self.solver.ddim_timesteps) - 1, num=self.cm_timesteps + 1, endpoint=True)
        inference_indices = (
                    torch.from_numpy(np.floor(inference_indices).astype(np.int64)).long().to(device)
                )
        inference_indices = inference_indices[1:]
        t_index = len(inference_indices) - 1
        x0.append(points)
        for i in tqdm(reversed(inference_indices), desc='sampling loop time step', total=len(inference_indices)):
            model_used = self.student_model
            points, occ, pred_x_0 = self.cm_sample(model_used, x0[-1], points, fixed_points, mat, scene_flag,
                                        torch.full((self.batch_size,), i, device=device, dtype=torch.long), t_index,
                                        text_emb, pelvis_goal, hand_goal, object_goal, is_pick, need_scene, 
                                        need_pelvis_dir, pi, end_pi, seq_length, need_pi, is_loco, is_object, obj_bps_data, object_points, obj_rot_mat_ref, obj_rest_verts, obj_vert_normals, seq_name_dict, human_dict, obj_rot_mat_prefix, guidance_fn, guidance_scale, object_only, w)
            if self.auto_regre_num > 0:
                self.set_fixed_points(points, None, fixed_points, mat, joint_id=self.mask_ind, fix_mode=True, fix_goal=False)

            points_orig = points
            imgs.append(points_orig)
            x0.append(pred_x_0)
            if occ is not None:
                occs.append(occ.cpu().numpy())

            t_index -= 1

        return imgs, occs

    @torch.no_grad()
    def cm_sample(self, model, x0, x, fixed_points, mat, scene_flag, t, t_index,
                 text_emb, pelvis_goal, hand_goal, object_goal, is_pick, need_scene,
                 need_pelvis_dir, pi, end_pi, seq_length, need_pi, is_loco, is_object, obj_bps_data, object_points, obj_rot_mat_ref, obj_rest_verts, obj_vert_normals, seq_name_dict, human_dict, obj_rot_mat_prefix, guidance_fn, guidance_scale, object_only=False, w=None):
        if self.dataset.load_scene:
            x_orig = transform_points(self.dataset.denormalize_torch(x[:, :, :84]), mat)
            mat_for_query = mat.clone()
            target_ind = self.mask_ind if self.mask_ind != -1 else 0
            mat_for_query[:, :3, 3] = x_orig[:, self.emb_f, target_ind * 3: target_ind * 3 + 3]
            mat_for_query[:, 1, 3] = 0
            
            self.grid = self.dataset.create_meshgrid(batch_size=self.batch_size).to(self.device)

            query_points = transform_points(self.grid, mat_for_query)
            occ = self.dataset.get_occ_for_points(query_points, object_points, scene_flag)
            nb_voxels = self.dataset.nb_voxels
            occ = occ.reshape(-1, nb_voxels[0], nb_voxels[1], nb_voxels[2]).float()
            
            if object_only:
                occ[occ == 1] = 0.

            if torch.logical_not(is_object).any():
                occ[torch.logical_not(is_object)][occ == 2] = 1.

            if self.scene_type in ['plane_two', 'occ_two', 'occ_temp']:
                mat_for_query_goal = mat.clone()
                
                # handle pelvis goal in the is_loco case
                pelvis_goal_copy = pelvis_goal.clone()
                hand_goal_copy = hand_goal.clone()

                pelvis_goal_copy[torch.logical_not(is_loco)] = hand_goal_copy[torch.logical_not(is_loco)]
                # pelvis_goal_copy[is_loco] = pelvis_goal_copy[is_loco] / (
                #             torch.norm(pelvis_goal_copy[is_loco], dim=-1, keepdim=True) + 1e-6) * 0.8
                pelvis_goal_orig = transform_points(pelvis_goal_copy.reshape(pelvis_goal_copy.shape[0], 1, 3), mat).squeeze(1)

                # handle object goal in the is_object case - no rotation needed
                object_goal_copy = object_goal.clone()
                object_goal_orig = transform_points(object_goal_copy.reshape(object_goal_copy.shape[0], 1, 3), mat).squeeze(1)

                mat_for_query_goal[need_pelvis_dir, :3, 3] = pelvis_goal_orig[need_pelvis_dir]
                mat_for_query_goal[is_object, :3, 3] = object_goal_orig[is_object]
                mat_for_query_goal[torch.logical_not(torch.logical_or(need_pelvis_dir, is_object)), :3, 3] = mat_for_query[
                                                                                torch.logical_not(torch.logical_or(need_pelvis_dir, is_object)), :3,
                                                                                3].clone()
                mat_for_query_goal[:, 1, 3] = 0.
                query_points_goal = transform_points(self.grid, mat_for_query_goal)
                occ_goal = self.dataset.get_occ_for_points(query_points_goal, object_points, scene_flag)

                if object_only:
                    occ_goal[occ_goal == 1] = 0.

                if torch.logical_not(is_object).any():
                    occ_goal[torch.logical_not(is_object)][occ_goal == 2] = 1.

                nb_voxels = self.dataset.nb_voxels
                occ_goal = occ_goal.reshape(-1, nb_voxels[0], nb_voxels[1], nb_voxels[2]).float()

                end_goal_pos = torch.zeros(self.batch_size, 2).to(self.device)
                end_goal_pos[need_pelvis_dir] = pelvis_goal_copy[need_pelvis_dir].reshape(-1, 3)[:, [0, 2]]
                end_goal_pos[is_object] = object_goal_copy[is_object].reshape(-1, 3)[:, [0, 2]]
            
            occ_pos = torch.zeros(0, self.batch_size, 2).to(self.device)
            occ_pos = torch.cat([occ_pos, end_goal_pos[None]], dim=0)
                
            occ_list = torch.zeros(0, nb_voxels[1], nb_voxels[0], nb_voxels[2]).to(self.device)
            occ_list = torch.cat([occ_list, occ], dim=0)
            occ_temp = None
            if self.scene_type == 'occ_temp':
                if self.dataset.vis:
                    # object_rot_mat = x0[:, :, 219:228].reshape(x.shape[0], -1, 3, 3)
                    # object_trans_orig = x0[:, :, 216:219] # [b, t, 3]
                    object_rot_mat = x[:, :, 219:228].reshape(x.shape[0], -1, 3, 3)
                    object_trans_orig = x[:, :, 216:219] # [b, t, 3]
                    object_trans_orig = transform_points(self.dataset.denormalize_torch(object_trans_orig, is_object=True), mat)

                    obj_name = seq_name_dict[0].split('_')[1]
                    pred_obj_rot_mat_seg = (obj_rot_mat_prefix[None] @ object_rot_mat[:, :, :].reshape(-1, 3, 3) @ obj_rot_mat_ref).reshape(-1, 3, 3)
                    pred_seq_com_pos_seg = object_trans_orig[:, :, :].reshape(-1, 3)
                    obj_rest_verts_seg = load_object_geometry_w_rest_geo(pred_obj_rot_mat_seg, pred_seq_com_pos_seg, obj_rest_verts[obj_name])
                    indices = torch.randperm(obj_rest_verts_seg.shape[1])[:1024]
                    object_points_temp = obj_rest_verts_seg[:, indices, :].reshape(1, -1, 1024, 3)
                else:
                    object_points_temp = object_points.clone()
                    pred_obj_rot_mat_rel = x[:, :, 219:228].reshape(x.shape[0], -1, 3, 3)
                    
                    obj_rot_mat_ref_temp = obj_rot_mat_ref
                    pred_obj_rot_mat = pred_obj_rot_mat_rel @ obj_rot_mat_ref_temp # [b, t, 3, 3]
                    pred_obj_rot_mat = pred_obj_rot_mat @ pred_obj_rot_mat[:, 0:1, :, :].transpose(2, 3)

                    pred_obj_trans = x[:, :, 216:219] # [b, t, 3]
                    pred_obj_trans = transform_points(self.dataset.denormalize_torch(pred_obj_trans, is_object=True), mat)
                    pred_obj_trans = pred_obj_trans - pred_obj_trans[:, 0:1, :]

                    object_points_temp = object_points_temp.unsqueeze(1).repeat(1, pred_obj_rot_mat.shape[1], 1, 1) # [b, t, 1024, 3]
                    object_points_temp = torch.matmul(pred_obj_rot_mat, object_points_temp.transpose(-2,-1)).transpose(-2,-1) + pred_obj_trans.unsqueeze(-2) # [b, t, 1024, 3]

                x_denorm = self.dataset.denormalize_torch(x0[:, :, :84])
                    
                # dynamically obtain temporal frame indices
                temp_indices = self._get_temp_frame_indices(self.temp_voxel_num)
                
                # only loop when temporal voxels exist
                for i in temp_indices:
                    x0_orig = transform_points(x_denorm, mat)
                    mat_for_query = mat.clone()
                    target_ind = self.mask_ind if self.mask_ind != -1 else 0
                    mat_for_query[:, :3, 3] = x0_orig[:, i, target_ind * 3: target_ind * 3 + 3]
                    mat_for_query[:, 1, 3] = 0
                    query_points = transform_points(self.grid, mat_for_query)
                    
                    occ_pos = torch.cat([occ_pos, x_denorm[:, i, [0, 2]][None]], dim=0)

                    occ_temp = self.dataset.get_occ_for_points(query_points, object_points_temp[:, i, :, :], scene_flag)
                    
                    if object_only:
                        occ_temp[occ_temp == 1] = 0.

                    if torch.logical_not(is_object).any():
                        occ_temp[torch.logical_not(is_object)][occ_temp == 2] = 1.
                        
                    nb_voxels = self.dataset.nb_voxels
                    occ_temp = occ_temp.reshape(-1, nb_voxels[0], nb_voxels[1], nb_voxels[2]).float()
                    occ_temp = occ_temp.permute(0, 2, 1, 3)

                    occ_list = torch.cat([occ_list, occ_temp], dim=0)

            if self.scene_type == 'occ':
                occ = occ.permute(0, 2, 1, 3)
            elif self.scene_type == 'plane':
                occ = occ.permute(0, 1, 3, 2)
                occ_cnt = occ * self.occ_idx
                occ = torch.argmax(occ_cnt, dim=-1).unsqueeze(1).float() / nb_voxels[1]
            elif self.scene_type == 'plane_two':
                occ = occ.permute(0, 1, 3, 2)
                occ_cnt = occ * self.occ_idx
                occ = torch.argmax(occ_cnt, dim=-1).unsqueeze(1).float() / nb_voxels[1]

                occ_goal = occ_goal.permute(0, 1, 3, 2)
                occ_goal_cnt = occ_goal * self.occ_idx
                occ_goal = torch.argmax(occ_goal_cnt, dim=-1).unsqueeze(1).float() / nb_voxels[1]
                occ = torch.cat([occ, occ_goal], dim=1)
            elif self.scene_type == 'occ_two':
                occ = occ.permute(0, 2, 1, 3)
                occ_goal = occ_goal.permute(0, 2, 1, 3)
                occ = torch.cat([occ, occ_goal], dim=1)
            elif self.scene_type == 'occ_temp':
                occ = occ_goal.permute(0, 2, 1, 3)

        else:
            occ = None

        # if w is None, set the w value based on t_index
        if w is None:
            is_uncondition = False
            w = torch.zeros((self.batch_size, 1), device=x.device)
        elif isinstance(w, (int, float)):
            if w == -1:
                is_uncondition = True
            else:
                is_uncondition = False
            w = torch.full((self.batch_size, 1), w, device=x.device)
        
        if t_index > 0:
            start_timestep = self.solver.ddim_timesteps[t]
            model_output = model(x, occ, start_timestep, text_emb, pelvis_goal, hand_goal, is_loco, need_scene, need_pelvis_dir, pi, end_pi, seq_length, need_pi, object_goal, is_object, obj_bps_data, occ_list, occ_pos, is_sample=True, is_uncondition=is_uncondition, cfg_scale=w)
            
            # uncond_model_output = model(x, occ, start_timestep, text_emb, pelvis_goal, hand_goal, is_pick, need_scene, need_pelvis_dir, pi, end_pi, seq_length, need_pi, object_goal, is_object, obj_bps_data, occ_list, occ_pos, is_sample=True, is_uncondition=True)

            # model_output = cond_model_output + self.w * (cond_model_output - uncond_model_output)

            c_skip, c_out = scalings_for_boundary_conditions(start_timestep)
            c_skip, c_out = [append_dims(item, x.ndim) for item in [c_skip, c_out]]
            
            pred_x_0 = c_skip * x + c_out * model_output
            self.set_fixed_points(pred_x_0, None, fixed_points, mat, joint_id=self.mask_ind, fix_mode=True, fix_goal=False)
            
            noise = torch.randn(x.shape).to(x.device)

            print('start_timestep', start_timestep[0])

            inference_indices = np.linspace(
                0, len(self.solver.ddim_timesteps), num=self.cm_timesteps, endpoint=False
            )
            inference_indices = np.floor(inference_indices).astype(np.int64)
            inference_indices = (
                torch.from_numpy(inference_indices).long().to(self.solver.ddim_timesteps.device)
            )
            expanded_timestep_index = t.unsqueeze(1).expand(
                -1, inference_indices.size(0)
            )
            last_valid_index = (expanded_timestep_index >= inference_indices).flip(dims=[1]).long().argmax(dim=1)
            last_valid_index = inference_indices.size(0) - 1 - last_valid_index
            timestep_index = inference_indices[last_valid_index]
            alpha_cumprod_prev = extract_into_tensor(
                self.solver.ddim_alpha_cumprods_prev, timestep_index, pred_x_0.shape
            ).float()
            x_prev = alpha_cumprod_prev.sqrt() * pred_x_0 + (1.0 - alpha_cumprod_prev).sqrt() * noise
            
            if is_object.any():
                with torch.enable_grad():
                    x_start = pred_x_0.detach().requires_grad_(True)
                    end_timesteps = self.solver.ddim_timesteps_prev[timestep_index]

                    global_jpos = x_start[:, :, :84].reshape(self.batch_size, self.dataset.max_window_size, 84)
                    global_jpos = transform_points(self.dataset.denormalize_torch(global_jpos), mat).reshape(self.batch_size, self.dataset.max_window_size, 28, 3)

                    # FK to get joint positions.
                    rest_human_offsets, transl, betas, gender = human_dict['rest_human_offsets'], human_dict['transl'], human_dict['betas'], human_dict['gender']
                    
                    curr_seq_local_jpos = rest_human_offsets # [b, t, 24, 3]
                    curr_seq_local_jpos = curr_seq_local_jpos.reshape(-1, 24, 3) # [b*t, 24, 3]
                    curr_seq_local_jpos[:, 0, :] = global_jpos.reshape(-1, 28, 3)[:, 0, :]

                    global_jrot_6d = x_start[:, :, 84:216].reshape(self.batch_size, self.dataset.max_window_size, 22, 6)
                    global_jrot_mat = transforms.rotation_6d_to_matrix(global_jrot_6d) # [b, t, 22, 3, 3]
                    global_jrot_mat = mat[:, None, None, :3, :3] @ global_jrot_mat

                    local_jrot_mat = self.dataset.quat_ik_torch(global_jrot_mat.reshape(-1, 22, 3, 3)) # [b*t, 22, 3, 3]
                    _, human_jnts = self.dataset.quat_fk_torch(local_jrot_mat, curr_seq_local_jpos) # [b*t, 24, 3]
                    human_jnts = human_jnts.reshape(self.batch_size, -1, 24, 3) # [b, t, 24, 3]

                    # transl, betas = transl.reshape(-1, 3), betas.reshape(-1, 16)
                    
                    # root_trans = yup_to_zup(global_jpos.reshape(-1, 28, 3)[:, 0, :] + transl)
                    # pose_pred = yup_to_zup(transforms.matrix_to_axis_angle(local_jrot_mat).reshape(-1, 22, 3))
                    
                    # verts, joints = run_smplx_model(pose_pred, root_trans, betas, 'male', joints_ind=None)
                    # verts, joints = zup_to_yup(verts), zup_to_yup(joints)
                    # verts = verts.reshape(self.batch_size, self.dataset.max_window_size, -1, 3)

                    pred_seq_com_pos = x_start[:, :, 216:219].reshape(self.batch_size, self.dataset.max_window_size, 3)
                    pred_seq_com_pos = transform_points(self.dataset.denormalize_torch(pred_seq_com_pos, is_object=True), mat)

                    object_rot_mat = x_start[:, :, 219:228].reshape(self.batch_size, self.dataset.max_window_size, 3, 3) # B X 16 X 3 X 3

                    if self.dataset.vis:
                        pred_obj_rot_mat = (obj_rot_mat_prefix @ object_rot_mat.reshape(self.batch_size, -1, 3, 3) @ obj_rot_mat_ref)
                    else:
                        pred_obj_rot_mat = (object_rot_mat.reshape(self.batch_size, -1, 3, 3) @ obj_rot_mat_ref)
                    
                    contact_labels = x_start[:, :, 228:232].reshape(self.batch_size, self.dataset.max_window_size, 4)

                    obj_verts = torch.zeros(0, self.dataset.max_window_size, 10000, 3).to(self.device)
                    obj_normals = torch.zeros(0, self.dataset.max_window_size, 10000, 3).to(self.device)

                    for seg_id in range(self.batch_size):
                        obj_name = seq_name_dict[seg_id].split('_')[1]
                        pred_obj_rot_mat_seg = pred_obj_rot_mat[seg_id].reshape(-1, 3, 3)
                        pred_seq_com_pos_seg = pred_seq_com_pos[seg_id].reshape(-1, 3)
                        obj_rest_verts_seg, obj_rest_normals_seg = load_object_geometry_w_rest_geo_and_normals(pred_obj_rot_mat_seg, pred_seq_com_pos_seg, obj_rest_verts[obj_name], obj_vert_normals[obj_name])
                        obj_rest_verts_seg = obj_rest_verts_seg.reshape(1, self.dataset.max_window_size, -1, 3) # 1 X T X Nv X 3
                        obj_rest_normals_seg = obj_rest_normals_seg.reshape(1, self.dataset.max_window_size, -1, 3) # 1 X T X Nv X 3
                        num_obj_verts = obj_rest_verts_seg.shape[2]
                        if num_obj_verts > 10000:
                            # randomly select indices of 10000 points
                            indices = torch.randperm(num_obj_verts)[:10000]
                            obj_rest_verts_seg = obj_rest_verts_seg[:, :, indices, :].reshape(1, self.dataset.max_window_size, 10000, 3)
                            obj_rest_normals_seg = obj_rest_normals_seg[:, :, indices, :].reshape(1, self.dataset.max_window_size, 10000, 3)
                        obj_verts = torch.cat([obj_verts, obj_rest_verts_seg], dim=0)
                        obj_normals = torch.cat([obj_normals, obj_rest_normals_seg], dim=0)

                    assert obj_verts.shape[0] == self.batch_size
                    
                    # loss, penetration_loss = guidance_fn(verts, human_jnts, obj_verts, obj_normals, pred_seq_com_pos, pred_obj_rot_mat, contact_labels)
                    # is_penetrating, nearest_free_points = self.dataset.get_nearest_free_voxel(human_jnts, scene_flag)
                    # loss += F.mse_loss(human_jnts, nearest_free_points) * 1000000000 # * 1200000000
                    # if is_penetrating.any():
                        # print(loss)

                    # penetration_loss += F.mse_loss(obj_verts[:, 1:], obj_verts[:, :-1]) * 100 * 50
                    # penetration_loss += F.mse_loss(human_jnts[:, 1:], human_jnts[:, :-1]) * 100 * 50

                    # gradient = torch.autograd.grad(-loss, x_start, retain_graph=True)[0] * 0.01
                    # penetration_gradient = torch.autograd.grad(-penetration_loss, x_start)[0]
                    
                    # alpha_cumprod = extract(self.alpha_cumprod, start_timestep, x_start.shape)
                    
                    # print('(1 - alpha_cumprod) * (alpha_cumprod_prev / alpha_cumprod).sqrt()', ((1 - alpha_cumprod) * (alpha_cumprod_prev / alpha_cumprod).sqrt())[0])
                    
                    # model_pred = model_pred + gradient * (1 - alpha_cumprod)

                    # model_pred[:, :, 216:] = model_pred[:, :, 216:] + penetration_gradient[:, :, 216:] * (1 - alpha_cumprod) * 0.00075
                    # model_pred[:, :, :216] = model_pred[:, :, :216] + penetration_gradient[:, :, :216] * (1 - alpha_cumprod) * 0.005

                    # sqrt_one_minus_alphas_cumprod_t = extract(
                    #         self.sqrt_one_minus_alphas_cumprod, start_timestep, x.shape
                    #     )
                    # sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, start_timestep, x.shape)
                    
                    # noise_pred = (x - sqrt_alphas_cumprod_t * pred_x_0) / sqrt_one_minus_alphas_cumprod_t

                    loss = guidance_fn(human_jnts, obj_verts, pred_seq_com_pos, pred_obj_rot_mat, contact_labels, scene_flag, self.dataset.get_nearest_free_voxel)
                    print(loss)

                    gradient = torch.autograd.grad(-loss, x_start, retain_graph=True)[0] * guidance_scale
                    # penetration_gradient = torch.autograd.grad(-penetration_loss, x_start)[0]
                    
                    alpha_cumprod = extract(self.alpha_cumprod, end_timesteps, x_start.shape)
                    
                    # print('(1 - alpha_cumprod) * (alpha_cumprod_prev / alpha_cumprod).sqrt()', ((1 - alpha_cumprod) * (alpha_cumprod_prev / alpha_cumprod).sqrt())[0])
                    
                    x_prev = x_prev + gradient # * (1 - alpha_cumprod)
            else:
                with torch.enable_grad():
                    x_start = pred_x_0.detach().requires_grad_(True)
                    end_timesteps = self.solver.ddim_timesteps_prev[timestep_index]

                    global_jpos = x_start[:, :, :84].reshape(self.batch_size, self.dataset.max_window_size, 84)
                    global_jpos = transform_points(self.dataset.denormalize_torch(global_jpos), mat).reshape(self.batch_size, self.dataset.max_window_size, 28, 3)

                    # FK to get joint positions.
                    rest_human_offsets, transl, betas, gender = human_dict['rest_human_offsets'], human_dict['transl'], human_dict['betas'], human_dict['gender']
                    
                    curr_seq_local_jpos = rest_human_offsets # [b, t, 24, 3]
                    curr_seq_local_jpos = curr_seq_local_jpos.reshape(-1, 24, 3) # [b*t, 24, 3]
                    curr_seq_local_jpos[:, 0, :] = global_jpos.reshape(-1, 28, 3)[:, 0, :]

                    global_jrot_6d = x_start[:, :, 84:216].reshape(self.batch_size, self.dataset.max_window_size, 22, 6)
                    global_jrot_mat = transforms.rotation_6d_to_matrix(global_jrot_6d) # [b, t, 22, 3, 3]
                    global_jrot_mat = mat[:, None, None, :3, :3] @ global_jrot_mat

                    local_jrot_mat = self.dataset.quat_ik_torch(global_jrot_mat.reshape(-1, 22, 3, 3)) # [b*t, 22, 3, 3]
                    _, human_jnts = self.dataset.quat_fk_torch(local_jrot_mat, curr_seq_local_jpos) # [b*t, 24, 3]
                    human_jnts = human_jnts.reshape(self.batch_size, -1, 24, 3) # [b, t, 24, 3]

                    loss = guidance_fn(human_jnts, scene_flag, self.dataset.get_nearest_free_voxel)
                    print(loss)

                    gradient = torch.autograd.grad(-loss, x_start, retain_graph=True)[0] * guidance_scale
                    # penetration_gradient = torch.autograd.grad(-penetration_loss, x_start)[0]
                    
                    alpha_cumprod = extract(self.alpha_cumprod, end_timesteps, x_start.shape)
                    
                    # print('(1 - alpha_cumprod) * (alpha_cumprod_prev / alpha_cumprod).sqrt()', ((1 - alpha_cumprod) * (alpha_cumprod_prev / alpha_cumprod).sqrt())[0])
                    
                    x_prev = x_prev + gradient # * (1 - alpha_cumprod)

        else:
            start_timestep = self.solver.ddim_timesteps[t]

            c_skip, c_out = scalings_for_boundary_conditions(start_timestep)
            c_skip, c_out = [append_dims(item, x.ndim) for item in [c_skip, c_out]]

            pred_x_0 = self.student_model(x, occ, start_timestep, text_emb, pelvis_goal, hand_goal, is_loco, need_scene, need_pelvis_dir, pi, end_pi, seq_length, need_pi, object_goal, is_object, obj_bps_data, occ_list, occ_pos, is_sample=True, is_uncondition=is_uncondition, cfg_scale=w)

            pred_x_0 = c_skip * x + c_out * pred_x_0
        
            noise_pred = torch.randn(x.shape).to(x.device)

            print('start_timestep', start_timestep[0])

            x_prev, end_timesteps = self.solver.ddim_style_multiphase_pred(
                    pred_x_0, noise_pred, t, self.cm_timesteps
                )

        # print('end_timesteps', end_timesteps[0])

        return x_prev.float(), occ, pred_x_0

    @torch.no_grad()
    def ddim_sample_loop(self, fixed_points, mat, scene_flag, text_emb, pelvis_goal, hand_goal, object_goal, \
                    is_pick, need_scene, need_pelvis_dir, pi, end_pi, seq_length, need_pi, is_loco, is_object, obj_bps_data, object_points, obj_rot_mat_ref, obj_rest_verts, seq_name_dict, rest_human_offsets, guidance_fn, guidance_scale, object_only=False):
        self.batch_size = fixed_points.shape[0]
        device = next(self.student_model.parameters()).device
        shape = (self.batch_size, self.dataset.max_window_size, self.channel)
        points = torch.randn(shape, device=device, dtype=torch.float32)

        if self.auto_regre_num > 0:
            self.set_fixed_points(points, None, fixed_points, mat, joint_id=self.mask_ind, fix_mode=True, fix_goal=False)
        imgs = []
        occs = []

        for i in tqdm(reversed(range(0, self.ddim_timesteps)), desc='sampling loop time step', total=self.ddim_timesteps):
            model_used = self.student_model

            points, occ = self.ddim_sample(model_used, points, fixed_points, mat, scene_flag,
                                        torch.full((self.batch_size,), i, device=device, dtype=torch.long), i,
                                        text_emb, pelvis_goal, hand_goal, object_goal, is_pick, need_scene, 
                                        need_pelvis_dir, pi, end_pi, seq_length, need_pi, is_loco, is_object, obj_bps_data, object_points, obj_rot_mat_ref, obj_rest_verts, seq_name_dict, rest_human_offsets, guidance_fn, guidance_scale, object_only)
            if self.auto_regre_num > 0:
                self.set_fixed_points(points, None, fixed_points, mat, joint_id=self.mask_ind, fix_mode=True, fix_goal=False)

            points_orig = points
            
            if occ is not None:
                occs.append(occ.cpu().numpy())
            imgs.append(points_orig)

        return imgs, occs

    @torch.no_grad()
    def ddim_sample(self, model, x, fixed_points, mat, scene_flag, t, t_index,
                 text_emb, pelvis_goal, hand_goal, object_goal, is_pick, need_scene,
                 need_pelvis_dir, pi, end_pi, seq_length, need_pi, is_loco, is_object, obj_bps_data, object_points, obj_rot_mat_ref, obj_rest_verts, seq_name_dict, rest_human_offsets, guidance_fn, guidance_scale, object_only=False):
        if self.dataset.load_scene:
            x_orig = transform_points(self.dataset.denormalize_torch(x[:, :, :84]), mat)
            mat_for_query = mat.clone()
            target_ind = self.mask_ind if self.mask_ind != -1 else 0
            mat_for_query[:, :3, 3] = x_orig[:, self.emb_f, target_ind * 3: target_ind * 3 + 3]
            mat_for_query[:, 1, 3] = 0
            
            self.grid = self.dataset.create_meshgrid(batch_size=self.batch_size).to(self.device)

            query_points = transform_points(self.grid, mat_for_query)
            occ = self.dataset.get_occ_for_points(query_points, object_points, scene_flag)
            nb_voxels = self.dataset.nb_voxels
            occ = occ.reshape(-1, nb_voxels[0], nb_voxels[1], nb_voxels[2]).float()
            
            if object_only:
                occ[occ == 1] = 0.

            if self.scene_type in ['plane_two', 'occ_two', 'occ_temp']:
                mat_for_query_goal = mat.clone()
                
                # handle pelvis goal in the is_loco case
                pelvis_goal_copy = pelvis_goal.clone()
                pelvis_goal_copy[is_loco] = pelvis_goal_copy[is_loco] / (
                            torch.norm(pelvis_goal_copy[is_loco], dim=-1, keepdim=True) + 1e-6) * 0.8
                pelvis_goal_orig = transform_points(pelvis_goal_copy.reshape(pelvis_goal_copy.shape[0], 1, 3), mat).squeeze(1)

                # handle object goal in the is_object case - no rotation needed
                object_goal_copy = object_goal.clone()
                object_goal_orig = transform_points(object_goal_copy.reshape(object_goal_copy.shape[0], 1, 3), mat).squeeze(1)

                mat_for_query_goal[need_pelvis_dir, :3, 3] = pelvis_goal_orig[need_pelvis_dir]
                mat_for_query_goal[is_object, :3, 3] = object_goal_orig[is_object]
                mat_for_query_goal[torch.logical_not(torch.logical_or(need_pelvis_dir, is_object)), :3, 3] = mat_for_query[
                                                                                torch.logical_not(torch.logical_or(need_pelvis_dir, is_object)), :3,
                                                                                3].clone()
                mat_for_query_goal[:, 1, 3] = 0.
                query_points_goal = transform_points(self.grid, mat_for_query_goal)
                occ_goal = self.dataset.get_occ_for_points(query_points_goal, object_points, scene_flag)

                if object_only:
                    occ_goal[occ_goal == 1] = 0.

                nb_voxels = self.dataset.nb_voxels
                occ_goal = occ_goal.reshape(-1, nb_voxels[0], nb_voxels[1], nb_voxels[2]).float()

            occ_list = torch.zeros(0, nb_voxels[1], nb_voxels[0], nb_voxels[2]).to(self.device)
            occ_list = torch.cat([occ_list, occ], dim=0)
            occ_temp = None
            if self.scene_type == 'occ_temp':
                object_points_temp = object_points.clone()
                pred_obj_rot_mat_rel = x[:, :, 219:228].reshape(x.shape[0], -1, 3, 3)
                
                obj_rot_mat_ref_temp = obj_rot_mat_ref
                pred_obj_rot_mat = pred_obj_rot_mat_rel @ obj_rot_mat_ref_temp # [b, t, 3, 3]
                pred_obj_rot_mat = pred_obj_rot_mat @ pred_obj_rot_mat[:, 0:1, :, :].transpose(2, 3)

                pred_obj_trans = x[:, :, 216:219] # [b, t, 3]
                pred_obj_trans = transform_points(self.dataset.denormalize_torch(pred_obj_trans, is_object=True), mat)
                pred_obj_trans = pred_obj_trans - pred_obj_trans[:, 0:1, :]

                object_points_temp = object_points_temp.unsqueeze(1).repeat(1, pred_obj_rot_mat.shape[1], 1, 1) # [b, t, 1024, 3]
                object_points_temp = torch.matmul(pred_obj_rot_mat, object_points_temp.transpose(-2,-1)).transpose(-2,-1) + pred_obj_trans.unsqueeze(-2) # [b, t, 1024, 3]

                # dynamically obtain temporal frame indices
                temp_indices = self._get_temp_frame_indices(self.temp_voxel_num)
                
                # only loop when temporal voxels exist
                for i in temp_indices:
                    mat_for_query = mat.clone()
                    target_ind = self.mask_ind if self.mask_ind != -1 else 0
                    mat_for_query[:, :3, 3] = x_orig[:, i, target_ind * 3: target_ind * 3 + 3]
                    mat_for_query[:, 1, 3] = 0
                    query_points = transform_points(self.grid, mat_for_query)
                    
                    occ_temp = self.dataset.get_occ_for_points(query_points, object_points_temp[:, i, :, :], scene_flag)
                    
                    if object_only:
                        occ_temp[occ_temp == 1] = 0.
                    
                    nb_voxels = self.dataset.nb_voxels
                    occ_temp = occ_temp.reshape(-1, nb_voxels[0], nb_voxels[1], nb_voxels[2]).float()
                    occ_temp = occ_temp.permute(0, 2, 1, 3)

                    occ_list = torch.cat([occ_list, occ_temp], dim=0)

            if self.scene_type == 'occ':
                occ = occ.permute(0, 2, 1, 3)
            elif self.scene_type == 'plane':
                occ = occ.permute(0, 1, 3, 2)
                occ_cnt = occ * self.occ_idx
                occ = torch.argmax(occ_cnt, dim=-1).unsqueeze(1).float() / nb_voxels[1]
            elif self.scene_type == 'plane_two':
                occ = occ.permute(0, 1, 3, 2)
                occ_cnt = occ * self.occ_idx
                occ = torch.argmax(occ_cnt, dim=-1).unsqueeze(1).float() / nb_voxels[1]

                occ_goal = occ_goal.permute(0, 1, 3, 2)
                occ_goal_cnt = occ_goal * self.occ_idx
                occ_goal = torch.argmax(occ_goal_cnt, dim=-1).unsqueeze(1).float() / nb_voxels[1]
                occ = torch.cat([occ, occ_goal], dim=1)
            elif self.scene_type == 'occ_two':
                occ = occ.permute(0, 2, 1, 3)
                occ_goal = occ_goal.permute(0, 2, 1, 3)
                occ = torch.cat([occ, occ_goal], dim=1)
            elif self.scene_type == 'occ_temp':
                occ = occ_goal.permute(0, 2, 1, 3)

        else:
            occ = None

        if t_index == 1 or t_index == 2 or t_index == 3 or t_index == 4 or t_index == 5:
            with torch.enable_grad():
                x = x.detach().requires_grad_(True)

                start_timestep = self.solver.ddim_timesteps[t]
                model_output = model(x, occ, start_timestep, text_emb, pelvis_goal, hand_goal, is_pick, need_scene, need_pelvis_dir, pi, end_pi, seq_length, need_pi, object_goal, is_object, obj_bps_data, occ_list)

                betas_t = extract(self.betas, start_timestep, x.shape)
                sqrt_one_minus_alphas_cumprod_t = extract(
                    self.sqrt_one_minus_alphas_cumprod, start_timestep, x.shape
                )
                sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, start_timestep, x.shape)
                
                x_start = model_output
                
                global_jpos = x_start[:, :, :84].reshape(self.batch_size, self.dataset.max_window_size, 84)
                global_jpos = transform_points(self.dataset.denormalize_torch(global_jpos), mat).reshape(self.batch_size, self.dataset.max_window_size, 28, 3)

                # FK to get joint positions.
                curr_seq_local_jpos = rest_human_offsets # [b, t, 24, 3]
                curr_seq_local_jpos = curr_seq_local_jpos.reshape(-1, 24, 3) # [b*t, 24, 3]
                curr_seq_local_jpos[:, 0, :] = global_jpos.reshape(-1, 28, 3)[:, 0, :]

                global_jrot_6d = x_start[:, :, 84:216].reshape(self.batch_size, self.dataset.max_window_size, 22, 6)
                global_jrot_mat = transforms.rotation_6d_to_matrix(global_jrot_6d) # [b, t, 22, 3, 3]
                global_jrot_mat = mat[:, None, None, :3, :3] @ global_jrot_mat

                local_jrot_mat = self.dataset.quat_ik_torch(global_jrot_mat.reshape(-1, 22, 3, 3)) # [b*t, 22, 3, 3]
                _, human_jnts = self.dataset.quat_fk_torch(local_jrot_mat, curr_seq_local_jpos) # [b*t, 24, 3]
                human_jnts = human_jnts.reshape(self.batch_size, -1, 24, 3) # [b, t, 24, 3]

                pred_seq_com_pos = x_start[:, :, 216:219].reshape(self.batch_size, self.dataset.max_window_size, 3)
                pred_seq_com_pos = transform_points(self.dataset.denormalize_torch(pred_seq_com_pos, is_object=True), mat)

                object_rot_mat = x_start[:, :, 219:228].reshape(self.batch_size, self.dataset.max_window_size, 3, 3) # B X 16 X 3 X 3
                pred_obj_rot_mat = (object_rot_mat.reshape(self.batch_size, -1, 3, 3) @ obj_rot_mat_ref)
                
                contact_labels = x_start[:, :, 228:232].reshape(self.batch_size, self.dataset.max_window_size, 4)

                obj_verts = torch.zeros(0, self.dataset.max_window_size, 10000, 3).to(self.device)
                
                for seg_id in range(self.batch_size):
                    obj_name = seq_name_dict[seg_id].split('_')[1]
                    pred_obj_rot_mat_seg = pred_obj_rot_mat[seg_id].reshape(-1, 3, 3)
                    pred_seq_com_pos_seg = pred_seq_com_pos[seg_id].reshape(-1, 3)
                    obj_rest_verts_seg = load_object_geometry_w_rest_geo(pred_obj_rot_mat_seg, pred_seq_com_pos_seg, obj_rest_verts[obj_name])
                    obj_rest_verts_seg = obj_rest_verts_seg.reshape(1, self.dataset.max_window_size, -1, 3) # 1 X T X Nv X 3
                    num_obj_verts = obj_rest_verts_seg.shape[2]
                    if num_obj_verts > 10000:
                        # randomly select indices of 10000 points
                        indices = torch.randperm(num_obj_verts)[:10000]
                        obj_rest_verts_seg = obj_rest_verts_seg[:, :, indices, :].reshape(1, self.dataset.max_window_size, 10000, 3)
                    obj_verts = torch.cat([obj_verts, obj_rest_verts_seg], dim=0)

                assert obj_verts.shape[0] == self.batch_size

                loss = guidance_fn(human_jnts, obj_verts, pred_seq_com_pos, pred_obj_rot_mat, contact_labels)
                # is_penetrating, nearest_free_points = self.dataset.get_nearest_free_voxel(human_jnts, scene_flag)
                # loss += F.mse_loss(human_jnts, nearest_free_points) * 1000000000 # * 1200000000
                # if is_penetrating.any():
                    # print(loss)

                gradient = torch.autograd.grad(-loss, x_start)[0] * guidance_scale

                # tmp_posterior_variance_t = extract(self.posterior_variance, start_timestep, x_start.shape)

                alpha_cumprod = extract(self.alpha_cumprod, start_timestep, x_start.shape)
                
                # pred_x_0 = x_start + gradient * (1 - alpha_cumprod)
                x_start[:, :, :216] = x_start[:, :, :216] + gradient[:, :, :216] * (1 - alpha_cumprod)
                pred_x_0 = x_start

                sqrt_one_minus_alphas_cumprod_t = extract(
                        self.sqrt_one_minus_alphas_cumprod, start_timestep, x.shape
                    )
                sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, start_timestep, x.shape)
                
                noise_pred = (x - sqrt_alphas_cumprod_t * pred_x_0) / sqrt_one_minus_alphas_cumprod_t
        else:    
            start_timestep = self.solver.ddim_timesteps[t]

            pred_x_0 = model(x, occ, start_timestep, text_emb, pelvis_goal, hand_goal, is_pick, need_scene, need_pelvis_dir, pi, end_pi, seq_length, need_pi, object_goal, is_object, obj_bps_data, occ_list)

            sqrt_one_minus_alphas_cumprod_t = extract(
                    self.sqrt_one_minus_alphas_cumprod, start_timestep, x.shape
                )
            sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, start_timestep, x.shape)
            
            noise_pred = (x - sqrt_alphas_cumprod_t * pred_x_0) / sqrt_one_minus_alphas_cumprod_t

        print('start_timestep', start_timestep[0])
        if t_index == 0:
            model_pred = pred_x_0
            print('t_index', t_index)
        # elif t_index == 1:
        #     model_pred = self.solver.ddim_step(pred_x_0, noise_pred, t) + gradient * tmp_posterior_variance_t
        else:
            model_pred = self.solver.ddim_step(pred_x_0, noise_pred, t)
        
        return model_pred.float(), occ

    def p_losses(self, x_start, joints, mat, scene_flag, mask, t, text_emb, pelvis_goal, hand_goal, object_goal, is_pick, need_scene, need_pelvis_dir, pi, end_pi, seq_length, need_pi, is_loco, is_object, obj_bps_data, obj_rot_mat_ref, rest_pose_obj_nn_pts, transformed_obj_verts, rest_human_offsets, object_points=None, noise=None, loss_type='huber'):
        if noise is None:
            noise = torch.randn_like(x_start)

        # ensure all inputs are float type and on the correct device
        noise = noise.to(x_start.device, dtype=torch.float32)
        mask = mask.to(x_start.device, dtype=torch.bool)

        # set the noise in the masked region to 0
        noise[mask] = 0.

        # generate noise data
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy[mask] = x_start[mask]
        x_noisy[torch.logical_not(is_object), :, 216:] = x_start[torch.logical_not(is_object), :, 216:]

        # print('x noisy in mask with scale')

        if self.dataset.load_scene:
            with torch.no_grad():
                # print(x_noisy.shape, joints.shape, mat.shape)
                x_orig = transform_points(self.dataset.denormalize_torch(x_noisy[:, :, :joints.shape[-1]], is_chois=is_object), mat)
                mat_for_query = mat.clone()
                target_ind = self.mask_ind if self.mask_ind != -1 else 0
                mat_for_query[:, :3, 3] = x_orig[:, self.emb_f, target_ind * 3: target_ind * 3 + 3]
                mat_for_query[:, 1, 3] = 0
                query_points = transform_points(self.grid, mat_for_query)
                occ = self.dataset.get_occ_for_points(query_points, object_points, scene_flag)

                nb_voxels = self.dataset.nb_voxels
                occ = occ.reshape(-1, nb_voxels[0], nb_voxels[1], nb_voxels[2]).float()


                if self.scene_type in ['plane_two', 'occ_two', 'occ_temp']:
                    mat_for_query_goal = mat.clone()
                    
                    # handle pelvis goal in the need_pelvis_dir case
                    pelvis_goal_copy = pelvis_goal.clone()
                    # handle pelvis goal in the not-is_loco case (static scene interaction), where pelvis is replaced by hand goal
                    hand_goal_copy = hand_goal.clone()

                    pelvis_goal_copy[torch.logical_not(is_loco)] = hand_goal_copy[torch.logical_not(is_loco)]
                    # handle pelvis goal in the is_loco case
                    # pelvis_goal_copy[is_loco] = pelvis_goal_copy[is_loco] / (torch.norm(pelvis_goal_copy[is_loco], dim=-1, keepdim=True) + 1e-6) * 0.8
                    pelvis_goal_orig = transform_points(pelvis_goal_copy.unsqueeze(1), mat).squeeze(1)

                    # handle object goal in the is_object case - no rotation needed
                    object_goal_copy = object_goal.clone()
                    # object_goal_copy[is_object] = object_goal_copy[is_object] / (torch.norm(object_goal_copy[is_object], dim=-1, keepdim=True) + 1e-6) * 0.8
                    object_goal_orig = transform_points(object_goal_copy.unsqueeze(1), mat).squeeze(1)

                    # set goal position based on need_pelvis_dir and is_object
                    mat_for_query_goal[need_pelvis_dir, :3, 3] = pelvis_goal_orig[need_pelvis_dir] # need_pelvis_dir: inter_scene, is_loco, is_object
                    mat_for_query_goal[is_object, :3, 3] = object_goal_orig[is_object] # is_object: inter_object
                    mat_for_query_goal[torch.logical_not(torch.logical_or(need_pelvis_dir, is_object)), :3, 3] = mat_for_query[torch.logical_not(torch.logical_or(need_pelvis_dir, is_object)), :3, 3].clone()
                    mat_for_query_goal[:, 1, 3] = 0.
                    
                    query_points = transform_points(self.grid, mat_for_query_goal)
                    occ_goal = self.dataset.get_occ_for_points(query_points, None, scene_flag)
                    nb_voxels = self.dataset.nb_voxels
                    occ_goal = occ_goal.reshape(-1, nb_voxels[0], nb_voxels[1], nb_voxels[2]).float()

                    end_goal_pos = torch.zeros(self.batch_size, 2).to(self.device)
                    end_goal_pos[need_pelvis_dir] = pelvis_goal_copy[need_pelvis_dir].reshape(-1, 3)[:, [0, 2]]
                    end_goal_pos[is_object] = object_goal_copy[is_object].reshape(-1, 3)[:, [0, 2]]
                
                occ_pos = torch.zeros(0, self.batch_size, 2).to(self.device)
                occ_pos = torch.cat([occ_pos, end_goal_pos[None]], dim=0)

                occ_list = torch.zeros(0, nb_voxels[1], nb_voxels[0], nb_voxels[2]).to(self.device)
                occ_list = torch.cat([occ_list, occ], dim=0)
                occ_temp = None
                if self.scene_type == 'occ_temp':
                    object_points_temp = object_points.clone()
                    if self.is_o0 == False:
                        pred_obj_rot_mat_rel = x_noisy[:, :, 219:228].reshape(joints.shape[0], -1, 3, 3)
                        
                        pred_obj_rot_mat_rel_aa = transforms.matrix_to_axis_angle(pred_obj_rot_mat_rel) # [b, t, 3]
                        # std_per_dim = torch.tensor([0.5, 1.5, 0.5], device=pred_obj_rot_mat_rel_aa.device).view(1, 1, 3)
                        # perturb = torch.randn_like(pred_obj_rot_mat_rel_aa) * std_per_dim
                        # pred_obj_rot_mat_rel_aa = pred_obj_rot_mat_rel_aa + perturb
                        pred_obj_rot_mat_rel = transforms.axis_angle_to_matrix(pred_obj_rot_mat_rel_aa)
                        
                        obj_rot_mat_ref_temp = obj_rot_mat_ref.unsqueeze(1).repeat(1, pred_obj_rot_mat_rel.shape[1], 1, 1)
                        pred_obj_rot_mat = pred_obj_rot_mat_rel @ obj_rot_mat_ref_temp # [b, t, 3, 3]
                        pred_obj_rot_mat = pred_obj_rot_mat @ pred_obj_rot_mat[:, 0:1, :, :].transpose(2, 3)

                        pred_obj_trans = x_noisy[:, :, 216:219] # [b, t, 3]
                        pred_obj_trans = transform_points(self.dataset.denormalize_torch(pred_obj_trans, is_object=True), mat)
                        pred_obj_trans = pred_obj_trans - pred_obj_trans[:, 0:1, :]

                        # perturb = (torch.rand_like(pred_obj_trans) - 0.5) * 0.4  # ∈ [-0.2, 0.2]
                        # pred_obj_trans = pred_obj_trans + perturb
                    else:
                        pred_obj_rot_mat_rel = x_start[:, :, 219:228].reshape(joints.shape[0], -1, 3, 3)
                        
                        pred_obj_rot_mat_rel_aa = transforms.matrix_to_axis_angle(pred_obj_rot_mat_rel) # [b, t, 3]
                        std_per_dim = torch.tensor([0.5, 1.5, 0.5], device=pred_obj_rot_mat_rel_aa.device).view(1, 1, 3)
                        perturb = torch.randn_like(pred_obj_rot_mat_rel_aa) * std_per_dim
                        pred_obj_rot_mat_rel_aa = pred_obj_rot_mat_rel_aa + perturb
                        pred_obj_rot_mat_rel = transforms.axis_angle_to_matrix(pred_obj_rot_mat_rel_aa)
                        
                        obj_rot_mat_ref_temp = obj_rot_mat_ref.unsqueeze(1).repeat(1, pred_obj_rot_mat_rel.shape[1], 1, 1)
                        pred_obj_rot_mat = pred_obj_rot_mat_rel @ obj_rot_mat_ref_temp # [b, t, 3, 3]
                        pred_obj_rot_mat = pred_obj_rot_mat @ pred_obj_rot_mat[:, 0:1, :, :].transpose(2, 3)

                        pred_obj_trans = x_start[:, :, 216:219] # [b, t, 3]
                        pred_obj_trans = transform_points(self.dataset.denormalize_torch(pred_obj_trans, is_object=True), mat)
                        pred_obj_trans = pred_obj_trans - pred_obj_trans[:, 0:1, :]

                        perturb = (torch.rand_like(pred_obj_trans) - 0.5) * 0.4  # ∈ [-0.2, 0.2]
                        pred_obj_trans = pred_obj_trans + perturb

                    object_points_temp = object_points_temp.unsqueeze(1).repeat(1, pred_obj_rot_mat.shape[1], 1, 1) # [b, t, 1024, 3]
                    object_points_temp = torch.matmul(pred_obj_rot_mat, object_points_temp.transpose(-2,-1)).transpose(-2,-1) + pred_obj_trans.unsqueeze(-2) # [b, t, 1024, 3]

                    x_denorm = self.dataset.denormalize_torch(x_start[:, :, :joints.shape[-1]])
                    perturb = (torch.rand_like(x_denorm) - 0.5) * 0.2  # ∈ [-0.1, 0.1]
                    x_denorm = x_denorm + perturb

                    # dynamically obtain temporal frame indices
                    temp_indices = self._get_temp_frame_indices(self.temp_voxel_num)
                    
                    # only loop when temporal voxels exist
                    for i in temp_indices:
                        x0_orig = transform_points(x_denorm, mat)
                        mat_for_query = mat.clone()
                        target_ind = self.mask_ind if self.mask_ind != -1 else 0
                        
                        mat_for_query[:, :3, 3] = x0_orig[:, i, target_ind * 3: target_ind * 3 + 3]
                        mat_for_query[:, 1, 3] = 0

                        occ_pos = torch.cat([occ_pos, x_denorm[:, i, [0, 2]][None]], dim=0)

                        query_points = transform_points(self.grid, mat_for_query)
                    
                        occ_temp = self.dataset.get_occ_for_points(query_points, object_points_temp[:, i, :, :], scene_flag)
                        nb_voxels = self.dataset.nb_voxels
                        occ_temp = occ_temp.reshape(-1, nb_voxels[0], nb_voxels[1], nb_voxels[2]).float()
                        occ_temp = occ_temp.permute(0, 2, 1, 3)

                        occ_list = torch.cat([occ_list, occ_temp], dim=0)

                if self.scene_type == 'occ':
                    occ = occ.permute(0, 2, 1, 3)
                elif self.scene_type == 'plane':
                    occ = occ.permute(0, 1, 3, 2)
                    occ_cnt = occ * self.occ_idx
                    occ = torch.argmax(occ_cnt, dim=-1).unsqueeze(1).float() / nb_voxels[1]
                elif self.scene_type == 'plane_two':
                    occ = occ.permute(0, 1, 3, 2)
                    occ_cnt = occ * self.occ_idx
                    occ = torch.argmax(occ_cnt, dim=-1).unsqueeze(1).float() / nb_voxels[1]

                    occ_goal = occ_goal.permute(0, 1, 3, 2)
                    occ_goal_cnt = occ_goal * self.occ_idx
                    occ_goal = torch.argmax(occ_goal_cnt, dim=-1).unsqueeze(1).float() / nb_voxels[1]
                    occ = torch.cat([occ, occ_goal], dim=1)
                elif self.scene_type == 'occ_two':
                    occ = occ.permute(0, 2, 1, 3)
                    occ_goal = occ_goal.permute(0, 2, 1, 3)
                    occ = torch.cat([occ, occ_goal], dim=1)
                elif self.scene_type == 'occ_temp':
                    occ = occ_goal.permute(0, 2, 1, 3)

        else:
            occ = None

        # print(pi[:10], end_pi[:10], seq_length[:10])
        # use the model to predict noise
        predicted_noise = self.student_model(x_noisy, occ, t, text_emb, pelvis_goal, hand_goal, is_loco, need_scene, need_pelvis_dir, pi, end_pi, seq_length, need_pi, object_goal, is_object, obj_bps_data, occ_list, occ_pos)

        # compute the loss
        mask_inv = torch.logical_not(mask)
        # if loss_type == 'l1':
        #     loss = F.l1_loss(x_start[mask_inv], predicted_noise[mask_inv])
        # elif loss_type == 'l2':
        #     loss = F.mse_loss(x_start[mask_inv], predicted_noise[mask_inv])
        # elif loss_type == "huber":
        #     loss = F.smooth_l1_loss(x_start[mask_inv], predicted_noise[mask_inv])
        # else:
        #     raise NotImplementedError()

        loss_jpos = F.mse_loss(x_start[:, :, :84][mask_inv[:, :, :84]], predicted_noise[:, :, :84][mask_inv[:, :, :84]])
        loss_jrot = F.l1_loss(x_start[:, :, 84:216][mask_inv[:, :, 84:216]], predicted_noise[:, :, 84:216][mask_inv[:, :, 84:216]])
        
        # create masks for the object-related losses
        mask_otrans = mask_inv[:, :, 216:219].clone()
        mask_orot = mask_inv[:, :, 219:228].clone()
        mask_contact = mask_inv[:, :, 228:232].clone()
        
        mask_otrans[torch.logical_not(is_object)] = False
        mask_orot[torch.logical_not(is_object)] = False
        mask_contact[torch.logical_not(is_object)] = False

        # compute the masked losses
        loss_otrans = F.mse_loss(x_start[:, :, 216:219][mask_otrans], predicted_noise[:, :, 216:219][mask_otrans]) if mask_otrans.any() else torch.tensor(0.0, device=x_start.device)
        loss_orot = F.l1_loss(x_start[:, :, 219:228][mask_orot], predicted_noise[:, :, 219:228][mask_orot]) if mask_orot.any() else torch.tensor(0.0, device=x_start.device)
        loss_contact = F.l1_loss(x_start[:, :, 228:232][mask_contact], predicted_noise[:, :, 228:232][mask_contact]) if mask_contact.any() else torch.tensor(0.0, device=x_start.device)

        loss = loss_jpos + loss_jrot + loss_otrans + loss_orot + loss_contact

        # add object loss (obj_rot_mat_ref, rest_pose_obj_nn_pts, transformed_obj_verts)
        if self.dataset.use_object_keypoints:
            hand_idx_28 = [20, 21, 25, 27]
            hand_idx_24 = [20, 21, 22, 23]
            foot_idx = [7, 8, 10, 11]
            
            gt_global_jpos = transform_points(self.dataset.denormalize_torch(joints, is_chois=is_object), mat).reshape(joints.shape[0], -1, 28, 3)
            gt_global_hand_jpos = gt_global_jpos[:, :, hand_idx_28, :]
            gt_global_foot_jpos = gt_global_jpos[:, :, foot_idx, :]

            global_jpos = transform_points(self.dataset.denormalize_torch(predicted_noise[:, :, :84], is_chois=is_object), mat).reshape(joints.shape[0], -1, 28, 3)

            # FK to get joint positions.
            curr_seq_local_jpos = rest_human_offsets[:, None].repeat(1, global_jpos.shape[1], 1, 1) # [b, t, 24, 3]
            curr_seq_local_jpos = curr_seq_local_jpos.reshape(-1, 24, 3) # [b*t, 24, 3]
            curr_seq_local_jpos[:, 0, :] = global_jpos.reshape(-1, 28, 3)[:, 0, :]

            global_jrot_6d = predicted_noise[:, :, 84:216].reshape(joints.shape[0], -1, 22, 6)
            global_jrot_mat = transforms.rotation_6d_to_matrix(global_jrot_6d) # [b, t, 22, 3, 3]
            global_jrot_mat = mat[:, None, None, :3, :3] @ global_jrot_mat
            
            local_jrot_mat = self.dataset.quat_ik_torch(global_jrot_mat.reshape(-1, 22, 3, 3)) # [b*t, 22, 3, 3]
            _, human_jnts = self.dataset.quat_fk_torch(local_jrot_mat, curr_seq_local_jpos) # [b*t, 24, 3]
            human_jnts = human_jnts.reshape(joints.shape[0], -1, 24, 3) # [b, t, 24, 3]

            pred_global_hand_jpos = human_jnts[:, :, hand_idx_24, :]
            pred_global_foot_jpos = human_jnts[:, :, foot_idx, :] # [b, t, 4, 3]

            mask_fk = torch.ones(mask_inv.shape[0], self.dataset.max_window_size, 4, 3, dtype=torch.bool).to(mask_inv.device)
            mask_fk[:, :self.auto_regre_num, :, :] = False
            # print(torch.equal(mask_fk, mask_inv[:, :, :3*4].reshape(mask_inv.shape[0], -1, 4, 3)))
            fk_hand_loss = F.mse_loss(pred_global_hand_jpos[mask_fk], gt_global_hand_jpos[mask_fk])
            fk_foot_loss = F.mse_loss(pred_global_foot_jpos[mask_fk], gt_global_foot_jpos[mask_fk])
            loss_fk = fk_hand_loss + fk_foot_loss
            
            model_mean = predicted_noise # x_start
            pred_obj_rot_mat_rel = model_mean[:, :, 219:228].reshape(joints.shape[0], -1, 3, 3)
            obj_rot_mat_ref = obj_rot_mat_ref.unsqueeze(1).repeat(1, pred_obj_rot_mat_rel.shape[1], 1, 1)
            pred_obj_rot_mat = pred_obj_rot_mat_rel @ obj_rot_mat_ref # [b, t, 3, 3]

            pred_obj_trans = model_mean[:, :, 216:219] # [b, t, 3]
            pred_obj_trans = transform_points(self.dataset.denormalize_torch(pred_obj_trans, is_object=True), mat)

            rest_pose_obj_nn_pts = rest_pose_obj_nn_pts.unsqueeze(1).repeat(1, pred_obj_rot_mat.shape[1], 1, 1) # [b, t, 100, 3]
            pred_seq_obj_kpts = torch.matmul(pred_obj_rot_mat, rest_pose_obj_nn_pts.transpose(-2,-1)).transpose(-2,-1) + pred_obj_trans.unsqueeze(-2) # [b, t, 100, 3]
            
            # transformed_obj_verts = self.dataset.normalize_torch(transformed_obj_verts, is_object=True)
            # pred_seq_obj_kpts = self.dataset.normalize_torch(pred_seq_obj_kpts, is_object=True)
            
            mask_points = torch.ones(mask_inv.shape[0], self.dataset.max_window_size, 100, 3, dtype=torch.bool).to(mask_inv.device)
            mask_points[:, :self.auto_regre_num, :, :] = False
            mask_points[torch.logical_not(is_object)] = False
            # print(mask_points[0, :, 0, 0])
            if loss_type == 'l1':
                loss_object = F.l1_loss(transformed_obj_verts[mask_points], pred_seq_obj_kpts[mask_points])
            elif loss_type == 'l2':
                loss_object = F.mse_loss(transformed_obj_verts[mask_points], pred_seq_obj_kpts[mask_points])
            elif loss_type == "huber":
                loss_object = F.smooth_l1_loss(transformed_obj_verts[mask_points], pred_seq_obj_kpts[mask_points])
            else:
                raise NotImplementedError()

            # --- Velocity Loss Calculation ---
            
            # 1. Human Velocity Loss
            # First, get ground truth 3D joint positions by applying FK to x_start to ensure consistency
            gt_global_jpos_for_fk = transform_points(self.dataset.denormalize_torch(x_start[:, :, :84], is_chois=is_object), mat).reshape(joints.shape[0], -1, 28, 3)
            gt_curr_seq_local_jpos = rest_human_offsets[:, None].repeat(1, gt_global_jpos_for_fk.shape[1], 1, 1)
            gt_curr_seq_local_jpos = gt_curr_seq_local_jpos.reshape(-1, 24, 3)
            gt_curr_seq_local_jpos[:, 0, :] = gt_global_jpos_for_fk.reshape(-1, 28, 3)[:, 0, :]
            
            gt_global_jrot_6d = x_start[:, :, 84:216].reshape(joints.shape[0], -1, 22, 6)
            gt_global_jrot_mat = transforms.rotation_6d_to_matrix(gt_global_jrot_6d)
            gt_global_jrot_mat = mat[:, None, None, :3, :3] @ gt_global_jrot_mat
            
            gt_local_jrot_mat = self.dataset.quat_ik_torch(gt_global_jrot_mat.reshape(-1, 22, 3, 3))
            _, gt_human_jnts = self.dataset.quat_fk_torch(gt_local_jrot_mat, gt_curr_seq_local_jpos)
            gt_human_jnts = gt_human_jnts.reshape(joints.shape[0], -1, 24, 3)

            # Calculate velocity for predicted (human_jnts) and ground truth (gt_human_jnts)
            vel_human_pred = human_jnts[:, 1:] - human_jnts[:, :-1]
            vel_human_gt = gt_human_jnts[:, 1:] - gt_human_jnts[:, :-1]
            loss_vel_human = F.mse_loss(vel_human_pred, vel_human_gt)

            # 2. Object Velocity Loss
            # Calculate velocity for predicted (pred_seq_obj_kpts) and ground truth (transformed_obj_verts)
            vel_obj_pred = pred_seq_obj_kpts[:, 1:] - pred_seq_obj_kpts[:, :-1]
            vel_obj_gt = transformed_obj_verts[:, 1:] - transformed_obj_verts[:, :-1]
            loss_vel_obj = F.mse_loss(vel_obj_pred, vel_obj_gt)
            
            # 3. Total Velocity Loss
            loss_vel = loss_vel_human + loss_vel_obj

        else: 
            loss_object = None
            loss_fk = None

        if occ_list is not None:
            del occ_list
        if occ is not None:
            del occ
        if occ_goal is not None:
            del occ_goal
        if occ_temp is not None:
            del occ_temp
            
        # _, nearest_free_points_human = self.dataset.get_nearest_free_voxel(human_jnts, scene_flag)
        # loss_h_pen = F.mse_loss(human_jnts, nearest_free_points_human)

        # _, nearest_free_points_obj = self.dataset.get_nearest_free_voxel(pred_seq_obj_kpts, scene_flag)
        # loss_o_pen = F.mse_loss(pred_seq_obj_kpts, nearest_free_points_obj)
        
        # loss_pen = loss_h_pen + 0.5 * loss_o_pen
        # print('loss_fk:', loss_fk, 'loss_object:', loss_object, 'loss_vel:', loss_vel)
        return loss, loss_fk, loss_object, loss_vel # , loss_pen

    @torch.no_grad()
    def p_sample_loop_guided(self, fixed_points, mat, scene_flag, text_emb, pelvis_goal, hand_goal, object_goal, \
                    is_pick, need_scene, need_pelvis_dir, pi, end_pi, seq_length, need_pi, is_loco, is_object, obj_bps_data, object_points, obj_rest_verts, seq_name_dict, obj_rot_mat_ref_first_step_batch, rest_human_offsets, guidance_fn, guidance_weight, object_only=False):
        self.batch_size = fixed_points.shape[0]
        device = next(self.student_model.parameters()).device
        shape = (self.batch_size, self.dataset.max_window_size, self.channel)
        points = torch.randn(shape, device=device)

        if self.auto_regre_num > 0:
            self.set_fixed_points(points, None, fixed_points, mat, joint_id=self.mask_ind, fix_mode=True, fix_goal=False)
        imgs = []
        occs = []
        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            model_used = self.student_model
            if guidance_fn is not None and i > 0 and i < 10:
                points, occ = self.p_sample_guided_reconstruction_guidance(model_used, points, fixed_points, mat, scene_flag,
                                        torch.full((self.batch_size,), i, device=device, dtype=torch.long), i,
                                        text_emb, pelvis_goal, hand_goal, object_goal, is_pick, need_scene, 
                                        need_pelvis_dir, pi, end_pi, seq_length, need_pi, is_loco, is_object, obj_bps_data, object_points, obj_rest_verts, seq_name_dict, obj_rot_mat_ref_first_step_batch, rest_human_offsets, guidance_fn, guidance_weight, object_only)
            else:
                points, occ = self.p_sample(model_used, points, fixed_points, mat, scene_flag,
                                        torch.full((self.batch_size,), i, device=device, dtype=torch.long), i,
                                        text_emb, pelvis_goal, hand_goal, object_goal, is_pick, need_scene, 
                                        need_pelvis_dir, pi, end_pi, seq_length, need_pi, is_loco, is_object, obj_bps_data, object_points, obj_rot_mat_ref_first_step_batch, object_only)
            
            if self.auto_regre_num > 0:
                self.set_fixed_points(points, None, fixed_points, mat, joint_id=self.mask_ind, fix_mode=True, fix_goal=False)

            points_orig = points

            imgs.append(points_orig)
            if occ is not None:
                occs.append(occ.cpu().numpy())

        return imgs, occs

    @torch.no_grad()
    def p_sample_guided_reconstruction_guidance(self, model, x, fixed_points, mat, scene_flag, t, t_index,
                 text_emb, pelvis_goal, hand_goal, object_goal, is_pick, need_scene, need_pelvis_dir, pi, end_pi, seq_length, need_pi, is_loco, is_object, obj_bps_data, object_points, obj_rest_verts, seq_name_dict, obj_rot_mat_ref_first_step_batch, rest_human_offsets, guidance_fn, guidance_weight, object_only=False):
        use_reconstruction_guidance = True

        if self.dataset.load_scene:
            x_orig = transform_points(self.dataset.denormalize_torch(x[:, :, :84]), mat)
            mat_for_query = mat.clone()
            target_ind = self.mask_ind if self.mask_ind != -1 else 0
            mat_for_query[:, :3, 3] = x_orig[:, self.emb_f, target_ind * 3: target_ind * 3 + 3]
            mat_for_query[:, 1, 3] = 0
            query_points = transform_points(self.grid, mat_for_query)
            occ = self.dataset.get_occ_for_points(query_points, object_points, scene_flag)
            nb_voxels = self.dataset.nb_voxels
            occ = occ.reshape(-1, nb_voxels[0], nb_voxels[1], nb_voxels[2]).float()

            if object_only:
                occ[occ == 1] = 0.

            if self.scene_type in ['plane_two', 'occ_two', 'occ_temp']:
                mat_for_query_goal = mat.clone()
                
                # handle pelvis goal in the is_loco case
                pelvis_goal_copy = pelvis_goal.clone()
                pelvis_goal_copy[is_loco] = pelvis_goal_copy[is_loco] / (
                            torch.norm(pelvis_goal_copy[is_loco], dim=-1, keepdim=True) + 1e-6) * 0.8
                pelvis_goal_orig = transform_points(pelvis_goal_copy.reshape(pelvis_goal_copy.shape[0], 1, 3), mat).squeeze(1)

                # handle object goal in the is_object case - no rotation needed
                object_goal_copy = object_goal.clone()
                object_goal_orig = transform_points(object_goal_copy.reshape(object_goal_copy.shape[0], 1, 3), mat).squeeze(1)

                mat_for_query_goal[need_pelvis_dir, :3, 3] = pelvis_goal_orig[need_pelvis_dir]
                mat_for_query_goal[is_object, :3, 3] = object_goal_orig[is_object]
                mat_for_query_goal[torch.logical_not(torch.logical_or(need_pelvis_dir, is_object)), :3, 3] = mat_for_query[
                                                                                torch.logical_not(torch.logical_or(need_pelvis_dir, is_object)), :3,
                                                                                3].clone()
                mat_for_query_goal[:, 1, 3] = 0.
                query_points_goal = transform_points(self.grid, mat_for_query_goal)
                occ_goal = self.dataset.get_occ_for_points(query_points_goal, object_points, scene_flag)
                nb_voxels = self.dataset.nb_voxels
                occ_goal = occ_goal.reshape(-1, nb_voxels[0], nb_voxels[1], nb_voxels[2]).float()

                if object_only:
                    occ_goal[occ_goal == 1] = 0.

            occ_list = torch.zeros(0, nb_voxels[1], nb_voxels[0], nb_voxels[2]).to(self.device)
            occ_list = torch.cat([occ_list, occ], dim=0)
            occ_temp = None
            if self.scene_type == 'occ_temp':
                object_points_temp = object_points.clone()
                pred_obj_rot_mat_rel = x[:, :, 219:228].reshape(x.shape[0], -1, 3, 3)
                
                obj_rot_mat_ref_temp = obj_rot_mat_ref_first_step_batch
                pred_obj_rot_mat = pred_obj_rot_mat_rel @ obj_rot_mat_ref_temp # [b, t, 3, 3]
                pred_obj_rot_mat = pred_obj_rot_mat @ pred_obj_rot_mat[:, 0:1, :, :].transpose(2, 3)

                pred_obj_trans = x[:, :, 216:219] # [b, t, 3]
                pred_obj_trans = transform_points(self.dataset.denormalize_torch(pred_obj_trans, is_object=True), mat)
                pred_obj_trans = pred_obj_trans - pred_obj_trans[:, 0:1, :]

                object_points_temp = object_points_temp.unsqueeze(1).repeat(1, pred_obj_rot_mat.shape[1], 1, 1) # [b, t, 1024, 3]
                object_points_temp = torch.matmul(pred_obj_rot_mat, object_points_temp.transpose(-2,-1)).transpose(-2,-1) + pred_obj_trans.unsqueeze(-2) # [b, t, 1024, 3]

                # dynamically obtain temporal frame indices
                temp_indices = self._get_temp_frame_indices(self.temp_voxel_num)
                
                # only loop when temporal voxels exist
                for i in temp_indices:
                    mat_for_query = mat.clone()
                    target_ind = self.mask_ind if self.mask_ind != -1 else 0
                    mat_for_query[:, :3, 3] = x_orig[:, i, target_ind * 3: target_ind * 3 + 3]
                    mat_for_query[:, 1, 3] = 0
                    query_points = transform_points(self.grid, mat_for_query)
                    
                    occ_temp = self.dataset.get_occ_for_points(query_points, object_points_temp[:, i, :, :], scene_flag)
                    nb_voxels = self.dataset.nb_voxels
                    occ_temp = occ_temp.reshape(-1, nb_voxels[0], nb_voxels[1], nb_voxels[2]).float()
                    occ_temp = occ_temp.permute(0, 2, 1, 3)

                    if object_only:
                        occ_temp[occ_temp == 1] = 0.

                    occ_list = torch.cat([occ_list, occ_temp], dim=0)
                    
            if self.scene_type == 'occ':
                occ = occ.permute(0, 2, 1, 3)
            elif self.scene_type == 'plane':
                occ = occ.permute(0, 1, 3, 2)
                occ_cnt = occ * self.occ_idx
                occ = torch.argmax(occ_cnt, dim=-1).unsqueeze(1).float() / nb_voxels[1]
            elif self.scene_type == 'plane_two':
                occ = occ.permute(0, 1, 3, 2)
                occ_cnt = occ * self.occ_idx
                occ = torch.argmax(occ_cnt, dim=-1).unsqueeze(1).float() / nb_voxels[1]

                occ_goal = occ_goal.permute(0, 1, 3, 2)
                occ_goal_cnt = occ_goal * self.occ_idx
                occ_goal = torch.argmax(occ_goal_cnt, dim=-1).unsqueeze(1).float() / nb_voxels[1]
                occ = torch.cat([occ, occ_goal], dim=1)
            elif self.scene_type == 'occ_two':
                occ = occ.permute(0, 2, 1, 3)
                occ_goal = occ_goal.permute(0, 2, 1, 3)
                occ = torch.cat([occ, occ_goal], dim=1)
            elif self.scene_type == 'occ_temp':
                occ = occ_goal.permute(0, 2, 1, 3)

        else:
            occ = None

        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            model_output = model(x, occ, t, text_emb, pelvis_goal, hand_goal, is_pick, need_scene, need_pelvis_dir, pi, end_pi, seq_length, need_pi, object_goal, is_object, obj_bps_data, occ_list)

            betas_t = extract(self.betas, t, x.shape)
            sqrt_one_minus_alphas_cumprod_t = extract(
                self.sqrt_one_minus_alphas_cumprod, t, x.shape
            )
            sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)
            
            x_start = model_output
            
            global_jpos = x_start[:, :, :84].reshape(self.batch_size, self.dataset.max_window_size, 84)
            global_jpos = transform_points(self.dataset.denormalize_torch(global_jpos), mat).reshape(self.batch_size, self.dataset.max_window_size, 28, 3)

            # FK to get joint positions.
            curr_seq_local_jpos = rest_human_offsets # [b, t, 24, 3]
            curr_seq_local_jpos = curr_seq_local_jpos.reshape(-1, 24, 3) # [b*t, 24, 3]
            curr_seq_local_jpos[:, 0, :] = global_jpos.reshape(-1, 28, 3)[:, 0, :]

            global_jrot_6d = x_start[:, :, 84:216].reshape(self.batch_size, self.dataset.max_window_size, 22, 6)
            global_jrot_mat = transforms.rotation_6d_to_matrix(global_jrot_6d) # [b, t, 22, 3, 3]
            global_jrot_mat = mat[:, None, None, :3, :3] @ global_jrot_mat

            local_jrot_mat = self.dataset.quat_ik_torch(global_jrot_mat.reshape(-1, 22, 3, 3)) # [b*t, 22, 3, 3]
            _, human_jnts = self.dataset.quat_fk_torch(local_jrot_mat, curr_seq_local_jpos) # [b*t, 24, 3]
            human_jnts = human_jnts.reshape(self.batch_size, -1, 24, 3) # [b, t, 24, 3]

            pred_seq_com_pos = x_start[:, :, 216:219].reshape(self.batch_size, self.dataset.max_window_size, 3)
            pred_seq_com_pos = transform_points(self.dataset.denormalize_torch(pred_seq_com_pos, is_object=True), mat)

            object_rot_mat = x_start[:, :, 219:228].reshape(self.batch_size, self.dataset.max_window_size, 3, 3) # B X 16 X 3 X 3
            pred_obj_rot_mat = (object_rot_mat.reshape(self.batch_size, -1, 3, 3) @ obj_rot_mat_ref_first_step_batch)
            
            contact_labels = x_start[:, :, 228:232].reshape(self.batch_size, self.dataset.max_window_size, 4)

            obj_verts = torch.zeros(0, self.dataset.max_window_size, 10000, 3).to(self.device)
            
            for seg_id in range(self.batch_size):
                obj_name = seq_name_dict[seg_id].split('_')[1]
                pred_obj_rot_mat_seg = pred_obj_rot_mat[seg_id].reshape(-1, 3, 3)
                pred_seq_com_pos_seg = pred_seq_com_pos[seg_id].reshape(-1, 3)
                obj_rest_verts_seg = load_object_geometry_w_rest_geo(pred_obj_rot_mat_seg, pred_seq_com_pos_seg, obj_rest_verts[obj_name])
                obj_rest_verts_seg = obj_rest_verts_seg.reshape(1, self.dataset.max_window_size, -1, 3) # 1 X T X Nv X 3
                num_obj_verts = obj_rest_verts_seg.shape[2]
                if num_obj_verts > 10000:
                    # randomly select indices of 10000 points
                    indices = torch.randperm(num_obj_verts)[:10000]
                    obj_rest_verts_seg = obj_rest_verts_seg[:, :, indices, :].reshape(1, self.dataset.max_window_size, 10000, 3)
                obj_verts = torch.cat([obj_verts, obj_rest_verts_seg], dim=0)

            assert obj_verts.shape[0] == self.batch_size

            loss = guidance_fn(human_jnts, obj_verts, pred_seq_com_pos, pred_obj_rot_mat, contact_labels)

            gradient = torch.autograd.grad(-loss, x_start)[0] * guidance_weight

            tmp_posterior_variance_t = extract(self.posterior_variance, t, x_start.shape)

            # x_start[216:] = x_start[216:] + gradient[216:] * tmp_posterior_variance_t
            x_start = x_start + gradient * tmp_posterior_variance_t

        model_mean = (
            extract(self.posterior_mean_coef1, t, x.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x.shape) * x
        )

        if t_index == 0:
            return model_mean, occ
        else:
            # posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            # return model_mean + torch.sqrt(posterior_variance_t) * torch.randn_like(x), occ
            model_log_variance = extract(self.posterior_log_variance_clipped, t, x.shape)
            return model_mean + (0.5 * model_log_variance).exp() * torch.randn_like(x), occ

    @torch.no_grad()
    def p_sample_loop(self, fixed_points, mat, scene_flag, text_emb, pelvis_goal, hand_goal, object_goal, \
                    is_pick, need_scene, need_pelvis_dir, pi, end_pi, seq_length, need_pi, is_loco, is_object, obj_bps_data, object_points, obj_rot_mat_ref, obj_rest_verts, seq_name_dict, obj_rot_mat_prefix, object_only=False):
        self.batch_size = fixed_points.shape[0]
        device = next(self.student_model.parameters()).device
        shape = (self.batch_size, self.dataset.max_window_size, self.channel)
        points = torch.randn(shape, device=device)

        if self.auto_regre_num > 0:
            self.set_fixed_points(points, None, fixed_points, mat, joint_id=self.mask_ind, fix_mode=True, fix_goal=False)
        imgs = []
        occs = []
        x0 = []
        x0.append(points)
        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            model_used = self.student_model

            points, occ, pred_x0 = self.p_sample(model_used, x0[-1], points, fixed_points, mat, scene_flag,
                                        torch.full((self.batch_size,), i, device=device, dtype=torch.long), i,
                                        text_emb, pelvis_goal, hand_goal, object_goal, is_pick, need_scene, 
                                        need_pelvis_dir, pi, end_pi, seq_length, need_pi, is_loco, is_object, obj_bps_data, object_points, obj_rot_mat_ref, obj_rest_verts, seq_name_dict, obj_rot_mat_prefix, object_only)
            if self.auto_regre_num > 0:
                self.set_fixed_points(points, None, fixed_points, mat, joint_id=self.mask_ind, fix_mode=True, fix_goal=False)

            points_orig = points
            imgs.append(points_orig)
            x0.append(pred_x0)
            if occ is not None:
                occs.append(occ.cpu().numpy())

        return imgs, occs

    @torch.no_grad()
    def p_sample(self, model, x0, x, fixed_points, mat, scene_flag, t, t_index,
                 text_emb, pelvis_goal, hand_goal, object_goal, is_pick, need_scene,
                 need_pelvis_dir, pi, end_pi, seq_length, need_pi, is_loco, is_object, obj_bps_data, object_points, obj_rot_mat_ref, obj_rest_verts, seq_name_dict, obj_rot_mat_prefix=None, object_only=False):
        if self.dataset.load_scene:
            x_orig = transform_points(self.dataset.denormalize_torch(x[:, :, :84]), mat)
            mat_for_query = mat.clone()
            target_ind = self.mask_ind if self.mask_ind != -1 else 0
            mat_for_query[:, :3, 3] = x_orig[:, self.emb_f, target_ind * 3: target_ind * 3 + 3]
            mat_for_query[:, 1, 3] = 0
            
            self.grid = self.dataset.create_meshgrid(batch_size=self.batch_size).to(self.device)

            query_points = transform_points(self.grid, mat_for_query)
            occ = self.dataset.get_occ_for_points(query_points, object_points, scene_flag)
            nb_voxels = self.dataset.nb_voxels
            occ = occ.reshape(-1, nb_voxels[0], nb_voxels[1], nb_voxels[2]).float()
            
            if object_only:
                occ[occ == 1] = 0.

            if torch.logical_not(is_object).any():
                occ[torch.logical_not(is_object)][occ == 2] = 1.
            
            if self.scene_type in ['plane_two', 'occ_two', 'occ_temp']:
                mat_for_query_goal = mat.clone()
                
                # handle pelvis goal in the is_loco case
                pelvis_goal_copy = pelvis_goal.clone()
                # handle pelvis goal in the not-is_loco case (static scene interaction), where pelvis is replaced by hand goal
                hand_goal_copy = hand_goal.clone()

                pelvis_goal_copy[torch.logical_not(is_loco)] = hand_goal_copy[torch.logical_not(is_loco)]
                # pelvis_goal_copy[is_loco] = pelvis_goal_copy[is_loco] / (
                #             torch.norm(pelvis_goal_copy[is_loco], dim=-1, keepdim=True) + 1e-6) * 0.8
                pelvis_goal_orig = transform_points(pelvis_goal_copy.reshape(pelvis_goal_copy.shape[0], 1, 3), mat).squeeze(1)

                # handle object goal in the is_object case - no rotation needed
                object_goal_copy = object_goal.clone()
                object_goal_orig = transform_points(object_goal_copy.reshape(object_goal_copy.shape[0], 1, 3), mat).squeeze(1)

                mat_for_query_goal[need_pelvis_dir, :3, 3] = pelvis_goal_orig[need_pelvis_dir]
                mat_for_query_goal[is_object, :3, 3] = object_goal_orig[is_object]
                mat_for_query_goal[torch.logical_not(torch.logical_or(need_pelvis_dir, is_object)), :3, 3] = mat_for_query[
                                                                                torch.logical_not(torch.logical_or(need_pelvis_dir, is_object)), :3,
                                                                                3].clone()
                mat_for_query_goal[:, 1, 3] = 0.
                query_points_goal = transform_points(self.grid, mat_for_query_goal)
                occ_goal = self.dataset.get_occ_for_points(query_points_goal, object_points, scene_flag)

                end_goal_pos = torch.zeros(self.batch_size, 2).to(self.device)
                end_goal_pos[need_pelvis_dir] = pelvis_goal_copy[need_pelvis_dir].reshape(-1, 3)[:, [0, 2]]
                end_goal_pos[is_object] = object_goal_copy[is_object].reshape(-1, 3)[:, [0, 2]]                    

                if object_only:
                    occ_goal[occ_goal == 1] = 0.

                if torch.logical_not(is_object).any():
                    occ_goal[torch.logical_not(is_object)][occ_goal == 2] = 1.

                nb_voxels = self.dataset.nb_voxels
                occ_goal = occ_goal.reshape(-1, nb_voxels[0], nb_voxels[1], nb_voxels[2]).float()

            occ_pos = torch.zeros(0, self.batch_size, 2).to(self.device)
            occ_pos = torch.cat([occ_pos, end_goal_pos[None]], dim=0)

            occ_list = torch.zeros(0, nb_voxels[1], nb_voxels[0], nb_voxels[2]).to(self.device)
            occ_list = torch.cat([occ_list, occ], dim=0)
            occ_temp = None
            if self.scene_type == 'occ_temp':
                if self.dataset.vis:
                    # object_rot_mat = x0[:, :, 219:228].reshape(x.shape[0], -1, 3, 3)
                    # object_trans_orig = x0[:, :, 216:219] # [b, t, 3]
                    object_rot_mat = x[:, :, 219:228].reshape(x.shape[0], -1, 3, 3)
                    object_trans_orig = x[:, :, 216:219] # [b, t, 3]
                    object_trans_orig = transform_points(self.dataset.denormalize_torch(object_trans_orig, is_object=True), mat)

                    obj_name = seq_name_dict[0].split('_')[1]
                    pred_obj_rot_mat_seg = (obj_rot_mat_prefix[None] @ object_rot_mat[:, :, :].reshape(-1, 3, 3) @ obj_rot_mat_ref).reshape(-1, 3, 3)
                    pred_seq_com_pos_seg = object_trans_orig[:, :, :].reshape(-1, 3)
                    obj_rest_verts_seg = load_object_geometry_w_rest_geo(pred_obj_rot_mat_seg, pred_seq_com_pos_seg, obj_rest_verts[obj_name])
                    indices = torch.randperm(obj_rest_verts_seg.shape[1])[:1024]
                    object_points_temp = obj_rest_verts_seg[:, indices, :].reshape(1, -1, 1024, 3)
                else:
                    object_points_temp = object_points.clone()
                    pred_obj_rot_mat_rel = x[:, :, 219:228].reshape(x.shape[0], -1, 3, 3)
                    
                    obj_rot_mat_ref_temp = obj_rot_mat_ref
                    pred_obj_rot_mat = pred_obj_rot_mat_rel @ obj_rot_mat_ref_temp # [b, t, 3, 3]
                    pred_obj_rot_mat = pred_obj_rot_mat @ pred_obj_rot_mat[:, 0:1, :, :].transpose(2, 3)

                    pred_obj_trans = x[:, :, 216:219] # [b, t, 3]
                    pred_obj_trans = transform_points(self.dataset.denormalize_torch(pred_obj_trans, is_object=True), mat)
                    pred_obj_trans = pred_obj_trans - pred_obj_trans[:, 0:1, :]

                    object_points_temp = object_points_temp.unsqueeze(1).repeat(1, pred_obj_rot_mat.shape[1], 1, 1) # [b, t, 1024, 3]
                    object_points_temp = torch.matmul(pred_obj_rot_mat, object_points_temp.transpose(-2,-1)).transpose(-2,-1) + pred_obj_trans.unsqueeze(-2) # [b, t, 1024, 3]

                x_denorm = self.dataset.denormalize_torch(x0[:, :, :84])
                # dynamically obtain temporal frame indices
                temp_indices = self._get_temp_frame_indices(self.temp_voxel_num)
                
                # only loop when temporal voxels exist
                for i in temp_indices:
                    x0_orig = transform_points(x_denorm, mat)
                    mat_for_query = mat.clone()
                    target_ind = self.mask_ind if self.mask_ind != -1 else 0
                    mat_for_query[:, :3, 3] = x0_orig[:, i, target_ind * 3: target_ind * 3 + 3]
                    mat_for_query[:, 1, 3] = 0

                    occ_pos = torch.cat([occ_pos, x_denorm[:, i, [0, 2]][None]], dim=0)

                    query_points = transform_points(self.grid, mat_for_query)
                    
                    occ_temp = self.dataset.get_occ_for_points(query_points, object_points_temp[:, i, :, :], scene_flag)
                    
                    if object_only:
                        occ_temp[occ_temp == 1] = 0.

                    if torch.logical_not(is_object).any():
                        occ_temp[torch.logical_not(is_object)][occ_temp == 2] = 1.

                    nb_voxels = self.dataset.nb_voxels
                    occ_temp = occ_temp.reshape(-1, nb_voxels[0], nb_voxels[1], nb_voxels[2]).float()
                    occ_temp = occ_temp.permute(0, 2, 1, 3)

                    occ_list = torch.cat([occ_list, occ_temp], dim=0)

            if self.scene_type == 'occ':
                occ = occ.permute(0, 2, 1, 3)
            elif self.scene_type == 'plane':
                occ = occ.permute(0, 1, 3, 2)
                occ_cnt = occ * self.occ_idx
                occ = torch.argmax(occ_cnt, dim=-1).unsqueeze(1).float() / nb_voxels[1]
            elif self.scene_type == 'plane_two':
                occ = occ.permute(0, 1, 3, 2)
                occ_cnt = occ * self.occ_idx
                occ = torch.argmax(occ_cnt, dim=-1).unsqueeze(1).float() / nb_voxels[1]

                occ_goal = occ_goal.permute(0, 1, 3, 2)
                occ_goal_cnt = occ_goal * self.occ_idx
                occ_goal = torch.argmax(occ_goal_cnt, dim=-1).unsqueeze(1).float() / nb_voxels[1]
                occ = torch.cat([occ, occ_goal], dim=1)
            elif self.scene_type == 'occ_two':
                occ = occ.permute(0, 2, 1, 3)
                occ_goal = occ_goal.permute(0, 2, 1, 3)
                occ = torch.cat([occ, occ_goal], dim=1)
            elif self.scene_type == 'occ_temp':
                occ = occ_goal.permute(0, 2, 1, 3)

        else:
            occ = None

        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)

        # model_mean = sqrt_recip_alphas_t * (
        #         x - betas_t * model(x, occ, t, text_emb, pelvis_goal, hand_goal, is_pick, need_scene, need_pelvis_dir, pi, need_pi, object_goal, is_object, obj_bps_data) / sqrt_one_minus_alphas_cumprod_t
        # )
        cond_model_output = model(x, occ, t, text_emb, pelvis_goal, hand_goal, is_loco, need_scene, need_pelvis_dir, pi, end_pi, seq_length, need_pi, object_goal, is_object, obj_bps_data, occ_list, occ_pos, is_sample=True)

        uncond_model_output = model(x, occ, t, text_emb, pelvis_goal, hand_goal, is_loco, need_scene, need_pelvis_dir, pi, end_pi, seq_length, need_pi, object_goal, is_object, obj_bps_data, occ_list, occ_pos, is_sample=True, is_uncondition=True)

        model_output = cond_model_output + self.w * (cond_model_output - uncond_model_output)
        
        # model_output = model(x, occ, t, text_emb, pelvis_goal, hand_goal, is_pick, need_scene, need_pelvis_dir, pi, end_pi, seq_length, need_pi, object_goal, is_object, obj_bps_data, occ_list, occ_pos, is_sample=True)

        model_mean = (
            extract(self.posterior_mean_coef1, t, x.shape) * model_output +
            extract(self.posterior_mean_coef2, t, x.shape) * x
        )

        if t_index == 0:
            return model_mean, occ, model_output
        else:
            # posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            # return model_mean + torch.sqrt(posterior_variance_t) * torch.randn_like(x), occ
            model_log_variance = extract(self.posterior_log_variance_clipped, t, x.shape)
            return model_mean + (0.5 * model_log_variance).exp() * torch.randn_like(x), occ, model_output
    

    def set_fixed_points(self, img, goal, fixed_points, mat, joint_id, fix_mode, fix_goal):
        '''
        set fixed points of goal and prefix frames

        img: [b, max_window_size, 3 * joint_num]
        fixed_points: [b, auto_regre_num, 3 * joint_num]

        '''

        if goal is not None and fix_goal:
            goal_len = goal.shape[1]
            goal = self.dataset.normalize_torch(transform_points(goal, torch.inverse(mat)))

            img[:, -goal_len:, joint_id * 3] = goal[:, :, 0]
            if joint_id != 0:
                img[:, -goal_len:, joint_id * 3 + 1] = goal[:, :, 1]
            img[:, -goal_len:, joint_id * 3 + 2] = goal[:, :, 2]

        if fixed_points is not None and fix_mode:
            img[:, :fixed_points.shape[1], :] = fixed_points

def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

class DDIMSolver:
    def __init__(self, alpha_cumprods, timesteps=500, ddim_timesteps=25):
        self.step_ratio = timesteps // ddim_timesteps
        self.ddim_timesteps = (
            np.arange(1, ddim_timesteps + 1) * self.step_ratio
        ).round().astype(np.int64) - 1
        self.ddim_alpha_cumprods = alpha_cumprods[self.ddim_timesteps]
        self.ddim_timesteps_prev = np.asarray([0] + self.ddim_timesteps[:-1].tolist())
        # self.ddim_alpha_cumprods_prev = np.asarray(
        #     [alpha_cumprods[0]] + alpha_cumprods[self.ddim_timesteps[:-1]].tolist()
        # )
        self.ddim_alpha_cumprods_prev = np.asarray(
            [1.0] + alpha_cumprods[self.ddim_timesteps[:-1]].tolist()
        )
        self.ddim_timesteps = torch.from_numpy(self.ddim_timesteps).long()
        self.ddim_timesteps_prev = torch.from_numpy(self.ddim_timesteps_prev).long()
        self.ddim_alpha_cumprods = torch.from_numpy(self.ddim_alpha_cumprods)
        self.ddim_alpha_cumprods_prev = torch.from_numpy(self.ddim_alpha_cumprods_prev)

    def to(self, device):
        self.ddim_timesteps = self.ddim_timesteps.to(device)
        self.ddim_timesteps_prev = self.ddim_timesteps_prev.to(device)

        self.ddim_alpha_cumprods = self.ddim_alpha_cumprods.to(device)
        self.ddim_alpha_cumprods_prev = self.ddim_alpha_cumprods_prev.to(device)
        return self

    def ddim_step(self, pred_x0, pred_noise, timestep_index):
        alpha_cumprod_prev = extract_into_tensor(
            self.ddim_alpha_cumprods_prev, timestep_index, pred_x0.shape
        )
        dir_xt = (1.0 - alpha_cumprod_prev).sqrt() * pred_noise
        x_prev = alpha_cumprod_prev.sqrt() * pred_x0 + dir_xt
        return x_prev

    def ddim_style_multiphase_pred(self, pred_x0, pred_noise, timestep_index, multiphase):
        inference_indices = np.linspace(
            0, len(self.ddim_timesteps), num=multiphase, endpoint=False
        )
        inference_indices = np.floor(inference_indices).astype(np.int64)
        inference_indices = (
            torch.from_numpy(inference_indices).long().to(self.ddim_timesteps.device)
        )
        expanded_timestep_index = timestep_index.unsqueeze(1).expand(
            -1, inference_indices.size(0)
        )
        valid_indices_mask = expanded_timestep_index >= inference_indices
        last_valid_index = valid_indices_mask.flip(dims=[1]).long().argmax(dim=1)
        last_valid_index = inference_indices.size(0) - 1 - last_valid_index
        timestep_index = inference_indices[last_valid_index]
        alpha_cumprod_prev = extract_into_tensor(
            self.ddim_alpha_cumprods_prev, timestep_index, pred_x0.shape
        )
        dir_xt = (1.0 - alpha_cumprod_prev).sqrt() * pred_noise
        x_prev = alpha_cumprod_prev.sqrt() * pred_x0 + dir_xt

        return x_prev, self.ddim_timesteps_prev[timestep_index]

class TimingModel(nn.Module):
    def __init__(self, dim_input, dim_model, num_heads, dropout_p, num_layers, language_feature_dim):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_model,
                                                   nhead=num_heads,
                                                   dim_feedforward=dim_model,
                                                   dropout=dropout_p,
                                                   activation="gelu")
        self.dim_model = dim_model
        self.positional_encoder = PositionalEncoding(
            dim_model=dim_model, dropout_p=dropout_p, max_len=5000
        )

        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                 num_layers=num_layers
                                                 )
        self.embedding_input = nn.Linear(dim_input, dim_model)
        self.out = nn.Linear(dim_model, 1)
        self.sigmoid = nn.Sigmoid()

        self.embed_timestep = TimestepEmbedder(self.dim_model, self.positional_encoder)

        self.embedding_language = LanguageEncoder(dim_output=dim_model, dim_input=language_feature_dim)

    def forward(self, x, text_emb, pi):
        need_pi = torch.ones_like(pi, dtype=torch.bool, device=pi.device)
        language_emb = self.embedding_language(text_emb, pi, need_pi)
        language_emb = language_emb.permute(1, 0, 2)

        x = x.permute(1, 0, 2)
        x = self.embedding_input(x) * math.sqrt(self.dim_model)

        x = torch.cat((language_emb, x), dim=0)
        x = self.positional_encoder(x)
        x = self.transformer(x)
        output = self.out(x)[-1]

        return output


class Unet(nn.Module):
    def __init__(
            self,
            dim_model,
            num_heads,
            num_layers,
            dropout_p,
            dim_input,
            dim_output,
            nb_voxels=None,
            temp_voxel_num=3,  # new param, default keeps current behavior
            free_p=0.1,
            load_scene=True,
            load_language=True,
            load_hand_goal=True,
            load_pelvis_goal=True,
            load_object_goal=True,
            language_feature_dim=768,
            scene_type=None,
            **kwargs
    ):
        super().__init__()

        self.model_type = "TransformerEncoder"
        self.dim_model = dim_model
        self.load_scene = load_scene
        self.load_language = load_language
        self.load_hand_goal = load_hand_goal
        self.load_pelvis_goal = load_pelvis_goal
        self.load_object_goal = load_object_goal
        self.scene_type = scene_type
        self.temp_voxel_num = temp_voxel_num  # store the number of temporal voxels

        if self.scene_type == 'plane':
            vit_channels = 1
        elif self.scene_type == 'occ':
            vit_channels = nb_voxels[1]
        elif self.scene_type == 'plane_two':
            vit_channels = 2
        elif self.scene_type == 'occ_two':
            vit_channels = 2*nb_voxels[1]
        elif self.scene_type == 'occ_temp':
            vit_channels = nb_voxels[1]

        if self.load_scene:
            self.scene_embedding = ViT(
                image_size=nb_voxels[0],
                patch_size=8,
                channels=vit_channels,
                num_classes=dim_model,
                dim=512,
                depth=6,
                heads=16,
                mlp_dim=1024,
                dropout=0.1,
                emb_dropout=0.1
            )
        self.free_p = free_p
        self.positional_encoder = PositionalEncoding(
            dim_model=dim_model, dropout_p=dropout_p, max_len=5000
        )

        self.embedding_input = nn.Linear(dim_input, dim_model)
        self.embedding_output = nn.Linear(dim_output, dim_model)

        if self.load_language:
            self.embedding_language = LanguageEncoder(dim_output=dim_model, dim_input=language_feature_dim)

        if self.load_hand_goal:
            self.embedding_hand_goal = GoalEncoder(mode='hand', dim_output=dim_model)

        if self.load_pelvis_goal:
            self.embedding_pelvis_goal = GoalEncoder(mode='pelvis', dim_output=dim_model)

        if self.load_object_goal:
            self.embedding_object_goal = GoalEncoder(mode='object', dim_output=dim_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_model,
                                                   nhead=num_heads,
                                                   dim_feedforward=dim_model,
                                                   dropout=dropout_p,
                                                   activation="gelu")

        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                 num_layers=num_layers
        )

        self.out = nn.Linear(dim_model, dim_output)

        self.embed_timestep = TimestepEmbedder(self.dim_model, self.positional_encoder)

        self.bps_encoder = nn.Sequential(
            nn.Linear(in_features=1024*3, out_features=768),
            nn.ReLU(),
            nn.Linear(in_features=768, out_features=self.dim_model),
            )
        
        # add the CFG scale embedding module
        self.cfg_scale_embedding = CFGScaleEmbedding(dim_model)
        
        self.division_term = torch.exp(
            torch.arange(0, dim_model//2, 2).float() * (-math.log(10000.0)) / (dim_model//2))  # 1000^(2i/dim_model)
        # self.register_buffer("division_term", division_term)

    def encode_2d_coordinate(self, pos, dim_model=512):
        # pos: [b, 2]
        # dim_model: int
        # return: [b, dim_model]
        pe_x = torch.zeros(pos.shape[0], dim_model//2).to(pos.device)
        pe_x[:, 0::2] = torch.sin(pos[:, 0:1] * self.division_term.to(pos.device))
        pe_x[:, 1::2] = torch.cos(pos[:, 0:1] * self.division_term.to(pos.device))

        pe_y = torch.zeros(pos.shape[0], dim_model//2).to(pos.device)
        pe_y[:, 0::2] = torch.sin(pos[:, 1:] * self.division_term.to(pos.device))
        pe_y[:, 1::2] = torch.cos(pos[:, 1:] * self.division_term.to(pos.device))

        return torch.cat((pe_x, pe_y), dim=1)[:, None, :] / math.sqrt(dim_model // 2)

    def forward(self, x, cond, timesteps, text_emb, pelvis_goal, hand_goal, is_loco, need_scene, need_pelvis_dir, \
                pi, end_pi, seq_length, need_pi, object_goal, is_object, obj_bps_data, occ_list, occ_pos, is_sample=False, is_uncondition=False, mask_timestep=10, cfg_scale=None):
        """
        Forward function, ensures all inputs have the correct type and device
        
        Args:
            x: input human and object motion [batch_size, seq_len, dim_input]
            cond: scene condition
            timesteps: timestep
            text_emb: text embedding
            pelvis_goal: pelvis goal position
            hand_goal: hand goal position
            is_pick: pick flag
            need_scene: scene-needed flag
            need_pelvis_dir: pelvis-direction-needed flag
            pi: progress indicator
            need_pi: progress-indicator-needed flag
            object_goal: object goal position
            is_object: object-present flag
            obj_bps_data: object BPS data [batch_size, 1024, 3]
        
        Returns:
            output: predicted noise
        """
        # ensure all inputs are float type
        x = x.to(dtype=torch.float32)
        self.batch_size = x.shape[0]
        
        if cond is not None:
            cond = cond.to(dtype=torch.float32)
        timesteps = timesteps.to(dtype=torch.long)
        text_emb = text_emb.to(dtype=torch.float32)
        pelvis_goal = pelvis_goal.to(dtype=torch.float32)
        hand_goal = hand_goal.to(dtype=torch.float32)
        object_goal = object_goal.to(dtype=torch.float32)
        obj_bps_data = obj_bps_data.to(dtype=torch.float32)
        
        t_emb = self.embed_timestep(timesteps)  # [b, 1, d]
        
        # if a CFG scale is provided, add the CFG embedding
        if cfg_scale is not None:
            if int(timesteps[0]) == 499 or is_uncondition: # todo: this should adapt to the timestep
                cfg_scale = torch.full((self.batch_size, 1), -1.0, device=x.device)
            cfg_emb = self.cfg_scale_embedding(cfg_scale)
            # add the CFG embedding to the timestep embedding
            t_emb = t_emb + cfg_emb.unsqueeze(1)

        if not self.load_scene:
            scene_emb = torch.zeros_like(t_emb)
            scene_emb_0 = torch.zeros_like(t_emb)
            scene_emb_1 = torch.zeros_like(t_emb)
            scene_emb_2 = torch.zeros_like(t_emb)
            scene_emb_3 = torch.zeros_like(t_emb)
        else:
            scene_emb = self.scene_embedding(cond).reshape(-1, 1, self.dim_model)
            scene_emb += self.encode_2d_coordinate(occ_pos[0], self.dim_model)

            if self.scene_type == 'occ_temp':
                scene_all = self.scene_embedding(occ_list).reshape(-1, 1, self.dim_model)
                
                # dynamically handle scene embedding
                scene_embs = []
                if self.temp_voxel_num == 0:
                    # current frame only, no temporal voxels
                    scene_emb_0 = scene_all
                    scene_emb_0 += self.encode_2d_coordinate(torch.zeros(self.batch_size, 2).to(occ_pos.device), self.dim_model)
                    scene_embs = [scene_emb_0]
                elif self.temp_voxel_num == 1:
                    # current frame + 1 temporal voxel
                    scene_emb_0 = scene_all[0:scene_all.shape[0]//2]
                    scene_emb_1 = scene_all[scene_all.shape[0]//2:]
                    scene_emb_0 += self.encode_2d_coordinate(torch.zeros(self.batch_size, 2).to(occ_pos.device), self.dim_model)
                    scene_emb_1 += self.encode_2d_coordinate(occ_pos[1], self.dim_model)
                    scene_embs = [scene_emb_0, scene_emb_1]
                elif self.temp_voxel_num == 2:
                    # current frame + 2 temporal voxels
                    scene_emb_0 = scene_all[0:scene_all.shape[0]//3]
                    scene_emb_1 = scene_all[scene_all.shape[0]//3:scene_all.shape[0]//3*2]
                    scene_emb_2 = scene_all[scene_all.shape[0]//3*2:]
                    
                    scene_emb_0 += self.encode_2d_coordinate(torch.zeros(self.batch_size, 2).to(occ_pos.device), self.dim_model)
                    scene_emb_1 += self.encode_2d_coordinate(occ_pos[1], self.dim_model)
                    scene_emb_2 += self.encode_2d_coordinate(occ_pos[2], self.dim_model)
                    scene_embs = [scene_emb_0, scene_emb_1, scene_emb_2]
                elif self.temp_voxel_num == 3:
                    # current frame + 3 temporal voxels (original implementation)
                    scene_emb_0 = scene_all[0:scene_all.shape[0]//4]
                    scene_emb_1 = scene_all[scene_all.shape[0]//4:scene_all.shape[0]//2]
                    scene_emb_2 = scene_all[scene_all.shape[0]//2:scene_all.shape[0]//4*3]
                    scene_emb_3 = scene_all[scene_all.shape[0]//4*3:scene_all.shape[0]]
                    
                    scene_emb_0 += self.encode_2d_coordinate(torch.zeros(self.batch_size, 2).to(occ_pos.device), self.dim_model)
                    scene_emb_1 += self.encode_2d_coordinate(occ_pos[1], self.dim_model)
                    scene_emb_2 += self.encode_2d_coordinate(occ_pos[2], self.dim_model)
                    scene_emb_3 += self.encode_2d_coordinate(occ_pos[3], self.dim_model)
                    scene_embs = [scene_emb_0, scene_emb_1, scene_emb_2, scene_emb_3]

                # handle dropout during sampling (temporal voxels only)
                if is_sample:
                    if int(timesteps[0]) == 499 or is_uncondition:
                        for i in range(1, len(scene_embs)):
                            scene_embs[i] = torch.zeros_like(t_emb)
                # when cfg_scale=-1, mask the scene condition (unconditional generation), Training
                elif cfg_scale is not None:
                    is_uncond = (cfg_scale == -1).squeeze(1)
                    if is_uncond.any():
                        mask = is_uncond.unsqueeze(1).unsqueeze(2)
                        for i in range(1, len(scene_embs)):
                            scene_embs[i] = torch.where(mask, torch.zeros_like(scene_embs[i]), scene_embs[i])
                else:
                    prob_mask = (torch.rand(scene_embs[0].size(0), 1, 1, device=scene_embs[0].device) < 0.1)
                    for i in range(1, len(scene_embs)):
                        scene_embs[i] = torch.where(prob_mask, torch.zeros_like(scene_embs[i]), scene_embs[i])
                
                # keep the original variable names for backward compatibility
                if self.temp_voxel_num == 0:
                    scene_emb_0 = scene_embs[0]
                    scene_emb_1 = torch.zeros_like(t_emb)
                    scene_emb_2 = torch.zeros_like(t_emb)
                    scene_emb_3 = torch.zeros_like(t_emb)
                elif self.temp_voxel_num == 1:
                    scene_emb_0 = scene_embs[0]
                    scene_emb_1 = scene_embs[1]
                    scene_emb_2 = torch.zeros_like(t_emb)
                    scene_emb_3 = torch.zeros_like(t_emb)
                elif self.temp_voxel_num == 2:
                    scene_emb_0 = scene_embs[0]
                    scene_emb_1 = scene_embs[1]
                    scene_emb_2 = scene_embs[2]
                    scene_emb_3 = torch.zeros_like(t_emb)
                elif self.temp_voxel_num == 3:
                    scene_emb_0 = scene_embs[0]
                    scene_emb_1 = scene_embs[1]
                    scene_emb_2 = scene_embs[2]
                    scene_emb_3 = scene_embs[3]

            else:
                scene_emb_0 = torch.zeros_like(t_emb)
                scene_emb_1 = torch.zeros_like(t_emb)
                scene_emb_2 = torch.zeros_like(t_emb)
                scene_emb_3 = torch.zeros_like(t_emb)

            not_need_scene = torch.logical_not(need_scene)
            scene_emb[not_need_scene] = 0.
            scene_emb_0[not_need_scene] = 0.
            scene_emb_1[not_need_scene] = 0.
            scene_emb_2[not_need_scene] = 0.
            scene_emb_3[not_need_scene] = 0.
            

        # print(timesteps[0], scene_emb_0[0,0,:20], scene_emb_1[0,0,:20], scene_emb_2[0,0,:20], scene_emb_3[0,0,:20])

        if not self.load_language:
            language_emb = torch.zeros_like(t_emb)
        else:
            language_emb = self.embedding_language(text_emb, pi, end_pi, seq_length, need_pi)

        # if not self.load_hand_goal:
        #     hand_goal_emb = torch.zeros_like(t_emb)
        # else:
        #     hand_goal_emb = self.embedding_hand_goal(hand_goal)
        #     is_not_pick = torch.logical_not(is_pick)
        #     hand_goal_emb[is_not_pick] = 0.

        # modified hand_goal handling logic: use hand_goal only when is_loco is False (as the goal for static scene interaction, replacing pelvis goal)
        hand_goal_emb = self.embedding_hand_goal(hand_goal)
        hand_goal_emb[is_loco] = 0.

        if not self.load_pelvis_goal:
            pelvis_goal_emb = torch.zeros_like(t_emb)
        else:
            pelvis_goal_emb = self.embedding_pelvis_goal(pelvis_goal)
            not_need_pelvis_dir = torch.logical_not(need_pelvis_dir)
            pelvis_goal_emb[not_need_pelvis_dir] = 0.

            # when is_loco is True, use pelvis goal; otherwise set pelvis goal to 0
            pelvis_goal_emb[torch.logical_not(is_loco)] = 0.
        # if not is_loco:
        #     import pdb; pdb.set_trace()
        if not self.load_object_goal:
            object_goal_emb = torch.zeros_like(t_emb)
        else:
            object_goal_emb = self.embedding_object_goal(object_goal)
            not_need_object = torch.logical_not(is_object)
            object_goal_emb[not_need_object] = 0.

        if not self.load_object_goal:
            obj_bps_data_emb = torch.zeros_like(t_emb)
        else:
            # ensure obj_bps_data is float type
            obj_bps_data = obj_bps_data.float()
            obj_bps_data_emb = obj_bps_data.reshape(-1, 1024*3)
            obj_bps_data_emb = self.bps_encoder(obj_bps_data_emb)
            obj_bps_data_emb = obj_bps_data_emb.reshape(-1, 1, self.dim_model)
            # for samples that do not need an object, set the object BPS feature to 0
            not_need_object = torch.logical_not(is_object)
            obj_bps_data_emb[not_need_object] = 0.

        t_emb = t_emb.permute(1, 0, 2)
        scene_emb = scene_emb.permute(1, 0, 2)
        scene_emb_0 = scene_emb_0.permute(1, 0, 2)
        scene_emb_1 = scene_emb_1.permute(1, 0, 2)
        scene_emb_2 = scene_emb_2.permute(1, 0, 2)
        scene_emb_3 = scene_emb_3.permute(1, 0, 2)
        language_emb = language_emb.permute(1, 0, 2)
        hand_goal_emb = hand_goal_emb.permute(1, 0, 2)
        pelvis_goal_emb = pelvis_goal_emb.permute(1, 0, 2)
        object_goal_emb = object_goal_emb.permute(1, 0, 2)
        obj_bps_data_emb = obj_bps_data_emb.permute(1, 0, 2)

        scene_emb = t_emb + scene_emb
        scene_emb_0 = t_emb + scene_emb_0
        scene_emb_1 = t_emb + scene_emb_1
        scene_emb_2 = t_emb + scene_emb_2
        scene_emb_3 = t_emb + scene_emb_3
        language_emb = t_emb + language_emb
        hand_goal_emb = t_emb + hand_goal_emb
        pelvis_goal_emb = t_emb + pelvis_goal_emb
        object_goal_emb = t_emb + object_goal_emb
        obj_bps_data_emb = t_emb + obj_bps_data_emb

        x = x.permute(1, 0, 2)
        x = self.embedding_input(x) * math.sqrt(self.dim_model)
        
        if self.scene_type == 'occ_temp':
            if self.temp_voxel_num == 0:
                # current frame only
                x = torch.cat((scene_emb, language_emb, hand_goal_emb, pelvis_goal_emb, 
                              object_goal_emb, obj_bps_data_emb, scene_emb_0, x), dim=0)
            elif self.temp_voxel_num == 1:
                # current frame + 1 temporal voxel
                x = torch.cat((scene_emb, language_emb, hand_goal_emb, pelvis_goal_emb, 
                              object_goal_emb, obj_bps_data_emb, scene_emb_0, x, 
                              scene_emb_1), dim=0)
            elif self.temp_voxel_num == 2:
                # current frame + 2 temporal voxels
                x = torch.cat((scene_emb, language_emb, hand_goal_emb, pelvis_goal_emb, 
                              object_goal_emb, obj_bps_data_emb, scene_emb_0, scene_emb_1, scene_emb_2, x), dim=0)
            elif self.temp_voxel_num == 3:
                # current frame + 3 temporal voxels (original implementation)
                x = torch.cat((scene_emb, language_emb, hand_goal_emb, pelvis_goal_emb, 
                              object_goal_emb, obj_bps_data_emb, scene_emb_0, x[0:5], 
                              scene_emb_1, x[5:10], scene_emb_2, x[10:15], 
                              scene_emb_3, x[15:]), dim=0)
        else:
            x = torch.cat((scene_emb, language_emb, hand_goal_emb, pelvis_goal_emb, object_goal_emb, obj_bps_data_emb, x), dim=0)
        
        x = self.positional_encoder(x)
        x = self.transformer(x)

        if self.scene_type == 'occ_temp':
            if self.temp_voxel_num == 0:
                # output indices: skip the first 7 (6 embeddings + 1 scene_emb_0)
                expected_len = 23  # 6 + 1 + 16
                assert x.shape[0] == expected_len, f"Expected {expected_len} but got {x.shape[0]}"
                x_index = list(range(7, 23))
            elif self.temp_voxel_num == 1:
                # output indices: skip the embedding and scene_emb positions
                expected_len = 24  # 6 + 2 + 16
                assert x.shape[0] == expected_len, f"Expected {expected_len} but got {x.shape[0]}"
                x_index = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
            elif self.temp_voxel_num == 2:
                # output indices: skip the embedding and scene_emb positions
                expected_len = 25  # 6 + 3 + 16
                assert x.shape[0] == expected_len, f"Expected {expected_len} but got {x.shape[0]}"
                x_index = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
            elif self.temp_voxel_num == 3:
                # original implementation
                expected_len = 26  # 6 + 4 + 16
                assert x.shape[0] == expected_len, f"Expected {expected_len} but got {x.shape[0]}"
                x_index = [7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 25]
            
            output = self.out(x)[x_index]
        else:
            output = self.out(x)[6:]
        output = output.permute(1, 0, 2)

        return output


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        # max_len determines how far the position can have an effect on a token (window)

        # Info
        self.dropout = nn.Dropout(dropout_p)

        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).reshape(-1, 1)  # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(
            torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model)  # 1000^(2i/dim_model)

        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)

        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(inplace=False),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pos_encoding[timesteps])


class CFGScaleEmbedding(nn.Module):
    """Fourier embedding module for the CFG scale"""
    def __init__(self, dim_model, max_period=10000, init_scale=0.0001):
        super().__init__()
        self.dim_model = dim_model
        self.max_period = max_period
        
        # Fourier frequencies
        half_dim = dim_model // 2
        self.freqs = torch.exp(
            -math.log(max_period) * torch.arange(half_dim) / half_dim
        )
        
        # projection layer
        self.proj = nn.Linear(dim_model, dim_model)
        # small-value initialization
        nn.init.normal_(self.proj.weight, mean=0.0, std=init_scale)
        nn.init.constant_(self.proj.bias, 0.0)
        
    def forward(self, w):
        """
        Args:
            w: CFG scale [batch_size, 1]
        Returns:
            w_emb: CFG embedding [batch_size, dim_model]
        """
        # expand the w dimension
        w = w.unsqueeze(-1) * 1000  # [batch_size, 1, 1]
        
        # compute Fourier features
        self.freqs = self.freqs.to(w.device)
        args = w * self.freqs  # [batch_size, 1, half_dim]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [batch_size, 1, dim_model]
        embedding = embedding.squeeze(1)  # [batch_size, dim_model]
        
        # pass through the zero-initialized projection layer
        w_emb = self.proj(embedding)
        
        return w_emb


class ProgressIndicatorEmbedding(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

    def forward(self, timesteps):
        return self.sequence_pos_encoder.pos_encoding[timesteps]


class DynamicProgressEmbedding(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder, dropout_p=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.dropout = nn.Dropout(dropout_p)
        # MLP that fuses start and end position information
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim * 3, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )
        self.sequence_pos_encoder = sequence_pos_encoder
        
    def forward(self, timesteps_start, timesteps_end, seq_length):
        # obtain the encodings of the start and end positions
        start_encoding = self.sequence_pos_encoder.pos_encoding[timesteps_start]  # [B, D]
        end_encoding = self.sequence_pos_encoder.pos_encoding[timesteps_end]      # [B, D]
        len_encoding = self.sequence_pos_encoder.pos_encoding[seq_length]

        # fuse the two position encodings
        combined = torch.cat([start_encoding, end_encoding, len_encoding], dim=-1)  # [B, 3D]
        progress_embedding = self.fusion(combined)      # [B, D]
        
        return self.dropout(progress_embedding)


class ActionTransformerEncoder(nn.Module):
    def __init__(self,
                 action_number,
                 dim_model,
                 nhead,
                 num_layers,
                 dim_feedforward,
                 dropout_p,
                 activation="gelu") -> None:
        super().__init__()
        self.positional_encoder = PositionalEncoding(
            dim_model=dim_model, dropout_p=dropout_p, max_len=5000
        )
        self.input_embedder = nn.Linear(action_number, dim_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_model,
                                                    nhead=nhead,
                                                    dim_feedforward=dim_feedforward,
                                                    dropout=dropout_p,
                                                    activation=activation)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer,
                                                 num_layers=num_layers
        )

    def forward(self, x):
        x = x.permute(1, 0, 2)
        x = self.input_embedder(x)
        x = self.positional_encoder(x)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)
        x = torch.mean(x, dim=1, keepdim=True)
        return x
    

class LanguageEncoder(nn.Module):
    def __init__(self, dim_output, dim_input, **kwargs):
        super().__init__()
        self.dim_model = dim_output

        self.embedding_input1 = nn.Sequential(
            nn.Linear(dim_input, dim_output),
            nn.SiLU(inplace=False),
            nn.Linear(dim_output, dim_output),
        )

        self.embedding_input2 = nn.Sequential(
            nn.Linear(dim_output, dim_output),
            nn.SiLU(inplace=False),
            nn.Linear(dim_output, dim_output),
        )

        self.positional_encoder = PositionalEncoding(
            dim_model=dim_output, dropout_p=0.1, max_len=5000
        )

        # self.embed_pi = ProgressIndicatorEmbedding(dim_output, self.positional_encoder)
        self.embed_pi = DynamicProgressEmbedding(dim_output, self.positional_encoder)

    def forward(self, x, pi, end_pi, seq_length, need_pi):
        # x.shape: [b, 1, 768]

        x = self.embedding_input1(x)
        pi = self.embed_pi(pi, end_pi, seq_length)

        # normalization
        pi = pi / np.sqrt(self.dim_model // 2)
        not_need_pi = torch.logical_not(need_pi)
        pi[not_need_pi] = 0.
        x = x + pi
        x = self.embedding_input2(x)
        return x

class GoalEncoder(nn.Module):
    def __init__(self, mode, dim_output, **kwargs):
        super().__init__()

        self.mode = mode
        if mode == 'pelvis':
            self.embedding_input = nn.Sequential(nn.Linear(2, dim_output),
                                                    nn.SiLU(inplace=False),
                                                    nn.Linear(dim_output, dim_output))
        elif mode == 'hand':
            self.embedding_input = nn.Sequential(nn.Linear(3, dim_output),
                                                    nn.SiLU(inplace=False),
                                                    nn.Linear(dim_output, dim_output))
        elif mode == 'object':
            self.embedding_input = nn.Sequential(nn.Linear(3, dim_output),
                                                    nn.SiLU(inplace=False),
                                                    nn.Linear(dim_output, dim_output))

    def forward(self, x):
        # x.shape: [b, 3] (includes object_goal)
        if self.mode == 'pelvis':
            x = x[..., [0, 2]]  # use only the x and z coordinates
        x = self.embedding_input(x)
        x = x.reshape(-1, 1, x.shape[-1])
        return x