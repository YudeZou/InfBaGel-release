import os
import pickle as pkl
import pickle
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from scipy.spatial.transform import Rotation as R
import trimesh

from utils import *
from constants import *
from datasets.infbagel import InfBaGelDataset
from guidance_loss import *
import json
from eval_metrics import *

import pytorch3d.transforms as transforms
from constants import *
import smplx

def run_smplx_model(pose_pred, transl, betas, gender, joints_ind=None):
    # pose_pred: [b*s*42, 22, 3]
    # transl: [b*s*42, 3]
    # joints_ind: [28]
    # joints_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 28, 40, 43]
    joints_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 28, 43]
    device = pose_pred.device

    smpl_model = smplx.create(SMPL_DIR, model_type='smplx',
                              gender=gender, ext='npz',
                              num_betas=16,
                              use_pca=False,
                              create_global_orient=True,
                              create_body_pose=True,
                              create_betas=True,
                              create_left_hand_pose=True,
                              create_right_hand_pose=True,
                              flat_hand_mean=True,
                              create_expression=True,
                              create_jaw_pose=True,
                              create_leye_pose=True,
                              create_reye_pose=True,
                              create_transl=True,
                              batch_size=pose_pred.shape[0],
                              ).to(device)
    
    smpl_output = smpl_model(transl=transl, body_pose=pose_pred[:, 1:], global_orient=pose_pred[:, :1], betas=betas, return_verts=True)
    
    return smpl_output.vertices, smpl_output.joints[:, joints_ind].reshape(pose_pred.shape[0], -1, 3)

def compute_metrics(sampler_body, cfg, points_orig, global_rot_6d, points_gt_orig, obj_trans_pred, obj_trans_gt, obj_rot_mat_pred, obj_rot_mat_gt, start_point_all_gt, start_object_trans_all_gt, end_object_trans_all_gt, xy_points_all_gt, seq_name_dict, obj_rest_verts, rest_human_offsets_all, transl_all, betas_all, gender_all):
    # points_orig: 534 X 7 X T X 84
    # global_rot_6d: 534 X 7 X T X 132
    # points_gt_orig: N X T X 84
    # obj_trans_pred: 534 X 7 X T X 3
    # obj_trans_gt: N X T X 3
    # obj_rot_mat_pred: 534 X 7 X T X 9
    # obj_rot_mat_gt: N X T X 9

    device = cfg.device

    object_sdf_root = '/cpfs04/shared/sport/zouyude/code/chois_release/processed_data/rest_object_sdf_256_npy_files'
    obj_sdf = {}
    obj_sdf_json = {}
    for file in os.listdir(object_sdf_root):
        if not file.endswith('.npy'):
            continue
        obj_name = file.split('.')[0]
        sdf_path = os.path.join(object_sdf_root, file)
        obj_sdf[obj_name] = np.load(sdf_path)
        obj_sdf_json[obj_name] = json.load(open(os.path.join(object_sdf_root, f'{file[:-4]}.json'), 'r'))

    batch_size = points_orig.shape[0]
    
    # seg_len = pkl.load(open('/cpfs04/shared/sport/zouyude/data_144/chois_lingo/test/language_motion_dict/seq_len.pkl', 'rb'))
    seg_id_dict = pkl.load(open('/cpfs04/shared/sport/zouyude/data_3/chois_lingo/test/language_motion_dict/seq_id.pkl', 'rb'))
    seg_id_dict = [0] + list(seg_id_dict.values())
    
    scene_name2file = pkl.load(open('/cpfs04/shared/sport/zouyude/data_144/chois_lingo/test/scene_name2file.pkl', 'rb'))
    with open('/cpfs04/shared/sport/zouyude/code/lingo-release/smpl_models/MANO_SMPLX_vertex_ids.pkl', 'rb') as f:
        idxs_data = pkl.load(f)
    hand_idxs = np.concatenate([idxs_data['left_hand'], idxs_data['right_hand']]) # 1556

    points_all_48 = torch.zeros(0, cfg.max_window_size*cfg.interp_s-6, 3*(cfg.dataset.nb_joints -4)).to(device)
    object_trans_all_48 = torch.zeros(0, cfg.max_window_size*cfg.interp_s-6, 3).to(device)
    object_rot_mat_all_48 = torch.zeros(0, cfg.max_window_size*cfg.interp_s-6, 9).to(device)

    start_point_all = torch.zeros(0, 1, 3*cfg.dataset.nb_joints).to(device)
    start_object_trans_all = torch.zeros(0, 1, 3).to(device)
    end_object_trans_all = torch.zeros(0, 1, 3).to(device)
    xy_points_all = torch.zeros(0, 1, 3*cfg.dataset.nb_joints).to(device)

    pose_pred_all = torch.zeros(0, cfg.max_window_size*cfg.interp_s-6, 22, 3).to(device)

    feet_height_list = []
    foot_sliding_list = []

    gt_contact_percent_list = []
    pred_contact_percent_list = []
    contact_acc_list = []
    contact_precision_list = []
    contact_recall_list = []
    contact_f1_list = []

    hand_pen_loss_list = []
    hand_pen_ratio_list = []

    human_pen_loss_list = []
    human_pen_ratio_list = []

    human_voxel_loss_list = []
    human_voxel_ratio_list = []
    object_voxel_loss_list = []
    object_voxel_ratio_list = []

    sum_len = 0
    seg_id_true = 0
    for seg_id in range(0, len(seg_id_dict)-1):
    # for seg_id in range(250, 300): # debug
        obj_name = seq_name_dict[seg_id_true].split('_')[1]

        points_orig_seg = points_orig[seg_id_true][:3].reshape(-1, cfg.max_window_size, 3*(cfg.dataset.nb_joints))
        points_orig_seg = points_orig_seg[:, cfg.auto_regre_num:, :]

        global_rot_6d_seg = global_rot_6d[seg_id_true][:3].reshape(-1, cfg.max_window_size, 22*6)
        global_rot_6d_seg = global_rot_6d_seg[:, cfg.auto_regre_num:, :]

        obj_trans_pred_seg = obj_trans_pred[seg_id_true][:3].reshape(-1, cfg.max_window_size, 3)
        obj_trans_pred_seg = obj_trans_pred_seg[:, cfg.auto_regre_num:, :]

        obj_rot_mat_pred_seg = obj_rot_mat_pred[seg_id_true][:3].reshape(-1, cfg.max_window_size, 9)
        obj_rot_mat_pred_seg = obj_rot_mat_pred_seg[:, cfg.auto_regre_num:, :]

        points_gt_orig_seg = points_gt_orig[sum_len:sum_len+3].reshape(-1, cfg.dataset.nb_joints-4, 3) # T * J * 3
        obj_trans_gt_seg = obj_trans_gt[sum_len:sum_len+3].reshape(-1, 3)
        obj_rot_mat_gt_seg = obj_rot_mat_gt[sum_len:sum_len+3].reshape(-1, 3, 3)
        sum_len += 3

        obj_rest_verts_gt_seg = load_object_geometry_w_rest_geo(obj_rot_mat_gt_seg, obj_trans_gt_seg, obj_rest_verts[obj_name])

        start_point_all = torch.cat((start_point_all, points_orig_seg[0:1, 0, :].unsqueeze(1)), dim=0)
        start_object_trans_all = torch.cat((start_object_trans_all, obj_trans_pred_seg[0:1, 0, :].unsqueeze(1)), dim=0)
        end_object_trans_all = torch.cat((end_object_trans_all, obj_trans_pred_seg[-1:, -1, :].unsqueeze(1)), dim=0)
        xy_points_all = torch.cat((xy_points_all, points_orig_seg[:, -2, :].unsqueeze(1)), dim=0)
        
        joints = interpolate_joints(points_orig_seg.reshape(-1, 3*(cfg.dataset.nb_joints)), scale=cfg.interp_s)
        obj_trans, obj_rot_mat = interp_object(obj_trans_pred_seg.reshape(-1, 3).cpu().numpy(), obj_rot_mat_pred_seg.reshape(-1, 9).cpu().numpy(), cfg.interp_s)
        
        joints = joints.reshape(-1, (cfg.max_window_size-2)*cfg.interp_s, 3*(cfg.dataset.nb_joints)) # S * 42 * 84
        
        # FK to get joint positions.
        curr_seq_local_jpos = rest_human_offsets_all[seg_id_true][:cfg.max_window_size*cfg.interp_s-6] # [42, 24, 3]
        curr_seq_local_jpos = curr_seq_local_jpos.repeat(joints.shape[0], 1, 1, 1) # [S, 42, 24, 3]
        curr_seq_local_jpos[:, :, 0, :] = joints.reshape(-1, 42, 28, 3)[:, :, 0, :]
        
        global_jrot_mat_seg = transforms.rotation_6d_to_matrix(global_rot_6d_seg.reshape(-1, 22, 6)) # [S*14, 22, 3, 3]
        local_jrot_mat_seg = sampler_body.dataset.quat_ik_torch(global_jrot_mat_seg.reshape(-1, 22, 3, 3))

        local_jrot_q_seg = transforms.matrix_to_quaternion(local_jrot_mat_seg)
        local_jrot_q_48 = interp_jrot(local_jrot_q_seg, 3).reshape(-1, cfg.max_window_size*cfg.interp_s-6, 22, 4) # [S, 42, 22, 4]
        
        local_jrot_mat_48 = transforms.quaternion_to_matrix(local_jrot_q_48).reshape(-1, 22, 3, 3) # [S*42, 22, 3, 3]

        _, human_jnts_48 = sampler_body.dataset.quat_fk_torch(local_jrot_mat_48, curr_seq_local_jpos.reshape(-1, 24, 3)) # [S*42, 24, 3]
        human_jnts_48 = human_jnts_48.detach()

        # reconstruct human verts and joints
        transl = transl_all[seg_id_true] # 3
        betas = betas_all[seg_id_true] # 16
        gender = gender_all[seg_id_true] # 'male'
        root_trans = yup_to_zup(joints.reshape(-1, 28, 3)[:, 0, :] + transl)
        pose_pred = yup_to_zup(transforms.matrix_to_axis_angle(local_jrot_mat_48).reshape(-1, 22, 3))
        
        verts, joints = run_smplx_model(pose_pred, root_trans, betas[None].repeat(root_trans.shape[0], 1), gender, joints_ind=None)
        verts, joints = zup_to_yup(verts), zup_to_yup(joints)
        
        points_all_48 = torch.cat((points_all_48, human_jnts_48.reshape(-1, cfg.max_window_size*cfg.interp_s-6, 3*(cfg.dataset.nb_joints-4))), dim=0)

        obj_trans = obj_trans.reshape(-1, (cfg.max_window_size-2)*cfg.interp_s, 3)
        obj_rot_mat = obj_rot_mat.reshape(-1, (cfg.max_window_size-2)*cfg.interp_s, 9)

        object_trans_all_48 = torch.cat((object_trans_all_48, torch.from_numpy(obj_trans).to(device)), dim=0)
        object_rot_mat_all_48 = torch.cat((object_rot_mat_all_48, torch.from_numpy(obj_rot_mat).to(device)), dim=0)
        
        model_name = cfg.ckpt_path.split('/')[-1]
        if not os.path.exists(os.path.join('t2m_results_48', cfg.exp_name, model_name[:-4])):
            os.makedirs(os.path.join('t2m_results_48', cfg.exp_name, model_name[:-4]))
        np.savez(os.path.join('t2m_results_48', cfg.exp_name, model_name[:-4], f"{seq_name_dict[seg_id_true]}.npz"), seq_name=seq_name_dict[seg_id_true], \
                    global_jpos=yup_to_zup(human_jnts_48).cpu().numpy()) # T X 24 X 3
        
        # Save motion parameters for mesh recovery
        if cfg.save_motion_params:
            motion_params_dir = os.path.join('motion_params_922', cfg.exp_name, model_name[:-4])
            if not os.path.exists(motion_params_dir):
                os.makedirs(motion_params_dir)

            # Prepare complete motion parameters
            motion_params = {
                'seq_name': seq_name_dict[seg_id_true],
                'human_motion': {
                    'pose_pred': pose_pred.cpu().numpy(),  # [T, 22, 3] - SMPL body pose (axis-angle)
                    'root_trans': root_trans.cpu().numpy(),  # [T, 3] - root translation
                    'betas': betas.cpu().numpy(),  # [16] - SMPL shape parameters
                    'gender': gender  # string - gender info
                },
                'object_motion': {
                    'obj_trans': obj_trans,  # [T, 3] - object translation
                    'obj_rot_mat': obj_rot_mat,  # [T, 3, 3] - object rotation matrices
                    'obj_name': obj_name  # string - object name
                }
            }

            # Save as pickle file for complete data preservation
            with open(os.path.join(motion_params_dir, f"{seq_name_dict[seg_id_true]}_motion_params.pkl"), 'wb') as f:
                pickle.dump(motion_params, f)

        floor_height = determine_floor_height_and_contacts(human_jnts_48.cpu().numpy().reshape(-1, 24, 3))
        foot_sliding = compute_foot_sliding_for_smpl(human_jnts_48.cpu().numpy().reshape(-1, 24, 3), floor_height)
        feet_height_list.append(floor_height)
        foot_sliding_list.append(foot_sliding)

        obj_rest_verts_pred_seg = load_object_geometry_w_rest_geo(torch.from_numpy(obj_rot_mat).reshape(-1, 3, 3).float().to(device), torch.from_numpy(obj_trans).reshape(-1, 3).float().to(device), obj_rest_verts[obj_name])

        gt_contact_percent, pred_contact_percent, contact_acc, contact_precision, contact_recall, contact_f1 = \
            compute_hand_object_interaction(human_jnts_48.reshape(-1, 24, 3), points_gt_orig_seg, obj_rest_verts_pred_seg, obj_rest_verts_gt_seg)
        
        gt_contact_percent_list.append(gt_contact_percent)
        pred_contact_percent_list.append(pred_contact_percent)
        contact_acc_list.append(contact_acc)
        contact_precision_list.append(contact_precision)
        contact_recall_list.append(contact_recall)
        contact_f1_list.append(contact_f1)

        verts = verts.reshape(-1, 10475, 3)
        hand_verts = verts[:, hand_idxs, :]

        obj_trans = torch.from_numpy(obj_trans).reshape(-1, 3).to(device)
        obj_rot_mat = torch.from_numpy(obj_rot_mat).reshape(-1, 3, 3).to(device)

        if obj_name not in ['woodchair', 'whitechair', 'largebox', 'largetable', 'plasticbox', 'trashcan']:   
            hand_pen_loss, hand_pen_ratio = compute_collision(yup_to_zup(hand_verts), obj_sdf[obj_name], obj_sdf_json[obj_name], yup_to_zup_rotation_matrix(obj_rot_mat), yup_to_zup(obj_trans))
            hand_pen_loss_list.append(hand_pen_loss)
            hand_pen_ratio_list.append(hand_pen_ratio)

            human_pen_loss, human_pen_ratio = compute_collision(yup_to_zup(verts), obj_sdf[obj_name], obj_sdf_json[obj_name], yup_to_zup_rotation_matrix(obj_rot_mat), yup_to_zup(obj_trans))
            human_pen_loss_list.append(human_pen_loss)
            human_pen_ratio_list.append(human_pen_ratio)

            print(f'scene_name: {seq_name_dict[seg_id_true]}, hand_pen_loss: {hand_pen_loss}, hand_pen_ratio: {hand_pen_ratio}, human_pen_loss: {human_pen_loss}, human_pen_ratio: {human_pen_ratio}')

        seg_id_true += 1
    
    hand_pen_loss = np.array(hand_pen_loss_list).mean()
    hand_pen_ratio = np.array(hand_pen_ratio_list).mean()
    human_pen_loss = np.array(human_pen_loss_list).mean()
    human_pen_ratio = np.array(human_pen_ratio_list).mean()

    mpjpe, trans_dist, obj_trans_dist, obj_rot_dist = compute_gt_difference(points_all_48, points_gt_orig, object_trans_all_48, obj_trans_gt, object_rot_mat_all_48, obj_rot_mat_gt)

    start_point_err, start_obj_trans_err, end_obj_trans_err, xy_points_err = compute_condition_matching(start_point_all, start_object_trans_all, end_object_trans_all, xy_points_all, start_point_all_gt, start_object_trans_all_gt, end_object_trans_all_gt, xy_points_all_gt)

    feet_height = np.array(feet_height_list).mean()
    foot_sliding = np.array(foot_sliding_list).mean()

    contact_precision = np.array(contact_precision_list).mean()
    contact_recall = np.array(contact_recall_list).mean()
    contact_f1 = np.array(contact_f1_list).mean()
    contact_percent = np.array(pred_contact_percent_list).mean()
    gt_contact_percent = np.array(gt_contact_percent_list).mean()
    contact_acc = np.array(contact_acc_list).mean()

    metrics = {
        'end_obj_trans_err': end_obj_trans_err,
        'xy_points_err': xy_points_err,
        'feet_height': feet_height,
        'foot_sliding': foot_sliding,
        'contact_precision': contact_precision,
        'contact_recall': contact_recall,
        'contact_f1': contact_f1,
        'contact_percent': contact_percent,
        'mpjpe': mpjpe,
        'trans_dist': trans_dist,
        'obj_trans_dist': obj_trans_dist,
        'obj_rot_dist': obj_rot_dist,
        'gt_contact_percent': gt_contact_percent,
        'contact_acc': contact_acc,
        'hand_pen_loss': hand_pen_loss,
        'hand_pen_ratio': hand_pen_ratio,
        'human_pen_loss': human_pen_loss,
        'human_pen_ratio': human_pen_ratio
    }

    return metrics


def sample_step(cfg, mat, fixed_points, sampler, scene_flag, text_clip_embedding, pelvis_goal, hand_goal, object_goal, 
                is_pick, need_scene, need_pelvis_dir, pi, end_pi, seq_length, need_pi, is_loco, is_object, obj_bps_data, object_points, obj_rest_verts, obj_vert_normals, seq_name_dict, obj_rot_mat_ref_first_step_batch, human_dict):
    batch_size = fixed_points.shape[0]
    object_goal_temp = object_goal.clone()
    pelvis_goal = transform_points(pelvis_goal.reshape(batch_size, 1, 3), torch.inverse(mat)).reshape(batch_size, 1, 3) # convert to local coordinates
    hand_goal = transform_points(hand_goal.reshape(batch_size, 1, 3), torch.inverse(mat)).reshape(batch_size, 1, 3)
    object_goal = transform_points(object_goal.reshape(batch_size, 1, 3), torch.inverse(mat)).reshape(batch_size, 1, 3)
    # print(f'pelvis_goal: {pelvis_goal}', 'hand_goal: ', hand_goal, 'object_goal: ', object_goal, 'pi: ', pi, 'need_pi: ', need_pi, 'need_scene: ', need_scene, 'need_pelvis_dir: ', need_pelvis_dir, 'is_object: ', is_object)

    if not cfg.add_object_voxel:
        object_points = None

    # switch via cfg.sample_type: cm -> consistency model sampling; dm -> diffusion model sampling
    if cfg.sample_type == 'cm':
        guidance_fn = apply_hoi_guidance_loss
        samples, occs = sampler.cm_sample_loop(fixed_points, mat, scene_flag, text_clip_embedding, pelvis_goal, hand_goal,
                                            object_goal, is_pick, need_scene, need_pelvis_dir, pi, end_pi, seq_length, need_pi, is_loco, is_object, obj_bps_data, object_points, obj_rot_mat_ref_first_step_batch, obj_rest_verts, obj_vert_normals, seq_name_dict, human_dict, guidance_fn, cfg.guidance_weight, object_only=True, w=cfg.w)
    else:
        samples, occs = sampler.p_sample_loop(fixed_points, mat, scene_flag, text_clip_embedding, pelvis_goal, hand_goal,
                                            object_goal, is_pick, need_scene, need_pelvis_dir, pi, end_pi, seq_length, need_pi, is_loco, is_object, obj_bps_data, object_points, obj_rot_mat_ref_first_step_batch, obj_rest_verts, seq_name_dict, object_only=True)

    points_gene = samples[-1]
    
    points = points_gene[:, :, :cfg.dataset.nb_joints*3].reshape(batch_size, cfg.max_window_size, cfg.dataset.nb_joints*3)
    points_orig = transform_points(sampler.dataset.denormalize_torch(points), mat)

    global_rot_6d = points_gene[:, :, 84:216].reshape(batch_size, cfg.max_window_size, 22*6)

    obj_trans = points_gene[:, :, 216:219].reshape(batch_size, cfg.max_window_size, 3)
    obj_rot = points_gene[:, :, 219:228].reshape(batch_size, cfg.max_window_size, 3, 3)
    obj_trans_orig = transform_points(sampler.dataset.denormalize_torch(obj_trans, is_object=True), mat)

    contact_label = points_gene[:, :, 228:232].reshape(batch_size, cfg.max_window_size, 4)
    # l_toe_height = points_orig.reshape(batch_size, cfg.max_window_size, cfg.dataset.nb_joints, 3)[:, :, 10, 1:2] # BS X T X 1
    # r_toe_height = points_orig.reshape(batch_size, cfg.max_window_size, cfg.dataset.nb_joints, 3)[:, :, 11, 1:2] # BS X T X 1
    # support_foot_height = torch.minimum(l_toe_height, r_toe_height) # BS X T X 1
    # end_obj_trans_err = torch.linalg.norm(obj_trans_orig[:, -1, :] - object_goal_temp, dim=-1).mean() * 100
    # import pdb; pdb.set_trace()

    info_dict = {
        'points_orig': points_orig.reshape(batch_size, cfg.max_window_size, 3*cfg.dataset.nb_joints),
        'obj_trans_orig': obj_trans_orig,
        'object_rot_mat': obj_rot.reshape(batch_size, cfg.max_window_size, 9),
        'contact_label': contact_label,
        'global_rot_6d': global_rot_6d,
        # 'pelvis_goal': transform_points(pelvis_goal.unsqueeze(1), mat).reshape(batch_size, 3), # global coordinates
        # 'pi': pi,
        # 'need_pi': need_pi,
        # 'need_scene': need_scene,
        # 'need_pelvis_dir': need_pelvis_dir,
        # 'scene_flag': scene_flag,
        # 'hand_goal': hand_goal,
        # 'is_pick': is_pick,
        # 'occ': occs[-1],
    }

    return info_dict


# aggregate evaluation results
def summarize_metrics(all_metrics):
    metrics_summary = {}

    # compute mean values
    for key in all_metrics[0].keys():
        values = [metrics[key] for metrics in all_metrics if key in metrics]
        if values:
            metrics_summary[key] = np.mean(values)
            metrics_summary[f"{key}_std"] = np.std(values)
    
    return metrics_summary

def get_mat(cfg, points):
    batch_size = points.shape[0]
    pelvis_new = points[:, -cfg.auto_regre_num, :9].cpu().numpy().reshape(batch_size, 3, 3)
    trans_mats = np.repeat(np.eye(4)[np.newaxis, :, :], batch_size, axis=0)
    for ip, pn in enumerate(pelvis_new):
        _, ret_R, ret_t = rigid_transform_3D(np.matrix(pn), rest_pelvis, False)
        ret_t[1] = 0.0
        rot_euler = R.from_matrix(ret_R).as_euler('zxy')
        shift_euler = np.array([0, 0, rot_euler[2]])
        shift_rot_matrix2 = R.from_euler('zxy', shift_euler).as_matrix()
        trans_mats[ip, :3, :3] = shift_rot_matrix2
        trans_mats[ip, :3, 3] = ret_t.reshape(-1)
    mat = torch.from_numpy(trans_mats).to(device=cfg.device, dtype=torch.float32)

    return mat

@hydra.main(version_base=None, config_path="config", config_name="config_sample_infbagel")
def test(cfg: DictConfig) -> None:
    device = cfg.device

    # seg_len = pkl.load(open('/cpfs04/shared/sport/zouyude/data_144/chois_lingo/test/language_motion_dict/seq_len.pkl', 'rb'))
    seg_id_dict = pkl.load(open('/cpfs04/shared/sport/zouyude/data_3/chois_lingo/test/language_motion_dict/seq_id.pkl', 'rb'))
    seg_id_dict = [0] + list(seg_id_dict.values())
    
    rest_verts_root = "/cpfs04/shared/sport/zouyude/code/chois_release/processed_data/rest_object_geo"
    obj_rest_verts = {}
    obj_vert_normals = {}
    for file in os.listdir(rest_verts_root):
        if not file.endswith('.ply'):
            continue
        obj_name = file.split('.')[0]
        rest_obj_path = os.path.join(rest_verts_root, file)
        mesh = trimesh.load_mesh(rest_obj_path)
        rest_verts = np.asarray(mesh.vertices) # Nv X 3
        obj_rest_verts[obj_name] = torch.from_numpy(zup_to_yup(rest_verts)).float().to(device)
        vert_normals = np.asarray(mesh.vertex_normals) # Nv X 3
        obj_vert_normals[obj_name] = torch.from_numpy(zup_to_yup(vert_normals)).float().to(device)

    object_sdf_root = '/cpfs04/shared/sport/zouyude/code/chois_release/processed_data/rest_object_sdf_256_npy_files'
    obj_sdf = {}
    obj_sdf_json = {}
    for file in os.listdir(object_sdf_root):
        if not file.endswith('.npy'):
            continue
        obj_name = file.split('.')[0]
        sdf_path = os.path.join(object_sdf_root, file)
        obj_sdf[obj_name] = np.load(sdf_path)
        obj_sdf_json[obj_name] = json.load(open(os.path.join(object_sdf_root, f'{file[:-4]}.json'), 'r'))
        
    sample_len = len(seg_id_dict)-1
    sample_len = 438 # 181 # for debug
    # sample_len = 50

    # only evaluate the single ckpt specified by cfg.ckpt_path (no longer iterate over the checkpoint directory)
    for model_name in [cfg.ckpt_path.split('/')[-1]]:
        metrics_filename = f"metrics_{model_name.split('.')[0].split('_')[-1]}.pkl"

        model_body = init_model(list(cfg.model.values())[0], device=device, eval=True)
        
        print(OmegaConf.to_yaml(cfg))
        
        # initialize the dataset
        synhsi_dataset = InfBaGelDataset(**cfg.dataset)

        sampler_body = hydra.utils.instantiate(cfg.sampler.pelvis)
        sampler_body.set_dataset_and_model(synhsi_dataset, model_body)

        # prepare the test metric results
        all_metrics = []

        # store results, compute metrics at the end
        points_all = torch.zeros(sample_len, 0, cfg.max_window_size, 3*cfg.dataset.nb_joints).to(device)
        global_rot_6d_all = torch.zeros(sample_len, 0, cfg.max_window_size, 22*6).to(device)
        object_trans_all = torch.zeros(sample_len, 0, cfg.max_window_size, 3).to(device)
        object_rot_mat_all = torch.zeros(sample_len, 0, cfg.max_window_size, 9).to(device)
        
        rest_human_offsets_all = torch.zeros(0, cfg.max_window_size*cfg.interp_s, 24, 3).to(device)
        transl_all = torch.zeros(0, 3).to(device)
        betas_all = torch.zeros(0, 16).to(device)
        gender_all = []
        
        transl_batch = []
        betas_batch = []

        points_all_gt = torch.zeros(0, cfg.max_window_size, 3*cfg.dataset.nb_joints).to(device)
        points_step_all_gt = torch.zeros(0, cfg.max_window_size, 3*cfg.dataset.nb_joints).to(device)

        object_trans_all_gt = torch.zeros(0, cfg.max_window_size, 3).to(device)
        object_rot_mat_all_gt = torch.zeros(0, cfg.max_window_size, 9).to(device)

        points_all_gt_48 = torch.zeros(0, cfg.max_window_size*cfg.interp_s-6, 3*cfg.dataset.nb_joints).to(device)
        pose_all_gt = torch.zeros(0, cfg.max_window_size*cfg.interp_s-6, 22, 3).to(device)

        object_trans_all_gt_48 = torch.zeros(0, cfg.max_window_size*cfg.interp_s-6, 3).to(device)
        object_rot_mat_all_gt_48 = torch.zeros(0, cfg.max_window_size*cfg.interp_s-6, 9).to(device)

        points_fk_all_gt_48 = torch.zeros(0, cfg.max_window_size*cfg.interp_s-6, 3*(cfg.dataset.nb_joints-4)).to(device)

        start_point_all_gt = torch.zeros(0, 1, 3*cfg.dataset.nb_joints).to(device)
        start_object_trans_all_gt = torch.zeros(0, 1, 3).to(device)
        end_object_trans_all_gt = torch.zeros(0, 1, 3).to(device)
        xy_points_all_gt = torch.zeros(0, 1, 3*cfg.dataset.nb_joints).to(device)

        mat_batch = []
        fixed_points_batch = []
        scene_flag_batch = []
        text_clip_embedding_batch = []
        pelvis_goal_batch = []
        hand_goal_batch = []
        object_goal_batch = []
        is_pick_batch = []
        need_scene_batch = []
        need_pelvis_dir_batch = []
        pi_batch = []
        end_pi_batch = []
        seq_length_batch = []
        need_pi_batch = []
        is_loco_batch = []
        is_object_batch = []
        obj_bps_data_first_step_batch = []
        obj_rot_mat_ref_first_step_batch = []
        object_points_batch = []
        first_object_points_batch = []
        first_object_trans_batch = []

        # hand_pen_loss_list = []
        # hand_pen_ratio_list = []
        # human_pen_loss_list = []
        # human_pen_ratio_list = []

        # max_len = max(seg_len.values())
        max_len = 3

        seq_name_dict = {}
        seg_id_true = 0
        for seg_id in range(0, len(seg_id_dict)-1):
        # for seg_id in range(250, 300): # debug
            data_dict = synhsi_dataset.__getitem__(seg_id_dict[seg_id])

            seq_name_dict[seg_id_true] = data_dict['seq_name']
            obj_name = seq_name_dict[seg_id_true].split('_')[1]
            seg_id_true += 1
            # if obj_name == 'woodchair' or obj_name == 'whitechair' or obj_name == 'largebox' or obj_name == 'largetable' \
            #     or obj_name == 'plasticbox' or obj_name == 'trashcan':
            #     continue
            
            joints, mat, object_trans, object_rot_mat, scene_flag, text_clip_embedding, pelvis_goal, hand_goal, object_goal, \
            is_pick, need_scene, need_pelvis_dir, pi, need_pi, is_loco, is_object, obj_bps_data, obj_rot_mat_ref, object_points = data_dict['joints'], data_dict['mat'], data_dict['object_trans'], data_dict['object_rot_mat'], data_dict['scene_flag'], data_dict['text_clip_embedding'], data_dict['pelvis_goal'], data_dict['hand_goal'], data_dict['object_goal'], data_dict['is_pick'], data_dict['need_scene'], data_dict['need_pelvis_dir'], data_dict['pi'], data_dict['need_pi'], data_dict['is_loco'], data_dict['is_object'], data_dict['obj_bps_data'], data_dict['obj_rot_mat_ref'], data_dict['object_points']
            
            end_pi = data_dict['end_pi']
            seq_length = data_dict['seg_len']

            contact_label = torch.from_numpy(data_dict['contact_label']).to(device)

            global_rot_6d = data_dict['global_rot_6d'].reshape(1, -1, 22*6).to(device)
            rest_human_offsets = torch.from_numpy(data_dict['rest_human_offsets']).to(device)
            transl = torch.from_numpy(data_dict['transl'])[None].to(device)
            betas = torch.from_numpy(data_dict['betas'])[None].to(device)
            transl_all = torch.cat([transl_all, transl], dim=0)
            
            rest_human_offsets_all = torch.cat([rest_human_offsets_all, rest_human_offsets.unsqueeze(0).repeat(48, 1, 1)[None]], dim=0)
            betas_all = torch.cat([betas_all, betas], dim=0)
            # gender_all.append(data_dict['gender'])
            gender_all.append('male')
            
            transl_batch.append(transl.repeat(1, 16, 1))
            betas_batch.append(betas.repeat(1, 16, 1))

            joints = torch.from_numpy(joints).to(device).reshape(1, -1, cfg.dataset.nb_joints*3)
            mat = torch.from_numpy(mat).to(device).reshape(1, 4, 4)
            object_trans = torch.from_numpy(object_trans).to(device).reshape(1, -1, 3)
            object_rot_mat = torch.from_numpy(object_rot_mat).to(device).reshape(1, -1, 9)
            text_clip_embedding = text_clip_embedding.to(device).unsqueeze(0)
            pelvis_goal = torch.from_numpy(pelvis_goal).to(device).unsqueeze(0)
            hand_goal = torch.from_numpy(hand_goal).to(device).unsqueeze(0)
            object_goal = torch.from_numpy(object_goal).to(device).unsqueeze(0)
            obj_bps_data = obj_bps_data.to(device).unsqueeze(0)
            obj_rot_mat_ref = torch.from_numpy(obj_rot_mat_ref).to(device).reshape(1, 3, 3)
            object_points = torch.from_numpy(object_points).reshape(1, -1, 3).to(device) # 1 X 1024 X 3

            # test
            # obj_name = data_dict['seq_name'].split('_')[1]
            # object_rot_mat_gt = torch.from_numpy(data_dict['object_rot_mat_gt']).to(device).reshape(1, -1, 9) # 1 X 48 X 9
            # object_trans_gt = torch.from_numpy(data_dict['object_trans_gt']).to(device).reshape(1, -1, 3) # 1 X 48 X 3
            # object_verts = load_object_geometry_w_rest_geo(object_rot_mat_gt.reshape(-1, 3, 3).float(), object_trans_gt.reshape(-1, 3).float(), obj_rest_verts[obj_name])
            # object_voxel_loss = synhsi_dataset.get_pene_occ_count(object_verts, [scene_flag])
            # print(data_dict['seq_name'], data_dict['scene_flag'], object_voxel_loss)
            #########################

            # convert to global coordinates
            pelvis_goal = transform_points(pelvis_goal.unsqueeze(1), mat).reshape(cfg.batch_size, 3)
            hand_goal = transform_points(hand_goal.unsqueeze(1), mat).reshape(cfg.batch_size, 3)
            object_goal = transform_points(object_goal.unsqueeze(1), mat).reshape(cfg.batch_size, 3)
            # print("global pelvis_goal: ", pelvis_goal, "hand_goal: ", hand_goal, "object_goal: ", object_goal)

            is_pick, need_scene, need_pelvis_dir, pi, need_pi, is_loco, is_object = \
                torch.from_numpy(np.array([is_pick])).to(device), torch.from_numpy(np.array([need_scene])).to(device), \
                torch.from_numpy(np.array([need_pelvis_dir])).to(device), torch.from_numpy(np.array([pi])).to(device), \
                torch.from_numpy(np.array([need_pi])).to(device), torch.from_numpy(np.array([is_loco])).to(device), \
                torch.from_numpy(np.array([is_object])).to(device)
            end_pi, seq_length = torch.from_numpy(np.array([end_pi])).to(device), torch.from_numpy(np.array([seq_length])).to(device)

            # convert to global coordinates
            points_orig = transform_points(synhsi_dataset.denormalize_torch(joints), mat)
            object_trans_orig = transform_points(synhsi_dataset.denormalize_torch(object_trans, is_object=True),mat)

            points_step_all_gt = torch.cat([points_step_all_gt, points_orig], dim=0)
            
            first_object_points = object_points
            object_points = object_points.squeeze(0) # 1024 X 3
            first_object_trans = object_trans_orig[:, 0, :].reshape(1, 3)
            
            fixed_points = points_orig[:, :cfg.auto_regre_num, :].reshape(cfg.batch_size, cfg.auto_regre_num, cfg.dataset.nb_joints*3)
            fixed_points = sampler_body.dataset.normalize_torch(transform_points(fixed_points, torch.inverse(mat)))

            obj_fixed = object_trans_orig[:, :cfg.auto_regre_num, :].reshape(cfg.batch_size, cfg.auto_regre_num, -1)
            obj_fixed = sampler_body.dataset.normalize_torch(transform_points(obj_fixed, torch.inverse(mat)), is_object=True)
            obj_rot_fixed = object_rot_mat[:, :cfg.auto_regre_num, :].reshape(cfg.batch_size, cfg.auto_regre_num, -1)

            fixed_contact_label = contact_label[:cfg.auto_regre_num, :].reshape(cfg.batch_size, cfg.auto_regre_num, -1)

            global_rot_6d_fixed = global_rot_6d[:, :cfg.auto_regre_num, :].reshape(cfg.batch_size, cfg.auto_regre_num, -1)
            # merge human and object data
            fixed_points = torch.cat([fixed_points, global_rot_6d_fixed, obj_fixed, obj_rot_fixed, fixed_contact_label], dim=-1) # 84 + 3 + 9 + 4

            mat_batch.append(mat)
            fixed_points_batch.append(fixed_points)
            scene_flag_batch.append(torch.tensor(scene_flag))
            text_clip_embedding_batch.append(text_clip_embedding)
            # pelvis_goal_batch.append(pelvis_goal)
            hand_goal_batch.append(hand_goal)
            object_goal_batch.append(object_goal)
            is_pick_batch.append(is_pick)
            need_scene_batch.append(need_scene)
            need_pelvis_dir_batch.append(need_pelvis_dir)
            # pi_batch.append(pi)
            need_pi_batch.append(need_pi)
            is_loco_batch.append(is_loco)
            is_object_batch.append(is_object)
            obj_bps_data_first_step_batch.append(obj_bps_data)
            obj_rot_mat_ref_first_step_batch.append(obj_rot_mat_ref)
            object_points_batch.append(object_points)
            first_object_points_batch.append(first_object_points)
            first_object_trans_batch.append(first_object_trans)

            pelvis_goal_batch_temp = []
            pi_batch_temp = []
            end_pi_batch_temp = []
            seq_length_batch_temp = []
            
            for step in range(max_len):
                data_dict = synhsi_dataset.__getitem__(seg_id_dict[seg_id] + step)
                joints, mat, object_trans, object_rot_mat, pelvis_goal, pi, obj_rot_mat_ref = data_dict['joints'], data_dict['mat'], data_dict['object_trans'], data_dict['object_rot_mat'], data_dict['pelvis_goal'], data_dict['pi'], data_dict['obj_rot_mat_ref']
                
                end_pi, seq_length = data_dict['end_pi'], data_dict['seg_len']

                joints = torch.from_numpy(joints).to(device).reshape(1, -1, cfg.dataset.nb_joints*3)
                mat = torch.from_numpy(mat).to(device).reshape(1, 4, 4)
                object_trans = torch.from_numpy(object_trans).to(device).reshape(1, -1, 3)
                object_rot_mat = torch.from_numpy(object_rot_mat).to(device).reshape(1, -1, 9)
                obj_rot_mat_ref = torch.from_numpy(obj_rot_mat_ref).to(device).reshape(1, 3, 3)
                pelvis_goal = torch.from_numpy(pelvis_goal).to(device).unsqueeze(0)
                pi = torch.from_numpy(np.array([pi])).to(device)

                end_pi, seq_length = torch.from_numpy(np.array([end_pi])).to(device), torch.from_numpy(np.array([seq_length])).to(device)

                pelvis_goal = transform_points(pelvis_goal.unsqueeze(1), mat).reshape(cfg.batch_size, 3)
                pelvis_goal_batch_temp.append(pelvis_goal)
                
                pi_batch_temp.append(pi)
                end_pi_batch_temp.append(end_pi)
                seq_length_batch_temp.append(seq_length)

                # convert to global coordinates
                points_orig = transform_points(synhsi_dataset.denormalize_torch(joints), mat)
                object_trans_orig = transform_points(synhsi_dataset.denormalize_torch(object_trans, is_object=True),mat)

                joints_gt = torch.from_numpy(data_dict['joints_gt']).to(device).reshape(1, -1, cfg.dataset.nb_joints*3) # 1 X 48 X 3*28
                object_rot_mat_gt = torch.from_numpy(data_dict['object_rot_mat_gt']).to(device).reshape(1, -1, 9) # 1 X 48 X 9
                object_trans_gt = torch.from_numpy(data_dict['object_trans_gt']).to(device).reshape(1, -1, 3) # 1 X 48 X 3

                # compare in global coordinates
                points_all_gt = torch.cat([points_all_gt, points_orig], dim=0)
                object_trans_all_gt = torch.cat([object_trans_all_gt, object_trans_orig], dim=0)
                object_rot_mat_global = (object_rot_mat.reshape(1, cfg.max_window_size, 3, 3) @ obj_rot_mat_ref).reshape(1, cfg.max_window_size, 9)
                object_rot_mat_all_gt = torch.cat([object_rot_mat_all_gt, object_rot_mat_global], dim=0)

                points_all_gt_48 = torch.cat([points_all_gt_48, joints_gt[:, 6:, :]], dim=0)
                object_trans_all_gt_48 = torch.cat([object_trans_all_gt_48, object_trans_gt[:, 6:, :]], dim=0)
                object_rot_mat_all_gt_48 = torch.cat([object_rot_mat_all_gt_48, object_rot_mat_gt[:, 6:, :]], dim=0)

                global_rot_6d_gt = data_dict['global_rot_6d_gt'].to(device) # [48, 22, 6]
                rest_human_offsets = torch.from_numpy(data_dict['rest_human_offsets']).to(global_rot_6d_gt.device) # [24, 3]

                # FK to get joint positions.
                curr_seq_local_jpos = rest_human_offsets.unsqueeze(0).repeat(global_rot_6d_gt.shape[0], 1, 1) # [48, 24, 3]
                curr_seq_local_jpos = curr_seq_local_jpos.reshape(-1, 24, 3) # [48, 24, 3]
                curr_seq_local_jpos[:, 0, :] = joints_gt.reshape(-1, 28, 3)[:, 0, :]
                
                global_jrot_mat_gt = mat[None, :, :3, :3] @ transforms.rotation_6d_to_matrix(global_rot_6d_gt) # [48, 22, 3, 3]
                local_jrot_mat_gt = synhsi_dataset.quat_ik_torch(global_jrot_mat_gt.reshape(-1, 22, 3, 3)) # [b*t, 22, 3, 3]

                pose_all_gt = torch.cat([pose_all_gt, transforms.matrix_to_axis_angle(local_jrot_mat_gt[6:]).reshape(1, -1, 22, 3)], dim=0)

                _, human_jnts_gt = synhsi_dataset.quat_fk_torch(local_jrot_mat_gt, curr_seq_local_jpos) # [48, 24, 3]
                points_fk_all_gt_48 = torch.cat([points_fk_all_gt_48, human_jnts_gt.reshape(1, 48, -1)[:, 6:, :]], dim=0)

            # # test
            # obj_trans = object_trans_all_gt_48[-seg_len[seg_id]:].reshape(-1, 3)
            # obj_rot_mat = object_rot_mat_all_gt_48[-seg_len[seg_id]:].reshape(-1, 3, 3)
            # points = points_all_gt_48[-seg_len[seg_id]:].reshape(-1, 28, 3)
            # root_trans = yup_to_zup(points[:, 0, :] + torch.from_numpy(data_dict['transl'])[None].to(device))
            # pose_pred = yup_to_zup(pose_all_gt[-seg_len[seg_id]:].reshape(-1, 22, 3))
            # verts, joints = run_smplx_model(pose_pred, root_trans, torch.from_numpy(data_dict['betas'])[None].repeat(root_trans.shape[0], 1).to(device), data_dict['gender'], joints_ind=None)
            # verts, joints = zup_to_yup(verts), zup_to_yup(joints)

            # with open('/cpfs04/shared/sport/zouyude/code/lingo-release/smpl_models/MANO_SMPLX_vertex_ids.pkl', 'rb') as f:
            #     idxs_data = pkl.load(f)
            # hand_idxs = np.concatenate([idxs_data['left_hand'], idxs_data['right_hand']]) # 1556

            # hand_verts = verts[:, hand_idxs, :]

            # hand_pen_loss, hand_pen_ratio = compute_collision(yup_to_zup(hand_verts), obj_sdf[obj_name], obj_sdf_json[obj_name], yup_to_zup_rotation_matrix(obj_rot_mat), yup_to_zup(obj_trans))
            # hand_pen_loss_list.append(hand_pen_loss)
            # hand_pen_ratio_list.append(hand_pen_ratio)

            # human_pen_loss, human_pen_ratio = compute_collision(yup_to_zup(verts), obj_sdf[obj_name], obj_sdf_json[obj_name], yup_to_zup_rotation_matrix(obj_rot_mat), yup_to_zup(obj_trans))
            # human_pen_loss_list.append(human_pen_loss)
            # human_pen_ratio_list.append(human_pen_ratio)
            # print(human_pen_loss, human_pen_ratio, hand_pen_loss, hand_pen_ratio)
            # ####################
            xy_points_all_gt = torch.cat([xy_points_all_gt, points_all_gt[-max_len:][:, -2, :].unsqueeze(1)], dim=0)
            start_point_all_gt = torch.cat([start_point_all_gt, points_all_gt[-max_len:][0:1, 0, :].unsqueeze(1)], dim=0)
            start_object_trans_all_gt = torch.cat([start_object_trans_all_gt, object_trans_all_gt[-max_len:][0:1, 0, :].unsqueeze(1)], dim=0)
            end_object_trans_all_gt = torch.cat([end_object_trans_all_gt, object_trans_all_gt[-max_len:][-1:, -1, :].unsqueeze(1)], dim=0)

            if not os.path.exists(os.path.join('t2m_results_48', 'gt', f"{data_dict['seq_name']}.npz")):
                if not os.path.exists(os.path.join('t2m_results_48', 'gt')):
                    os.makedirs(os.path.join('t2m_results_48', 'gt'))

                np.savez(os.path.join('t2m_results_48', 'gt', f"{data_dict['seq_name']}.npz"), seq_name=data_dict['seq_name'], \
                            global_jpos=yup_to_zup(points_fk_all_gt_48.reshape(-1, 24, 3)).cpu().numpy()) # T X 24 X 3

            # if seg_len[seg_id] < max_len:
            #     pelvis_goal_batch_temp.extend([pelvis_goal_batch_temp[-1]] * (max_len - seg_len[seg_id]))
            #     pi_batch_temp.extend([pi_batch_temp[-1]] * (max_len - seg_len[seg_id]))
            #     end_pi_batch_temp.extend([end_pi_batch_temp[-1]] * (max_len - seg_len[seg_id]))
            #     seq_length_batch_temp.extend([seq_length_batch_temp[-1]] * (max_len - seg_len[seg_id]))
            
            pelvis_goal_batch.append(torch.stack(pelvis_goal_batch_temp))
            
            pi_batch.append(torch.stack(pi_batch_temp))
            end_pi_batch.append(torch.stack(end_pi_batch_temp))
            seq_length_batch.append(torch.stack(seq_length_batch_temp))

            # pi_batch.append(torch.tensor([0, 42, 84]).reshape(3, 1).to(device))
            # end_pi_batch.append(torch.tensor([48, 90, 132]).reshape(3, 1).to(device))
            # seq_length_batch.append(torch.tensor([132, 132, 132]).reshape(3, 1).to(device))

        # human_pen_loss_list = np.array(human_pen_loss_list)
        # human_pen_ratio_list = np.array(human_pen_ratio_list)
        # hand_pen_loss_list = np.array(hand_pen_loss_list)
        # hand_pen_ratio_list = np.array(hand_pen_ratio_list)
        # import pdb; pdb.set_trace()
        # np.savez(os.path.join('t2m_results_48', f"t2m_mean_std_jpos.npz"), \
        #         jpos_mean=curr_pred_global_jpos_gt_all.mean(axis=0).reshape(72), jpos_std=curr_pred_global_jpos_gt_all.std(axis=0).reshape(72))

        transl_batch = torch.stack(transl_batch)
        betas_batch = torch.stack(betas_batch)

        mat_batch = torch.stack(mat_batch).reshape(-1, 4, 4) # 534 X 4 X 4
        fixed_points_batch = torch.stack(fixed_points_batch).reshape(-1, 2, 232) # 534 X 2 X 232
        scene_flag_batch = torch.stack(scene_flag_batch) # 534 X 1
        text_clip_embedding_batch = torch.stack(text_clip_embedding_batch).reshape(-1, 1, 768) # 534 X 1 X 1 X 768
        pelvis_goal_batch = torch.stack(pelvis_goal_batch) # 534 X 7 X 1 X 3
        hand_goal_batch = torch.stack(hand_goal_batch) # 534 X 1 X 3
        object_goal_batch = torch.stack(object_goal_batch) # 534 X 1 X 3
        is_pick_batch = torch.stack(is_pick_batch).reshape(-1) # 534
        need_scene_batch = torch.stack(need_scene_batch).reshape(-1) # 534
        need_pelvis_dir_batch = torch.stack(need_pelvis_dir_batch).reshape(-1) # 534
        pi_batch = torch.stack(pi_batch).reshape(-1, max_len) # 534 X 7
        end_pi_batch = torch.stack(end_pi_batch).reshape(-1, max_len) # 534 X 7
        seq_length_batch = torch.stack(seq_length_batch).reshape(-1, max_len) # 534 X 7
        need_pi_batch = torch.stack(need_pi_batch).reshape(-1) # 534
        is_loco_batch = torch.stack(is_loco_batch).reshape(-1) # 534
        is_object_batch = torch.stack(is_object_batch).reshape(-1) # 534
        obj_bps_data_first_step_batch = torch.stack(obj_bps_data_first_step_batch) # 534 X 1 X 1 X 1024 X 3
        obj_rot_mat_ref_first_step_batch = torch.stack(obj_rot_mat_ref_first_step_batch) # 534 X 1 X 3 X 3
        object_points_batch = torch.stack(object_points_batch) # 534 X 3 X 1024
        first_object_points_batch = torch.stack(first_object_points_batch).reshape(sample_len, 1024, 3) # 534 X 1024 X 3
        first_object_trans_batch = torch.stack(first_object_trans_batch) # 534 X 1 X 3

        batch_size = object_points_batch.shape[0]
        
        for step in range(0, max_len):
            print(f"step: {step}")
            if step != 0:
                # import pdb; pdb.set_trace()
                rel_object_trans = obj_trans[:, -cfg.auto_regre_num, :].reshape(batch_size, 1, 3) - first_object_trans_batch # 534 X 1 X 3
                rel_object_rot_mat = obj_rot_mat[:,-cfg.auto_regre_num,:].reshape(batch_size, 3, 3)
                object_points = rel_object_rot_mat.bmm(first_object_points_batch.transpose(1, 2)) + rel_object_trans.transpose(1, 2)

                object_points_batch = object_points.transpose(1, 2) # 534 X 1024 X 3

                mat = get_mat(cfg, points)

                global_rot_6d = global_rot_6d.reshape(sample_len, cfg.max_window_size, 22, 6)

                init_global_rot_mat = transforms.rotation_6d_to_matrix(global_rot_6d[:, -cfg.auto_regre_num, 0, :]).reshape(batch_size, 3, 3)
                init_global_orient = transforms.matrix_to_axis_angle(init_global_rot_mat).cpu().numpy() # 534 X 3
                init_global_orient_euler = R.from_rotvec(init_global_orient).as_euler('zxy')
                shift_euler = np.zeros_like(init_global_orient_euler)
                shift_euler[:, 2] = -init_global_orient_euler[:, 2]
                shift_rot_matrix = R.from_euler('zxy', shift_euler).as_matrix() # 534 X 3 X 3

                global_jrot_mat = transforms.rotation_6d_to_matrix(global_rot_6d)
                global_jrot_mat = torch.from_numpy(shift_rot_matrix).float()[:, None, None].to(device) @ global_jrot_mat

                mat[:, :3, :3] = torch.from_numpy(np.linalg.inv(shift_rot_matrix)).float().to(device)
                init_joints = points.reshape(batch_size, cfg.max_window_size, -1, 3)[:, -cfg.auto_regre_num, 0, :].float().clone()
                # init_joints[:, 1] = 0. # B X 3
                mat[:, 0, 3] = init_joints[:, 0]
                mat[:, 2, 3] = init_joints[:, 2]
    
                fixed_points = points[:, -cfg.auto_regre_num:].reshape(batch_size, cfg.auto_regre_num, cfg.dataset.nb_joints*3).clone()
                fixed_points = sampler_body.dataset.normalize_torch(transform_points(fixed_points, torch.inverse(mat)))

                obj_fixed = obj_trans[:, -cfg.auto_regre_num:].reshape(batch_size, cfg.auto_regre_num, -1).clone()
                obj_fixed = sampler_body.dataset.normalize_torch(transform_points(obj_fixed, torch.inverse(mat)), is_object=True)
                obj_rot_fixed = obj_rot_mat[:, -cfg.auto_regre_num:].reshape(batch_size, cfg.auto_regre_num, -1).clone()

                fixed_contact_label = contact_label[:, -cfg.auto_regre_num:].reshape(batch_size, cfg.auto_regre_num, -1).clone()
                
                global_rot_6d = transforms.matrix_to_rotation_6d(global_jrot_mat).reshape(sample_len, cfg.max_window_size, 22*6)

                global_rot_6d_fixed = global_rot_6d[:, -cfg.auto_regre_num:].reshape(batch_size, cfg.auto_regre_num, -1)

                fixed_points_batch = torch.cat([fixed_points, global_rot_6d_fixed, obj_fixed, obj_rot_fixed, fixed_contact_label], dim=-1) # 84 + 3 + 9 + 4
                mat_batch = mat

            human_dict = {'rest_human_offsets': rest_human_offsets_all[:, :cfg.max_window_size], 'transl': transl_batch, 'betas': betas_batch, 'gender': gender_all}
            
            info_dict = sample_step(cfg, mat_batch, fixed_points_batch, sampler_body, scene_flag_batch, text_clip_embedding_batch, pelvis_goal_batch[:,step], hand_goal_batch, object_goal_batch, is_pick_batch, need_scene_batch, need_pelvis_dir_batch, pi_batch[:,step], end_pi_batch[:,step], seq_length_batch[:,step], need_pi_batch, is_loco_batch, is_object_batch, obj_bps_data_first_step_batch, object_points_batch, obj_rest_verts, obj_vert_normals, seq_name_dict, obj_rot_mat_ref_first_step_batch, human_dict)
            
            points = info_dict['points_orig'].clone() # 534 X T X 3*28
            obj_trans = info_dict['obj_trans_orig'].clone() # 534 X T X 3
            obj_rot_mat = info_dict['object_rot_mat'].clone() # 534 X T X 9
            contact_label = info_dict['contact_label'].clone() # 534 X T X 4
            global_rot_6d = info_dict['global_rot_6d'].clone() # 534 X T X 22*6

            object_rot_mat_global = (obj_rot_mat.reshape(sample_len, cfg.max_window_size, 3, 3) @ obj_rot_mat_ref_first_step_batch).reshape(sample_len, cfg.max_window_size, 9)
            points_all = torch.cat([points_all, points.unsqueeze(1)], dim=1)
            object_trans_all = torch.cat([object_trans_all, obj_trans.unsqueeze(1)], dim=1)
            object_rot_mat_all = torch.cat([object_rot_mat_all, object_rot_mat_global.unsqueeze(1)], dim=1)
            
            global_jrot_mat = transforms.rotation_6d_to_matrix(global_rot_6d.reshape(sample_len, cfg.max_window_size, 22, 6))
            global_jrot_mat = mat_batch[:, None, None, :3, :3] @ global_jrot_mat
            global_rot_6d = transforms.matrix_to_rotation_6d(global_jrot_mat).reshape(sample_len, cfg.max_window_size, 22*6)

            global_rot_6d_all = torch.cat([global_rot_6d_all, global_rot_6d.unsqueeze(1)], dim=1) # B * S * T * (22*6)

        metrics = compute_metrics(sampler_body, cfg, points_all, global_rot_6d_all, points_fk_all_gt_48, object_trans_all, object_trans_all_gt_48, object_rot_mat_all, object_rot_mat_all_gt_48, start_point_all_gt, start_object_trans_all_gt, end_object_trans_all_gt, xy_points_all_gt, seq_name_dict, obj_rest_verts, rest_human_offsets_all, transl_all, betas_all, gender_all)
        
        all_metrics.append(metrics)
        
        # aggregate and save evaluation metrics
        metrics_summary = summarize_metrics(all_metrics)
        
        if not os.path.exists(os.path.join('results', cfg.exp_name)):
            os.makedirs(os.path.join('results', cfg.exp_name))
        with open(os.path.join('results', cfg.exp_name, metrics_filename), 'wb') as f:
            pkl.dump({'all_metrics': all_metrics, 'summary': metrics_summary}, f)

        # print the evaluation metrics summary
        print(f"\n{metrics_filename} Evaluation Metrics Summary:")
        for key, value in metrics_summary.items():
            if not key.endswith('_std'):
                std_value = metrics_summary.get(f"{key}_std", 0)
                print(f"  {key}: {value:.4f} ± {std_value:.4f}")

        print(f"\nTest completed.")

if __name__ == '__main__':
    os.environ['HYDRA_FULL_ERROR'] = '1'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['ROOT_DIR'] = '../'

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)

    OmegaConf.register_new_resolver("times", lambda x, y: int(x) * int(y))
    test()