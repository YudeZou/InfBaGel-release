import os
import pickle as pkl
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from scipy.spatial.transform import Rotation as R
from tqdm.auto import tqdm
import trimesh

from utils import *
from constants import *
from clip_utils import get_clip_features

from astar import get_path
import matplotlib.pyplot as plt
from guidance_loss import *
from sklearn.cluster import DBSCAN
import json

def compute_signed_distances(sdf, sdf_centroid, sdf_extents, query_points):
    # sdf: 1 X 256 X 256 X 256 
    # sdf_centroid: 1 X 3, center of the bounding box.  
    # sdf_extents: 1 X 3, width, height, depth of the box.  
    # query_points: T X Nv X 3 

    # query_pts_norm = (query_points - sdf_centroid[None, :, :]) * 2 / sdf_extents[None, :, :] # Convert to range [-1, 1]
    query_pts_norm = (query_points - sdf_centroid[None, :, :]) * 2 / sdf_extents.max() # Convert to range [-1, 1]
     
    query_pts_norm = query_pts_norm[...,[2, 1, 0]] # Switch the order to depth, height, width
    
    num_steps, nv, _ = query_pts_norm.shape # T X Nv X 3 

    query_pts_norm = query_pts_norm[None, :, None, :, :] # 1 X T X 1 X Nv X 3 

    signed_dists = F.grid_sample(sdf[:, None, :, :, :], query_pts_norm.float(), padding_mode='border', align_corners=True)
    # F.grid_sample: N X C X D_in X H_in X W_in, N X D_out X H_out X W_out X 3, output: N X C X D_out X H_out X W_out 
    # sdf: 1 X 1 X 256 X 256 X 256, query_pts: 1 X T X 1 X Nv X 3 -> 1 X 1 X T X 1 X Nv  

    signed_dists = signed_dists[0, 0, :, 0, :] * sdf_extents.max() / 2. # T X Nv 
    
    return signed_dists
    
def compute_collision(ori_verts_pred, sdf_data, sdf_json_data, obj_rot_mat, obj_trans): 
    # ori_verts_pred: T X Nv X 3 
    # sdf_data: 256 X 256 X 256 
    # obj_rot_mat: T X 3 X 3 
    # obj_trans: T X 3 

    # Convert human vertices to align with the initial object geometry. 
    tmp_verts = (ori_verts_pred - obj_trans[:, None, :]) # * (1/obj_scale[:, None, None]) # T X Nv X 3 
    transformed_human_verts = torch.matmul(obj_rot_mat.transpose(1, 2), tmp_verts.transpose(1, 2)) # T X 3 X Nv     
    transformed_human_verts = transformed_human_verts.transpose(1, 2) # T X Nv X 3 
    # transformed_human_verts = torch.matmul(obj_rot_mat[:, None, :, :].repeat(1, tmp_verts.shape[1], 1, 1), \
    #                                     tmp_verts[..., None]).squeeze(-1) # T X Nv X 3     
    # transformed_human_verts = transformed_human_verts[:, :, 0, :] # T X Nv X 3 

    sdf_centroid = torch.from_numpy(np.asarray(sdf_json_data['centroid']))[None, :].to(transformed_human_verts.device) # 1 X 3 
    sdf_extents = torch.from_numpy(np.asarray(sdf_json_data['extents']))[None, :].to(transformed_human_verts.device) # 1 X 3 

    sdf = torch.from_numpy(sdf_data).float()[None, :].to(transformed_human_verts.device) # 1 X 256 X 256 X 256 

    signed_dists = compute_signed_distances(sdf, sdf_centroid, sdf_extents, transformed_human_verts)
    
    # Compute collision percentage
    collision_threshold = -0.04  # collision threshold of 4 cm
    collision_mask = (signed_dists < collision_threshold)  # find all collision points
    collision_frames = torch.any(collision_mask, dim=1)  # whether each frame has a collision
    pene_ratio = torch.sum(collision_frames).float() / collision_frames.shape[0]  # compute collision frame ratio

    penetration_score = torch.minimum(signed_dists, torch.zeros_like(signed_dists)).abs().mean() # The smaller, the better 
    # pen_loss = 0
    # pen_cnt = 0
    # for t_idx in range(signed_dists.shape[0]):
    #     neg_dists_mask = signed_dists[t_idx] < 0
    #     neg_dists = torch.abs(signed_dists[t_idx][neg_dists_mask])
    #     if len(neg_dists) != 0:
    #         pen_loss += neg_dists.mean()
    #         pen_cnt += 1

    # if pen_cnt > 0:
    #     pen_loss = pen_loss/pen_cnt 
    # else:
    #     pen_loss = torch.tensor(0.)

    return penetration_score.detach().cpu().numpy() * 100, pene_ratio.detach().cpu().numpy()
    # return pen_loss.detach().cpu().numpy() * 100, pene_ratio.detach().cpu().numpy()

# Evaluation metric computation
def get_frobenious_norm_rot_only(x, y):
    # x, y: N X 3 X 3 
    error = 0.0
    ident_mat = np.identity(3)
    for i in range(len(x)):
        x_mat = x[i][:3, :3]
        y_mat_inv = np.linalg.inv(y[i][:3, :3])
        error_mat = np.matmul(x_mat, y_mat_inv)
        error += np.linalg.norm(ident_mat - error_mat, 'fro')
    return error / len(x)


def determine_floor_height_and_contacts(body_joint_seq, fps=30):
    # body_joint_seq: T X J X 3
    FLOOR_VEL_THRESH = 0.005
    FLOOR_HEIGHT_OFFSET = 0.01
    
    if body_joint_seq.shape[1] == 24:
        left_foot_idx = 7
        right_foot_idx = 8
        left_toe_idx = 10
        right_toe_idx = 11
    elif body_joint_seq.shape[1] == 28:
        # Adapt to different skeleton indices
        left_foot_idx = 10
        right_foot_idx = 11
        left_toe_idx = 7
        right_toe_idx = 8
    
    left_foot_idx = 10
    right_foot_idx = 11
    
    # Compute foot positions
    root_seq = body_joint_seq[:, 0, :]
    left_foot_seq = body_joint_seq[:, left_foot_idx, :]
    right_foot_seq = body_joint_seq[:, right_foot_idx, :]
    left_foot_vel = np.linalg.norm(left_foot_seq[1:] - left_foot_seq[:-1], axis=1)
    left_foot_vel = np.append(left_foot_vel, left_foot_vel[-1])
    right_foot_vel = np.linalg.norm(right_foot_seq[1:] - right_foot_seq[:-1], axis=1)
    right_foot_vel = np.append(right_foot_vel, right_foot_vel[-1])
    # left_toe_seq = body_joint_seq[:, left_toe_idx, :]
    # right_toe_seq = body_joint_seq[:, right_toe_idx, :]
    
    left_foot_heights = left_foot_seq[:, 1]
    right_foot_heights = right_foot_seq[:, 1]
    root_heights = root_seq[:, 1]

    # filter out heights when velocity is greater than some threshold (not in contact)
    all_inds = np.arange(left_foot_heights.shape[0])
    left_static_foot_heights = left_foot_heights[left_foot_vel < FLOOR_VEL_THRESH]
    left_static_inds = all_inds[left_foot_vel < FLOOR_VEL_THRESH]
    right_static_foot_heights = right_foot_heights[right_foot_vel < FLOOR_VEL_THRESH]
    right_static_inds = all_inds[right_foot_vel < FLOOR_VEL_THRESH]

    all_static_foot_heights = np.append(left_static_foot_heights, right_static_foot_heights)
    all_static_foot_inds = np.append(left_static_inds, right_static_inds)

    if all_static_foot_heights.shape[0] > 0:
        cluster_heights = []
        cluster_root_heights = []
        cluster_sizes = []

        # cluster foot heights and find one with smallest median
        clustering = DBSCAN(eps=0.005, min_samples=3).fit(all_static_foot_heights.reshape(-1, 1))
        all_labels = np.unique(clustering.labels_)

        min_median = min_root_median = float('inf')
        for cur_label in all_labels:
            cur_clust = all_static_foot_heights[clustering.labels_ == cur_label]
            cur_clust_inds = np.unique(all_static_foot_inds[clustering.labels_ == cur_label])

            cur_median = np.median(cur_clust)
            cluster_heights.append(cur_median)
            cluster_sizes.append(cur_clust.shape[0])

            cur_root_clust = root_heights[cur_clust_inds]
            cur_root_median = np.median(cur_root_clust)
            cluster_root_heights.append(cur_root_median)

            if cur_median < min_median:
                min_median = cur_median
                min_root_median = cur_root_median
        
        floor_height = min_median
        offset_floor_height = floor_height - FLOOR_HEIGHT_OFFSET
    else:
        floor_height = offset_floor_height = 0.0

    return floor_height

def compute_foot_sliding_for_smpl(pred_global_jpos, floor_height):
    # pred_global_jpos: T X J X 3 
    seq_len = pred_global_jpos.shape[0]

    # Put human mesh to floor y = 0 and compute. 
    pred_global_jpos[:, :, 1] -= floor_height

    lankle_pos = pred_global_jpos[:, 7, :] # T X 3 
    ltoe_pos = pred_global_jpos[:, 10, :] # T X 3 

    rankle_pos = pred_global_jpos[:, 8, :] # T X 3 
    rtoe_pos = pred_global_jpos[:, 11, :] # T X 3 

    H_ankle = 0.08 # meter
    H_toe = 0.04 # meter 

    # y-up
    lankle_disp = np.linalg.norm(lankle_pos[1:, [0,2]] - lankle_pos[:-1, [0,2]], axis = 1) # T 
    ltoe_disp = np.linalg.norm(ltoe_pos[1:, [0,2]] - ltoe_pos[:-1, [0,2]], axis = 1) # T 
    rankle_disp = np.linalg.norm(rankle_pos[1:, [0,2]] - rankle_pos[:-1, [0,2]], axis = 1) # T 
    rtoe_disp = np.linalg.norm(rtoe_pos[1:, [0,2]] - rtoe_pos[:-1, [0,2]], axis = 1) # T 

    lankle_subset = lankle_pos[:-1, 1] < H_ankle
    ltoe_subset = ltoe_pos[:-1, 1] < H_toe
    rankle_subset = rankle_pos[:-1, 1] < H_ankle
    rtoe_subset = rtoe_pos[:-1, 1] < H_toe
   
    lankle_sliding_stats = np.abs(lankle_disp * (2 - 2 ** (lankle_pos[:-1, 1]/H_ankle)))[lankle_subset]
    lankle_sliding = np.sum(lankle_sliding_stats)/seq_len * 100

    ltoe_sliding_stats = np.abs(ltoe_disp * (2 - 2 ** (ltoe_pos[:-1, 1]/H_toe)))[ltoe_subset]
    ltoe_sliding = np.sum(ltoe_sliding_stats)/seq_len * 100

    rankle_sliding_stats = np.abs(rankle_disp * (2 - 2 ** (rankle_pos[:-1, 1]/H_ankle)))[rankle_subset]
    rankle_sliding = np.sum(rankle_sliding_stats)/seq_len * 100

    rtoe_sliding_stats = np.abs(rtoe_disp * (2 - 2 ** (rtoe_pos[:-1, 1]/H_toe)))[rtoe_subset]
    rtoe_sliding = np.sum(rtoe_sliding_stats)/seq_len * 100

    sliding = (lankle_sliding + ltoe_sliding + rankle_sliding + rtoe_sliding) / 4.

    return sliding 

def compute_hand_object_interaction(jpos_pred, jpos_gt, pred_obj_verts, gt_obj_verts):
    # Compute hand-object interaction
    # jpos_pred: T X J X 3
    # obj_verts: T X No X 3
    
    # Define hand joint indices
    if jpos_pred.shape[1] == 24:
        lhand_idx = 22
        rhand_idx = 23
    else:
        lhand_idx = 24
        rhand_idx = 26
    
    num_obj_verts = gt_obj_verts.shape[1]
    contact_threshold = 0.05  # contact threshold (m)

    gt_lhand_jnt = jpos_gt[:, lhand_idx, :] # T X 3
    gt_rhand_jnt = jpos_gt[:, rhand_idx, :] # T X 3

    gt_lhand2obj_dist = torch.sqrt(((gt_lhand_jnt[:, None, :].repeat(1, num_obj_verts, 1) - gt_obj_verts.to(gt_lhand_jnt.device))**2).sum(dim=-1)) # T X N  
    gt_rhand2obj_dist = torch.sqrt(((gt_rhand_jnt[:, None, :].repeat(1, num_obj_verts, 1) - gt_obj_verts.to(gt_rhand_jnt.device))**2).sum(dim=-1)) # T X N  
    
    gt_lhand2obj_dist_min = gt_lhand2obj_dist.min(dim=1)[0] # T 
    gt_rhand2obj_dist_min = gt_rhand2obj_dist.min(dim=1)[0] # T 

    gt_lhand_contact = (gt_lhand2obj_dist_min < contact_threshold)
    gt_rhand_contact = (gt_rhand2obj_dist_min < contact_threshold)

    lhand_jnt = jpos_pred[:, lhand_idx, :]
    rhand_jnt = jpos_pred[:, rhand_idx, :]

    lhand_jnt2obj_dist = torch.sqrt(((lhand_jnt[:, None, :].repeat(1, num_obj_verts, 1) - pred_obj_verts.to(lhand_jnt.device))**2).sum(dim=-1)) # T X N  
    rhand_jnt2obj_dist = torch.sqrt(((rhand_jnt[:, None, :].repeat(1, num_obj_verts, 1) - pred_obj_verts.to(rhand_jnt.device))**2).sum(dim=-1)) # T X N  
    
    lhand_jnt2obj_dist_min = lhand_jnt2obj_dist.min(dim=1)[0] # T 
    rhand_jnt2obj_dist_min = rhand_jnt2obj_dist.min(dim=1)[0] # T 

    lhand_contact = (lhand_jnt2obj_dist_min < contact_threshold)
    rhand_contact = (rhand_jnt2obj_dist_min < contact_threshold)

    num_steps = lhand_contact.shape[0]
    
    # Compute the distance between hand joint and object for frames that are in contact with object in GT. 
    contact_dist = 0
    gt_contact_dist = 0

    gt_contact_cnt = 0
    
    for idx in range(num_steps):
        if gt_lhand_contact[idx] or gt_rhand_contact[idx]:
            gt_contact_cnt += 1

            contact_dist += min(lhand_jnt2obj_dist_min[idx], rhand_jnt2obj_dist_min[idx])
            gt_contact_dist += min(gt_lhand2obj_dist_min[idx], gt_rhand2obj_dist_min[idx])
        
    
    if gt_contact_cnt == 0:
        contact_dist = 0
        gt_contact_dist = 0
    else:
        contact_dist = contact_dist.detach().cpu().numpy() / float(gt_contact_cnt)
        gt_contact_dist = gt_contact_dist.detach().cpu().numpy() / float(gt_contact_cnt)
    
    pred_contact_cnt = 0
    # Compute precision and recall for contact.
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for idx in range(num_steps):
        gt_in_contact = (gt_lhand_contact[idx] or gt_rhand_contact[idx])
        pred_in_contact = (lhand_contact[idx] or rhand_contact[idx])
        if gt_in_contact and pred_in_contact:
            TP += 1
        elif (not gt_in_contact) and pred_in_contact:
            FP += 1
        elif (not gt_in_contact) and (not pred_in_contact):
            TN += 1
        else:
            FN += 1

        if pred_in_contact:
            pred_contact_cnt += 1
    
    gt_contact_percent = gt_contact_cnt / float(num_steps)
    pred_contact_percent = pred_contact_cnt / float(num_steps)

    contact_acc = (TP + TN) / (TP + TN + FP + FN)

    if (TP + FP) == 0:
        contact_precision = 0
        # print('Contact precision, TP + FP == 0!!')
    else:
        contact_precision = TP / (TP + FP)
    
    if (TP + FN) == 0:
        contact_recall = 0
        # print('Contact recall, TP + FN == 0!!')
    else:
        contact_recall = TP / (TP + FN)

    if contact_precision == 0 and contact_recall == 0:
        contact_f1 = 0
    else:
        contact_f1 = 2 * contact_precision * contact_recall / (contact_precision + contact_recall)
    
    return gt_contact_percent, pred_contact_percent, contact_acc, contact_precision, contact_recall, contact_f1

def compute_condition_matching(start_point_all, start_object_trans_all, end_object_trans_all, xy_points_all, start_point_all_gt, start_object_trans_all_gt, end_object_trans_all_gt, xy_points_all_gt):
    # start_point_all: N X 1 X 84
    # start_object_trans_all: N X 1 X 3
    # end_object_trans_all: N X 1 X 3
    # xy_points_all: N X S X 84
    # start_point_all_gt: N X 1 X 84
    # start_object_trans_all_gt: N X 1 X 3
    # end_object_trans_all_gt: N X 1 X 3
    # xy_points_all_gt: N X S X 84

    # Convert to numpy arrays
    start_point_all = start_point_all.detach().cpu().numpy()
    start_object_trans_all = start_object_trans_all.detach().cpu().numpy()
    end_object_trans_all = end_object_trans_all.detach().cpu().numpy()
    xy_points_all = xy_points_all.detach().cpu().numpy()
    start_point_all_gt = start_point_all_gt.detach().cpu().numpy()
    start_object_trans_all_gt = start_object_trans_all_gt.detach().cpu().numpy()
    end_object_trans_all_gt = end_object_trans_all_gt.detach().cpu().numpy()
    xy_points_all_gt = xy_points_all_gt.detach().cpu().numpy()

    # Compute the start pose error
    start_point_all = start_point_all.reshape(-1, 28, 3)
    start_point_all_gt = start_point_all_gt.reshape(-1, 28, 3)

    start_point_err = np.mean(np.linalg.norm(start_point_all[:, 0, :] - start_point_all_gt[:, 0, :], axis=-1)) * 100

    # Compute the start object position error
    start_obj_trans_err = np.mean(np.linalg.norm(start_object_trans_all - start_object_trans_all_gt, axis=-1)) * 100

    # Compute the end object position error
    end_obj_trans_err = np.mean(np.linalg.norm(end_object_trans_all - end_object_trans_all_gt, axis=-1)) * 100

    # Compute the trajectory error
    xy_points_all = xy_points_all.reshape(-1, 28, 3)
    xy_points_all_gt = xy_points_all_gt.reshape(-1, 28, 3)
    xy_points_all[:, 0, 1] = 0.
    xy_points_all_gt[:, 0, 1] = 0.
    xy_points_err = np.mean(np.linalg.norm(xy_points_all[:, 0, :] - xy_points_all_gt[:, 0, :], axis=-1)) * 100

    return start_point_err, start_obj_trans_err, end_obj_trans_err, xy_points_err

def compute_gt_difference(points_orig, points_gt_orig, obj_trans_pred, obj_trans_gt, obj_rot_mat_pred, obj_rot_mat_gt):
    # points_orig: N X T X 84
    # points_gt_orig: N X T X 84
    # obj_trans_pred: N X T X 3
    # obj_trans_gt: N X T X 3
    # obj_rot_mat_pred: N X T X 9
    # obj_rot_mat_gt: N X T X 9
    points_orig = points_orig.detach().cpu().numpy().reshape(-1, 24, 3)
    points_gt_orig = points_gt_orig.detach().cpu().numpy().reshape(-1, 24, 3)
    obj_trans_pred = obj_trans_pred.detach().cpu().numpy().reshape(-1, 3)
    obj_trans_gt = obj_trans_gt.detach().cpu().numpy().reshape(-1, 3)
    obj_rot_mat_pred = obj_rot_mat_pred.detach().cpu().numpy().reshape(-1, 3, 3)
    obj_rot_mat_gt = obj_rot_mat_gt.detach().cpu().numpy().reshape(-1, 3, 3)

    jpos_pred = points_orig - points_orig[:, 0:1, :]
    jpos_gt = points_gt_orig - points_gt_orig[:, 0:1, :]
    mpjpe = np.linalg.norm(jpos_pred - jpos_gt, axis=2).mean() * 100
    trans_dist = np.linalg.norm(points_orig[:, 0, :] - points_gt_orig[:, 0, :], axis=1).mean() * 100

    obj_rot_dist = get_frobenious_norm_rot_only(obj_rot_mat_pred, obj_rot_mat_gt)
    obj_trans_dist = np.linalg.norm(obj_trans_pred - obj_trans_gt, axis=1).mean() * 100
    # end_obj_trans_err = np.linalg.norm(obj_trans_pred[-1:] - obj_trans_gt[-1:], axis=1).mean() * 100
    return mpjpe, trans_dist, obj_trans_dist, obj_rot_dist # , end_obj_trans_err