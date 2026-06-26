import torch
import torch.nn.functional as F

def apply_hand_object_interaction_guidance_loss(human_jnts, obj_verts, pred_seq_com_pos, pred_obj_rot_mat, contact_labels):
    # human_jnts: BS X T X 24 X 3
    # obj_verts: BS X T X Nv' X 3
    # pred_seq_com_pos: BS X T X 3
    # pred_obj_rot_mat: BS X T X 3 X 3
    # contact_labels: BS X T X 4

    num_seq = human_jnts.shape[0]
    num_steps = human_jnts.shape[1]

    # Contact loss: minimize the distance between palm joints and the nearest object vertices.
    l_palm_idx = 22
    r_palm_idx = 23

    left_palm_jpos = human_jnts[:, :, l_palm_idx, :] # BS X T X 3
    right_palm_jpos = human_jnts[:, :, r_palm_idx, :] # BS X T X 3

    contact_points = torch.cat((left_palm_jpos[:, :, None, :], \
                right_palm_jpos[:, :, None, :]), dim=2) # BS X T X 2 X 3
    bs, seq_len, _, _ = contact_points.shape

    dists = torch.cdist(contact_points.reshape(bs*seq_len, 2, 3)[:, :, :], \
                obj_verts.reshape(bs*seq_len, -1, 3)) # (BS*T) X 2 X N_object
    dists, _ = torch.min(dists, 2) # (BS*T) X 2

    pred_contact_semantic = contact_labels[:, :, -4:-2] # BS X T X 2

    contact_labels = pred_contact_semantic > 0.95

    contact_labels = contact_labels.reshape(bs*seq_len, -1)[:, :2].detach().to(dists.device) # (BS*T) X 2

    zero_target = torch.zeros_like(dists).to(dists.device)
    contact_threshold = 0.02

    loss_contact = F.l1_loss(torch.maximum(dists*contact_labels[:, :2]-contact_threshold, zero_target), \
            zero_target)

    # Temporal consistency loss.
    left_palm_to_obj_com = left_palm_jpos - pred_seq_com_pos.detach() # BS X T X 3
    right_palm_to_obj_com = right_palm_jpos - pred_seq_com_pos.detach()
    relative_left_palm_jpos = torch.matmul(pred_obj_rot_mat.detach().transpose(2, 3), \
                    left_palm_to_obj_com[:, :, :, None]).squeeze(-1) # BS X T X 3
    relative_right_palm_jpos = torch.matmul(pred_obj_rot_mat.detach().transpose(2, 3), \
                    right_palm_to_obj_com[:, :, :, None]).squeeze(-1)

    contact_labels = contact_labels.reshape(num_seq, num_steps, -1) # BS X T X 2

    # Expand dimensions of contact_labels for multiplication
    left_contact_labels_expanded = contact_labels[:, :, 0:1]
    left_contact_mask = left_contact_labels_expanded * left_contact_labels_expanded.transpose(-1, -2)

    right_contact_labels_expanded = contact_labels[:, :, 1:2]
    right_contact_mask = right_contact_labels_expanded * right_contact_labels_expanded.transpose(-1, -2) # BS X T X T

    left_norms = torch.norm(relative_left_palm_jpos, dim=-1, keepdim=True)
    left_normalized = relative_left_palm_jpos / left_norms
    left_similarity = torch.matmul(left_normalized, left_normalized.transpose(-1, -2))

    right_norms = torch.norm(relative_right_palm_jpos, dim=-1, keepdim=True)
    right_normalized = relative_right_palm_jpos / right_norms
    right_similarity = torch.matmul(right_normalized, right_normalized.transpose(-1, -2)) # BS X T X T

    loss_consistency = 1 - torch.mean(left_similarity * left_contact_mask) + \
                1 - torch.mean(right_similarity * right_contact_mask)

    loss = bs * (loss_contact + loss_consistency)

    return loss

def apply_feet_floor_contact_guidance(human_jnts):
    # human_jnts: BS X T X 28 X 3
    left_toe_idx = 10
    right_toe_idx = 11

    l_toe_height = human_jnts[:, :, left_toe_idx, 1:2] # BS X T X 1
    r_toe_height = human_jnts[:, :, right_toe_idx, 1:2] # BS X T X 1
    support_foot_height = torch.minimum(l_toe_height, r_toe_height) # BS X T X 1

    loss_feet_floor_contact = F.mse_loss(support_foot_height, torch.ones_like(support_foot_height)*0.02)

    loss = human_jnts.shape[0] * loss_feet_floor_contact

    return loss

def apply_hoi_guidance_loss(human_jnts, obj_verts, pred_seq_com_pos, pred_obj_rot_mat, contact_labels, scene_flag, get_nearest_free_voxel):
    # Hand-object contact + temporal consistency, plus feet-floor contact.
    loss_feet_floor_contact = apply_feet_floor_contact_guidance(human_jnts)

    loss_hand_object_interaction = apply_hand_object_interaction_guidance_loss(human_jnts, obj_verts, pred_seq_com_pos, pred_obj_rot_mat, contact_labels)
    loss = loss_hand_object_interaction * 10 + loss_feet_floor_contact * 500
    return loss

def apply_hosi_guidance_loss(human_jnts, obj_verts, pred_seq_com_pos, pred_obj_rot_mat, contact_labels, scene_flag, get_nearest_free_voxel):
    bs = human_jnts.shape[0]

    loss_hand_object_interaction = apply_hand_object_interaction_guidance_loss(human_jnts, obj_verts, pred_seq_com_pos, pred_obj_rot_mat, contact_labels)
    loss = loss_hand_object_interaction * 10

    # Floor-object penetration loss (y-up), moved out of apply_hand_object_interaction_guidance_loss.
    # Weight 100 = inner 10 (former loss_floor_object * 10) x outer 10 (loss_hand_object_interaction * 10).
    loss_floor_object = torch.minimum(obj_verts[:, :, :, -2], \
                torch.zeros_like(obj_verts[:, :, :, -2])).abs().mean()
    loss += bs * loss_floor_object * 100

    is_penetrating, nearest_free_points = get_nearest_free_voxel(human_jnts, scene_flag)
    loss += F.mse_loss(human_jnts, nearest_free_points) * 20000

    is_penetrating, nearest_free_points = get_nearest_free_voxel(obj_verts, scene_flag)
    loss += F.mse_loss(obj_verts, nearest_free_points) * 1000

    return loss
