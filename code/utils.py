import torch
import hydra
import numpy as np
from einops import rearrange
import smplx
from constants import SMPL_DIR
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d

def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]

# From LCMScheduler.get_scalings_for_boundary_condition_discrete
def scalings_for_boundary_conditions(timestep, sigma_data=0.5, timestep_scaling=10.0):
    c_skip = sigma_data**2 / ((timestep / 0.1) ** 2 + sigma_data**2)
    c_out = (timestep / 0.1) / ((timestep / 0.1) ** 2 + sigma_data**2) ** 0.5
    return c_skip, c_out

def quaternion_slerp(q1, q2, step, eps=1e-6):
    # q1, q2: [..., 4]

    dot = torch.sum(q1 * q2, dim=-1, keepdim=True)

    # 1. Ensure a short arc
    q1 = torch.where(dot < 0, -q1, q1)
    dot = torch.sum(q1 * q2, dim=-1, keepdim=True)

    # 2. For critical cases, degradation to LERP
    use_lerp = dot > (1.0 - eps)

    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)
    factor0 = torch.sin((1 - step) * omega) / sin_omega
    factor1 = torch.sin(step * omega) / sin_omega

    slerped = q1 * factor0 + q2 * factor1
    lerped = q1 * step + q2 * (1 - step)

    result = torch.where(use_lerp, lerped, slerped)
    result = result / torch.norm(result, dim=-1, keepdim=True)

    return result

def interp_jrot(local_jrot_q, interp_s=3):
    # local_jrot_q: (T, 22, 4)
    # interp_s: default 3
    t, j, _ = local_jrot_q.shape
    local_jrot_q_interp = torch.zeros((t*interp_s, j, _)).to(local_jrot_q.device)

    # Interpolate over each time step
    for i in range(t-1):
        # Convert to quaternions
        quat1 = local_jrot_q[i]
        quat2 = local_jrot_q[i+1]

        # Quaternion interpolation
        for j in range(interp_s):
            t = j / interp_s
            # Spherical linear interpolation
            quat_interp = quaternion_slerp(quat1, quat2, t)
            local_jrot_q_interp[i*interp_s + j] = quat_interp

    # Handle the last frame
    local_jrot_q_interp[-interp_s:] = local_jrot_q[-1]

    return local_jrot_q_interp

def load_object_geometry_w_rest_geo(obj_rot, obj_com_pos, rest_verts):
    # obj_rot: T X 3 X 3, obj_com_pos: T X 3, rest_verts: Nv X 3
    rest_verts = rest_verts[None].repeat(obj_rot.shape[0], 1, 1)
    transformed_obj_verts = obj_rot.bmm(rest_verts.transpose(1, 2)) + obj_com_pos[:, :, None]
    transformed_obj_verts = transformed_obj_verts.transpose(1, 2) # T X Nv X 3

    return transformed_obj_verts  # T X Nv X 3

def load_object_geometry_w_rest_geo_and_normals(obj_rot, obj_com_pos, rest_verts, rest_normals):
    # obj_rot: T X 3 X 3, obj_com_pos: T X 3, rest_verts: Nv X 3, rest_normals: Nv X 3
    rest_verts = rest_verts[None].repeat(obj_rot.shape[0], 1, 1)
    rest_normals = rest_normals[None].repeat(obj_rot.shape[0], 1, 1)
    transformed_obj_verts = obj_rot.bmm(rest_verts.transpose(1, 2)) + obj_com_pos[:, :, None]
    transformed_obj_verts = transformed_obj_verts.transpose(1, 2) # T X Nv X 3
    transformed_obj_normals = obj_rot.bmm(rest_normals.transpose(1, 2))
    transformed_obj_normals = transformed_obj_normals.transpose(1, 2) # T X Nv X 3

    return transformed_obj_verts, transformed_obj_normals  # T X Nv X 3

def interp_object(object_trans, object_rot_mat, interp_s):
    # object_trans: (N, 3)
    # object_rot_mat: (N, 9)
    # interp_s: default 3

    # Initialize the interpolated arrays
    N = object_trans.shape[0]
    interp_trans = np.zeros((N * interp_s, 3))
    interp_rot_mat = np.zeros((N * interp_s, 9))

    # Interpolate over each time step
    for i in range(N-1):
        # Linearly interpolate the object position
        for j in range(interp_s):
            t = j / interp_s
            interp_trans[i*interp_s + j] = (1-t) * object_trans[i] + t * object_trans[i+1]

        # Use quaternions for spherical linear interpolation
        rot1 = object_rot_mat[i].reshape(3,3)
        rot2 = object_rot_mat[i+1].reshape(3,3)

        # Convert to quaternions
        quat1 = R.from_matrix(rot1).as_quat()
        quat2 = R.from_matrix(rot2).as_quat()

        quat1 = torch.from_numpy(quat1)
        quat2 = torch.from_numpy(quat2)

        # Quaternion interpolation
        for j in range(interp_s):
            t = j / interp_s
            quat_interp = quaternion_slerp(quat1, quat2, t)
            # Convert back to a rotation matrix
            quat_interp = quat_interp.numpy()
            rot_interp = R.from_quat(quat_interp).as_matrix()
            interp_rot_mat[i*interp_s + j] = rot_interp.reshape(-1)

    # Handle the last frame
    interp_trans[-interp_s:] = object_trans[-1]
    interp_rot_mat[-interp_s:] = object_rot_mat[-1]

    return interp_trans, interp_rot_mat

def transform_points(x, mat):
    shape = x.shape
    x = rearrange(x, 'b t (j c) -> b (t j) c', c=3)  # B x N x 3
    x = torch.einsum('bpc,bck->bpk', mat[:, :3, :3], x.permute(0, 2, 1))  # B x 3 x N   N x B x 3
    x = x.permute(2, 0, 1) + mat[:, :3, 3]
    x = x.permute(1, 0, 2)
    x = x.reshape(shape)

    return x


def create_meshgrid(bbox, size, batch_size=1):
    x = torch.linspace(bbox[0], bbox[1], size[0])
    y = torch.linspace(bbox[2], bbox[3], size[1])
    z = torch.linspace(bbox[4], bbox[5], size[2])
    xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
    grid = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
    grid = grid.repeat(batch_size, 1, 1)

    return grid


def zup_to_yup(coord):
    # change the coordinate from z-up to y-up
    if len(coord.shape) > 1:
        coord = coord[..., [0, 2, 1]]
        coord[..., 2] *= -1
    else:
        coord = coord[[0, 2, 1]]
        coord[2] *= -1

    return coord


def yup_to_zup(coord):
    # change the coordinate from y-up to z-up
    if len(coord.shape) > 1:
        coord = coord[..., [0, 2, 1]]
        coord[..., 1] *= -1
    else:
        coord = coord[[0, 2, 1]]
        coord[1] *= -1

    return coord


def yup_to_zup_rotation_matrix(rot_matrix):
    T_inv = torch.tensor([[1, 0, 0],
                          [0, 0, -1],
                          [0, 1, 0]], dtype=rot_matrix.dtype, device=rot_matrix.device)

    # Perform matrix multiplication
    return T_inv @ rot_matrix @ T_inv.T


def rigid_transform_3D(A, B, scale=False):
    assert len(A) == len(B)

    N = A.shape[0]  # total points

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # center the points
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))

    # dot is matrix multiplication for array
    if scale:
        H = np.transpose(BB) * AA / N
    else:
        H = np.transpose(BB) * AA

    U, S, Vt = np.linalg.svd(H)

    R = Vt.T * U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("Reflection detected")
        # return None, None, None
        Vt[2, :] *= -1
        R = Vt.T * U.T

    if scale:
        varA = np.var(A, axis=0).sum()
        c = 1 / (1 / varA * np.sum(S))  # scale factor
        t = -R * (centroid_B.T * c) + centroid_A.T
    else:
        c = 1
        t = -R * centroid_B.T + centroid_A.T

    return c, R, t


def find_free_port():
    from contextlib import closing
    import socket

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def init_model(model_cfg, device, eval, load_state_dict=False, need_ddp=True):
    model = hydra.utils.instantiate(model_cfg)
    if eval:
        load_state_dict_eval(model, model_cfg.ckpt, device=device)
    else:
        model = model.to(device)
        if need_ddp:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], broadcast_buffers=False,
                                                              find_unused_parameters=True)
        if load_state_dict:
            # Use strict=False to allow partial loading while recording missing keys
            checkpoint = torch.load(model_cfg.ckpt)
            missing_keys, unexpected_keys = model.module.load_state_dict(checkpoint, strict=False)

            if missing_keys:
                print(f"Missing keys in checkpoint (will use initialized values): {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys in checkpoint (will be ignored): {unexpected_keys}")

            model.train()

    return model


def load_state_dict_eval(model, state_dict_path, map_location='cuda:0', device='cuda'):
    state_dict = torch.load(state_dict_path, map_location=map_location)
    key_list = [key for key in state_dict.keys()]
    for old_key in key_list:
        new_key = old_key.replace('module.', '')
        state_dict[new_key] = state_dict.pop(old_key)

    # Use strict=False to allow partial loading
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    if missing_keys:
        print(f"[Eval] Missing keys in checkpoint (will use initialized values): {missing_keys}")
    if unexpected_keys:
        print(f"[Eval] Unexpected keys in checkpoint (will be ignored): {unexpected_keys}")

    model.to(device)
    model.eval()


def run_smplx_model(pose_pred, transl, betas, gender, joints_ind=None, smpl_model=None):
    # pose_pred: [b*s*42, 22, 3]
    # transl: [b*s*42, 3]
    # joints_ind: [28]
    # joints_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 28, 40, 43]
    joints_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 28, 43]
    device = pose_pred.device

    if smpl_model is None:
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


def interpolate_joints(joints, scale):
    if scale == 1:
        return joints
    device = joints.device
    joints = joints.detach().cpu().numpy()
    in_len = joints.shape[0]
    out_len = int(in_len * scale)
    joints = joints.reshape(in_len, -1)
    x = np.array(range(in_len))
    xnew = np.linspace(0, in_len - 1, out_len)
    f = interp1d(x, joints, axis=0)
    joints_new = f(xnew)
    joints_new = torch.from_numpy(joints_new).to(device).float()
    return joints_new
