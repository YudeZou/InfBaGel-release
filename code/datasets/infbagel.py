import os
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset
import pickle as pkl
import trimesh
import random
from datasets.utils import get_occupancy_from_npy, zup_to_yup, get_smpl_parents

from bps_torch.bps import bps_torch
import pytorch3d.transforms as transforms
from scipy.ndimage import distance_transform_edt

class InfBaGelDataset(Dataset):
    def __init__(self, folder, device, mesh_grid, batch_size, step, nb_voxels, train=True,
                 load_scene=True, load_language=True, load_pelvis_goal=False, load_hand_goal=False,
                 load_object_goal=False, use_random_frame_bps=False, use_object_keypoints=False,
                 max_window_size=16,
                 use_pi=True,
                 vis=True,
                 start_type='stand',
                 test_scene_name=None,
                 **kwargs):

        self.folder = folder
        self.device = device
        self.train = train
        self.load_scene = load_scene
        self.load_language = load_language
        self.load_pelvis_goal = load_pelvis_goal
        self.load_hand_goal = load_hand_goal
        self.load_object_goal = load_object_goal
        self.use_pi = use_pi
        self.vis = vis
        self.start_type = start_type
        self.test_scene_name = test_scene_name
        self.max_window_size = max_window_size

        self.rest_object_geo_folder = os.path.join(folder, 'rest_object_geo')
        self.global_orient = np.load(os.path.join(folder, 'human_orient.npy'))
        self.global_orient = zup_to_yup(self.global_orient)

        self.human_pose = np.load(os.path.join(folder, 'human_pose.npy'))
        self.human_pose = zup_to_yup(self.human_pose.reshape(-1, 3)).reshape(self.human_pose.shape)

        self.transl = np.load(os.path.join(folder, 'transl_aligned.npy'))
        
        betas_path = os.path.join(folder, 'betas.npy')
        self.betas = np.load(betas_path)

        gender_path = os.path.join(folder, 'gender.pkl')
        with open(gender_path, 'rb') as f:
            self.gender = pkl.load(f)
        
        self.joints = np.load(os.path.join(folder, 'human_joints_aligned.npy'))
        self.ori_sequence_start_idx = np.load(os.path.join(folder, 'start_idx.npy')).astype(np.int64)
        self.ori_sequence_end_idx = np.load(os.path.join(folder, 'end_idx.npy')).astype(np.int64)

        self.use_random_frame_bps = use_random_frame_bps
        self.use_object_keypoints = use_object_keypoints

        self.use_pen_loss = kwargs.get('use_pen_loss', False)

        self.parents_22 = get_smpl_parents(use_joints24=False) # 22
        self.parents_24 = get_smpl_parents(use_joints24=True) # 24
        if self.load_object_goal:
            # load object data
            self.object_rot_mat = np.load(os.path.join(folder, 'object_rot_mat.npy'))
            self.object_trans = np.load(os.path.join(folder, 'object_trans.npy'))
            if os.path.exists(os.path.join(folder, 'object_points.npy')):
                self.object_points = np.load(os.path.join(folder, 'object_points.npy'))
            else:
                self.object_points = None
            if self.use_object_keypoints:
                pass
                # self.transformed_obj_verts = np.load(os.path.join(folder, 'transformed_obj_verts.npy'))
                # self.rest_pose_obj_nn_pts = np.load(os.path.join(folder, 'rest_pose_obj_nn_pts.npy'))
            with open(os.path.join(folder, 'object_name.pkl'), 'rb') as f:
                self.object_name = pkl.load(f)
            # self.ori_w_idx = np.load(os.path.join(folder, 'ori_w_idx.npy'))
            if True: # self.train:
                self.dest_obj_bps_npy_folder = os.path.join(folder, 'cano_object_bps_npy_files_joints24_120')
            else:
                self.dest_obj_bps_npy_folder = os.path.join(folder, 'cano_object_bps_npy_files_for_test_joints24_120')
            self.rest_object_geo_folder = os.path.join(folder, 'rest_object_geo')

            self.obj_rest_verts = {}
            self.obj_vert_normals = {}
            for file in os.listdir(self.rest_object_geo_folder):
                if not file.endswith('.ply'):
                    continue
                obj_name = file.split('.')[0]
                rest_obj_path = os.path.join(self.rest_object_geo_folder, file)
                mesh = trimesh.load_mesh(rest_obj_path)
                rest_verts = np.asarray(mesh.vertices) # Nv X 3
                self.obj_rest_verts[obj_name] = torch.from_numpy(zup_to_yup(rest_verts)).float()
                vert_normals = np.asarray(mesh.vertex_normals) # Nv X 3
                self.obj_vert_normals[obj_name] = torch.from_numpy(zup_to_yup(vert_normals)).float()

        else:
            self.object_rot_mat = None
            self.object_trans = None
            self.object_points = None
            self.object_name = None
        
        self.rest_human_offsets = np.load(os.path.join(folder, 'rest_human_offsets_aligned.npy'))

        if self.load_language:
            if self.max_window_size == 16:
                language_motion_dict_filename = 'language_motion_dict__inter_and_loco__16.pkl'

            with open(os.path.join(self.folder, 'language_motion_dict', language_motion_dict_filename), 'rb') as f:
                language_motion_dict = pkl.load(f)
            self.end_range = language_motion_dict['end_range']
            self.text = language_motion_dict['text']

            self.clip_features = np.load(os.path.join(self.folder, 'clip_features.npy'))
            with open(os.path.join(self.folder, 'text2features_idx.pkl'), 'rb') as f:
                self.text2features_idx = pkl.load(f)

            self.need_scene = language_motion_dict['need_scene']
            self.need_pelvis_dir = language_motion_dict['need_pelvis_dir']
            self.pi = language_motion_dict['pi']
            self.need_pi = language_motion_dict['need_pi']
            self.left_hand_inter_frame = language_motion_dict['left_hand_inter_frame']
            self.right_hand_inter_frame = language_motion_dict['right_hand_inter_frame']

            self.start_ind = language_motion_dict['start_idx']
            self.end_ind = language_motion_dict['end_idx']
            
            if self.load_object_goal:
                self.need_object = language_motion_dict['need_object']
            
            self.ori_sequence_idx = language_motion_dict['ori_sequence_idx']

            # if self.vis:  # for sampling first two frames
            #     if self.start_type == 'stand':
            #         valid_idx = np.load('datasets/valid_idx_stand.npy')
            #         self.start_ind = self.start_ind[valid_idx]
            #         self.end_ind = self.end_ind[valid_idx]
            #         self.text = [self.text[idx] for idx in valid_idx]
            #     elif self.start_type == 'sit':
            #         valid_idx = np.load('datasets/valid_idx_sit.npy')
            #         self.start_ind = self.start_ind[valid_idx]
            #         self.end_ind = self.end_ind[valid_idx]
            #         self.text = [self.text[idx] for idx in valid_idx]

        self.step = step
        self.batch_size = batch_size

        if self.load_scene:
            self.mesh_grid = mesh_grid
            self.nb_voxels = nb_voxels
            self.scene_occ = []
            self.scene_occ_ref = []
            self.scene_dict = {}
            with open(os.path.join(folder, 'scene_name.pkl'), 'rb') as f:
                self.scene_name = pkl.load(f) # list of scene names
            if not self.vis:
                self.scene_folder = os.path.join(folder, 'Scene')
                scene_file_list = sorted(os.listdir(self.scene_folder))
            else:
                self.scene_folder = os.path.join(folder, 'Scene_vis')
                scene_file_list = sorted(os.listdir(self.scene_folder))
                scene_file_list = [file for file in scene_file_list if file.split('.')[0] == self.test_scene_name]

            for sid, file in enumerate(scene_file_list):
                # print(f"{sid} Loading Scene Mesh {file}")
                if 'occ' not in file:
                    scene_occ = np.load(os.path.join(self.scene_folder, file))
                    scene_occ = torch.from_numpy(scene_occ).to(device=device, dtype=bool)
                else:
                    scene_occ = np.load(os.path.join(self.scene_folder, file))
                
                self.scene_occ.append(scene_occ)
                self.scene_dict[file[:-4]] = sid
            if not self.vis and self.load_object_goal: # todo: can be optimized
                self.scene_occ = get_occupancy_from_npy(self.scene_occ)
                self.scene_occ = torch.from_numpy(self.scene_occ).to(device=self.device, dtype=bool)
                with open(os.path.join(folder, 'scene_name2file.pkl'), 'rb') as f:
                    self.scene_name2file = pkl.load(f)
            else:
                self.scene_occ = torch.stack(self.scene_occ)
            
            if self.vis:
                self.scene_occ_ref = self.compute_occ_ref(self.scene_occ)

            if not self.vis:
                self.scene_grid_np = np.array([-3, 0, -4, 3, 2, 4, 300, 100, 400])
                self.scene_grid_torch = torch.tensor([-3, 0, -4, 3, 2, 4, 300, 100, 400]).to(device)
            else:
                if 'demo' not in self.test_scene_name:
                    self.scene_grid_np = np.array([-3, 0, -4, 3, 2, 4, 300, 100, 400])
                    self.scene_grid_torch = torch.tensor([-3, 0, -4, 3, 2, 4, 300, 100, 400]).to(device)
                else:
                    self.scene_grid_np = np.array([-4, 0, -6, 4, 2, 6, 400, 100, 600])
                    self.scene_grid_torch = torch.tensor([-4, 0, -6, 4, 2, 6, 400, 100, 600]).to(device)

            # self.scene_grid_np = np.array([-3, 0, -4, 3, 2, 4, 300, 100, 400])
            # self.scene_grid_torch = torch.tensor([-3, 0, -4, 3, 2, 4, 300, 100, 400]).to(device)

            self.batch_id = torch.linspace(0, batch_size - 1, batch_size).tile((nb_voxels[0]*nb_voxels[1]*nb_voxels[2], 1)).T \
                .reshape(-1, 1).to(device=device, dtype=torch.long)

            if self.load_object_goal:
                self.batch_id_obj = torch.linspace(0, batch_size - 1, batch_size).tile((1024, 1)).T \
                    .reshape(-1, 1).to(device=device, dtype=torch.long)

        if self.max_window_size == 16:
            # norm = np.load(os.path.join(folder, 'norm_inter_and_loco__16frames.npy'))
            norm = np.load(os.path.join(folder, 'norm.npy'))

        self.min = norm[0].astype(np.float32)
        self.max = norm[1].astype(np.float32)
        self.min_torch = torch.tensor(self.min).to(device)
        self.max_torch = torch.tensor(self.max).to(device)

        if self.load_object_goal:
            self.obj_min = norm[2].astype(np.float32)
            self.obj_max = norm[3].astype(np.float32)
            self.obj_min_torch = torch.tensor(self.obj_min).to(device)
            self.obj_max_torch = torch.tensor(self.obj_max).to(device)
            self.obj_bps_data = []
            self.bps_dict = {}
            self.rest_bps_data = {}
            self.rest_obj_verts = {}

            start_idx = 0
            bps_file_list = sorted(os.listdir(self.dest_obj_bps_npy_folder))
            for bid, file in enumerate(bps_file_list):
                obj_bps_npy_path = os.path.join(self.dest_obj_bps_npy_folder, file)
                obj_bps_data = torch.from_numpy(np.load(obj_bps_npy_path))# .to(device) # T X 1024 X 3 
                self.obj_bps_data.append(obj_bps_data)
                self.bps_dict[file[:-4]] = (start_idx, start_idx + obj_bps_data.shape[0])
                start_idx += obj_bps_data.shape[0]
            self.obj_bps_data = torch.cat(self.obj_bps_data, dim=0)

            if self.use_object_keypoints:
                self.bps_torch = bps_torch()
                self.obj_bps = zup_to_yup(torch.load('./bps.pt')['obj'])
                for object_file in os.listdir(self.rest_object_geo_folder):
                    if object_file.endswith('.npy'):
                        object_name = object_file.split('.')[0]
                        self.rest_bps_data[object_name] = zup_to_yup(np.load(os.path.join(self.rest_object_geo_folder, object_file)))

        if self.load_object_goal:
            self.contact_label = {}
            contact_label_folder = '/cpfs04/shared/sport/zouyude/code/chois_release/processed_data/contact_labels_w_semantics_npy_files/'
            for file in os.listdir(contact_label_folder):
                self.contact_label[file[:-4]] = np.load(os.path.join(contact_label_folder, file))

    def set_test_scene(self, test_scene_name):
        """[accel] Switch test scene: recompute only scene-related data (scene_occ / scene_dict /
        scene_occ_ref / scene_grid), reusing all scene-independent loaded data (human/obj_bps/
        contact_label/clip etc.), avoiding repeated reads of large files when rebuilding the whole dataset.

        Logic is identical to the scene-related part of the self.load_scene branch in __init__; consumes no
        global RNG (torch/numpy/python), so the effect on results is zero, bit-for-bit identical."""
        self.test_scene_name = test_scene_name
        if not self.load_scene:
            return

        folder = self.folder
        device = self.device

        self.scene_occ = []
        self.scene_occ_ref = []
        self.scene_dict = {}
        with open(os.path.join(folder, 'scene_name.pkl'), 'rb') as f:
            self.scene_name = pkl.load(f)  # list of scene names
        if not self.vis:
            self.scene_folder = os.path.join(folder, 'Scene')
            scene_file_list = sorted(os.listdir(self.scene_folder))
        else:
            self.scene_folder = os.path.join(folder, 'Scene_vis')
            scene_file_list = sorted(os.listdir(self.scene_folder))
            scene_file_list = [file for file in scene_file_list if file.split('.')[0] == self.test_scene_name]

        for sid, file in enumerate(scene_file_list):
            if 'occ' not in file:
                scene_occ = np.load(os.path.join(self.scene_folder, file))
                scene_occ = torch.from_numpy(scene_occ).to(device=device, dtype=bool)
            else:
                scene_occ = np.load(os.path.join(self.scene_folder, file))

            self.scene_occ.append(scene_occ)
            self.scene_dict[file[:-4]] = sid
        if not self.vis and self.load_object_goal:
            self.scene_occ = get_occupancy_from_npy(self.scene_occ)
            self.scene_occ = torch.from_numpy(self.scene_occ).to(device=self.device, dtype=bool)
            with open(os.path.join(folder, 'scene_name2file.pkl'), 'rb') as f:
                self.scene_name2file = pkl.load(f)
        else:
            self.scene_occ = torch.stack(self.scene_occ)

        if self.vis:
            self.scene_occ_ref = self.compute_occ_ref(self.scene_occ)

        if not self.vis:
            self.scene_grid_np = np.array([-3, 0, -4, 3, 2, 4, 300, 100, 400])
            self.scene_grid_torch = torch.tensor([-3, 0, -4, 3, 2, 4, 300, 100, 400]).to(device)
        else:
            if 'demo' not in self.test_scene_name:
                self.scene_grid_np = np.array([-3, 0, -4, 3, 2, 4, 300, 100, 400])
                self.scene_grid_torch = torch.tensor([-3, 0, -4, 3, 2, 4, 300, 100, 400]).to(device)
            else:
                self.scene_grid_np = np.array([-4, 0, -6, 4, 2, 6, 400, 100, 600])
                self.scene_grid_torch = torch.tensor([-4, 0, -6, 4, 2, 6, 400, 100, 600]).to(device)

    def __getitem__(self, idx):
        if self.load_language:
            start_idx = int(self.start_ind[idx])
            end_idx = int(self.end_ind[idx])
            assert end_idx - start_idx == self.max_window_size * 3

            pelvis_goal = np.zeros((3, )).astype(np.float32)
            hand_goal = np.zeros((3, )).astype(np.float32)
            object_goal = np.zeros((3, )).astype(np.float32)
            is_pick = np.zeros((1, )).astype(bool)
            is_loco = False
            is_object = False

            text = self.text[idx][0]
            text_clip_embedding = self.clip_features[self.text2features_idx[text]]  # (1, 768)
            text_clip_embedding = torch.from_numpy(text_clip_embedding).float().reshape(1, -1)
            text_clip_embedding = text_clip_embedding / torch.norm(text_clip_embedding, dim=1, keepdim=True)
            # text_clip_embedding = np.zeros((1, 768)).astype(np.float32)
            
            left_hand_inter_frame = self.left_hand_inter_frame[idx]
            right_hand_inter_frame = self.right_hand_inter_frame[idx]
            if self.load_object_goal:
                is_object = self.need_object[idx]

            # if is_object:
            #     origin_sequence_idx = self.ori_sequence_idx[idx] # todo
            # else:
            #     origin_sequence_idx = start_idx
            origin_sequence_idx = self.ori_sequence_idx[idx] 

            if left_hand_inter_frame != -1:
                hand_goal = self.joints[left_hand_inter_frame, 24].copy()  # left hand index1
                is_pick = np.ones((1,)).astype(bool)
            elif right_hand_inter_frame != -1:
                hand_goal = self.joints[right_hand_inter_frame, 26].copy()  # right hand index1
                is_pick = np.ones((1,)).astype(bool)

            seq_len = self.ori_sequence_end_idx[origin_sequence_idx] - self.ori_sequence_start_idx[origin_sequence_idx]
            need_scene = self.need_scene[idx]
            need_pelvis_dir = self.need_pelvis_dir[idx]
            pi = self.pi[idx]
            need_pi = self.need_pi[idx]
            if need_pi and self.train:
                pi = pi + np.random.randint(-5, 5)
                pi = max(pi, 0)
                # pi = float(pi) / (self.ori_sequence_end_idx[origin_sequence_idx] - self.ori_sequence_start_idx[origin_sequence_idx])
            if not need_pi:
                pi = np.random.randint(0, seq_len - self.max_window_size * self.step)
                

            if need_pelvis_dir:
                if 'sit down' in text or 'lie down' in text:
                    # pelvis_goal = self.joints[int(self.end_range[idx]), 0].copy()
                    hand_goal = self.joints[int(self.end_range[idx]), 0].copy() # use hand goal to locate end pelvis goal
                    pelvis_goal = self.joints[end_idx-3, 0].copy() # align with chois
                else:
                    pelvis_goal = self.joints[end_idx-3, 0].copy()
                    is_loco = True
                pelvis_goal[1] = 0.

        joints = self.joints[start_idx: end_idx: self.step]
        init_joints = np.array([joints[0, 0, 0], 0., joints[0, 0, 2]]) # human's local frame
        joints = joints - init_joints
        pelvis_goal = pelvis_goal - init_joints
        hand_goal = hand_goal - init_joints

        if is_object:
            object_goal = self.object_trans[int(self.end_range[idx])-4].copy() - init_joints # human's local frame
            assert int(self.end_range[idx]) == int(self.ori_sequence_end_idx[origin_sequence_idx])
            # object_goal[1] = 0. (3-dim position represent final object position)
            if self.scene_name[origin_sequence_idx] in self.bps_dict:
                bps_start_idx, bps_end_idx = self.bps_dict[self.scene_name[origin_sequence_idx]]
                obj_bps_data = self.obj_bps_data[bps_start_idx:bps_end_idx]
                assert obj_bps_data.shape[0] == self.ori_sequence_end_idx[origin_sequence_idx] - self.ori_sequence_start_idx[origin_sequence_idx]
                if self.use_random_frame_bps:
                    random_sampled_t_idx = random.sample(list(range(obj_bps_data.shape[0])), 1)[0]
                else: # use the first frame of this window for object bps
                    random_sampled_t_idx = start_idx - self.ori_sequence_start_idx[origin_sequence_idx] 
                obj_bps_data = obj_bps_data[random_sampled_t_idx:random_sampled_t_idx+1] # 1 X 1024 X 3
                obj_bps_data = zup_to_yup(obj_bps_data)
                # bps_set = self.obj_bps + self.object_trans[self.ori_sequence_start_idx[origin_sequence_idx]+random_sampled_t_idx][None,None,:] # 1X1024X3
                # lhand_point = self.joints[self.ori_sequence_start_idx[origin_sequence_idx]+random_sampled_t_idx][24,:] # 3
                # rhand_point = self.joints[self.ori_sequence_start_idx[origin_sequence_idx]+random_sampled_t_idx][26,:] # 3
                # lhand_delta = torch.from_numpy(lhand_point[None, None, :]) - bps_set
                # rhand_delta = torch.from_numpy(rhand_point[None, None, :]) - bps_set
                # obj_bps_data = torch.cat([obj_bps_data, lhand_delta, rhand_delta], axis=-1) # 1 X 1024 X 9
        else:
            # print("obj_bps_npy not found: ", self.scene_name[origin_sequence_idx])
            obj_bps_data = torch.zeros((1, 1024, 3), dtype=torch.float32)

        # transform object goal to human's local frame
        if is_object:
            object_name = self.object_name[origin_sequence_idx]
            
            object_rot_mat = self.object_rot_mat[start_idx: end_idx: self.step] # human-relative rotation matrix
            if self.use_random_frame_bps:
                object_rot_mat_ref = self.object_rot_mat[self.ori_sequence_start_idx[origin_sequence_idx]: self.ori_sequence_end_idx[origin_sequence_idx]][random_sampled_t_idx]
            else:
                object_rot_mat_ref = object_rot_mat[0]
            object_rot_mat_orig = object_rot_mat.copy()
            object_rot_mat = self.prep_rel_obj_rot_mat_w_reference_mat(object_rot_mat, object_rot_mat_ref)
            
            object_trans = self.object_trans[start_idx: end_idx: self.step]
            object_trans_orig = object_trans.copy()
            object_trans = object_trans - init_joints
        else:
            object_name = "none"
            object_rot_mat = np.zeros((joints.shape[0], 3, 3))
            object_rot_mat_ref = object_rot_mat[0]
            object_trans = np.zeros((joints.shape[0], 3))
        
        if is_object and self.use_object_keypoints:
            rest_obj_bps_data = self.rest_bps_data[self.object_name[origin_sequence_idx]]
            nn_pts_on_mesh = self.obj_bps + torch.from_numpy(rest_obj_bps_data).float().to(self.obj_bps.device) # 1 X 1024 X 3 
            nn_pts_on_mesh = nn_pts_on_mesh.squeeze(0) # 1024 X 3 
            
            # random sample 100 points used for training
            # sampled_vidxs = random.sample(list(range(1024)), 100) 
            # sampled_nn_pts_on_mesh = nn_pts_on_mesh[sampled_vidxs] # 100 X 3 
            # rest_pose_obj_nn_pts = sampled_nn_pts_on_mesh.clone()
            rest_pose_obj_nn_pts = self.obj_rest_verts[object_name]
            indices = torch.randperm(rest_pose_obj_nn_pts.shape[0])[:100]
            rest_pose_obj_nn_pts = rest_pose_obj_nn_pts[indices] # 100 X 3
            sampled_nn_pts_on_mesh = rest_pose_obj_nn_pts.clone() # 100 X 3
            rest_pose_obj_normals = self.obj_vert_normals[object_name][indices] # 100 X 3

            # compute nn points for each frame
            object_rot_mat_orig = torch.from_numpy(object_rot_mat_orig).to(sampled_nn_pts_on_mesh.device) # T X 3 X 3
            object_trans_orig = torch.from_numpy(object_trans_orig).to(sampled_nn_pts_on_mesh.device)
            sampled_nn_pts_on_mesh = sampled_nn_pts_on_mesh[None].repeat(object_rot_mat_orig.shape[0], 1, 1) # T X 100 X 3
            transformed_obj_verts = object_rot_mat_orig.bmm(sampled_nn_pts_on_mesh.transpose(2, 1)) + \
                object_trans_orig.unsqueeze(2) # T X 3 X 100, in global frame
            transformed_obj_verts = transformed_obj_verts.transpose(1, 2) # T X 100 X 3
            # transformed_obj_verts = self.transformed_obj_verts[start_idx: end_idx: self.step]
            # rest_pose_obj_nn_pts = self.rest_pose_obj_nn_pts[origin_sequence_idx]
        else:
            transformed_obj_verts = torch.zeros((object_rot_mat.shape[0], 100, 3))
            rest_pose_obj_nn_pts = torch.zeros((100, 3))
            rest_pose_obj_normals = torch.zeros((100, 3))
        
        # rest_verts, obj_mesh_faces, transformed_obj_verts = \
        #             self.load_rest_pose_object_geometry_and_transform(
        #                 object_name, object_rot_mat, object_trans)

        global_orient = self.global_orient[start_idx: end_idx]
        init_global_orient = global_orient[0]
        init_global_orient_euler = R.from_rotvec(init_global_orient).as_euler('zxy')
        shift_euler = np.array([0, 0, -init_global_orient_euler[2]])
        shift_rot_matrix = R.from_euler('zxy', shift_euler).as_matrix()

        global_orient = torch.from_numpy(global_orient).reshape(-1, 1, 3) # T X 3 X 3
        human_pose = torch.from_numpy(self.human_pose[start_idx: end_idx]).reshape(-1, 21, 3) # T X 21 X 3

        local_rot_aa = torch.cat([global_orient, human_pose], dim=1) # T X 22 X 3
        local_rot_mat = transforms.axis_angle_to_matrix(local_rot_aa)
        global_rot_mat = self.local2global_pose(local_rot_mat) # T X 22 X 3 X 3

        global_rot_mat = torch.from_numpy(shift_rot_matrix).float()[None, None] @ global_rot_mat.float()

        global_rot_6d = transforms.matrix_to_rotation_6d(global_rot_mat) # T X 22 X 6

        mat = np.eye(4)
        mat[:3, :3] = np.linalg.inv(shift_rot_matrix.T).T
        mat[:3, 3] = init_joints
        mat = mat.astype(np.float32)

        joints = joints @ shift_rot_matrix.T
        pelvis_goal = pelvis_goal @ shift_rot_matrix.T
        hand_goal = hand_goal @ shift_rot_matrix.T
        if is_object:
            object_trans = object_trans @ shift_rot_matrix.T
            # object_rot_mat = mat[:3, :3] @ object_rot_mat
            object_goal = object_goal @ shift_rot_matrix.T

        # if is_loco and not is_object:
        #     pelvis_goal_norm = np.linalg.norm(pelvis_goal)
        #     if pelvis_goal_norm >= 0.8:
        #         pelvis_goal = pelvis_goal / pelvis_goal_norm * 0.8

        # if is_object:
        #     object_goal_norm = np.linalg.norm(object_goal)
        #     if object_goal_norm >= 0.8:
        #         object_goal = object_goal / object_goal_norm * 0.8

        joints = self.normalize(joints)
        joints = joints.astype(np.float32).reshape((joints.shape[0], -1))

        if is_object:
            object_trans = self.normalize(object_trans, is_object=True)

        if not self.vis and self.load_scene:
            if self.load_object_goal:
                scene_flag = self.scene_dict[f'occ_{self.scene_name2file[self.scene_name[origin_sequence_idx]]}']
            else:
                scene_flag = self.scene_dict[self.scene_name[start_idx]]
        else:
            scene_flag = 0

        if not self.use_pi:
            pi = 0
            need_pi = False

        if is_object and self.object_points is not None:
            object_points = self.object_points[start_idx]
        else:
            object_points = np.zeros((1024, 3))

        if is_object and self.load_object_goal:
            contact_label = self.contact_label[self.scene_name[origin_sequence_idx]]
            contact_label = contact_label[start_idx - self.ori_sequence_start_idx[origin_sequence_idx]: end_idx - self.ori_sequence_start_idx[origin_sequence_idx]: self.step]
        else:
            contact_label = np.zeros((len(joints), 4))

        if self.train:
            transl = self.transl[start_idx] - self.joints[start_idx][0]
        else:
            transl = self.transl[origin_sequence_idx]

        info = {
            'joints': joints.astype(np.float32),
            'global_rot_6d': global_rot_6d[::self.step],
            'mat': mat.astype(np.float32),
            'object_trans': object_trans.astype(np.float32),
            'object_rot_mat': object_rot_mat.astype(np.float32),
            'scene_flag': scene_flag,
            'text_clip_embedding': text_clip_embedding,
            'pelvis_goal': pelvis_goal.astype(np.float32),
            'hand_goal': hand_goal.astype(np.float32),
            'object_goal': object_goal.astype(np.float32),
            'is_pick': is_pick.astype(np.float32),
            'need_scene': need_scene,
            'need_pelvis_dir': need_pelvis_dir,
            'pi': pi,
            'need_pi': need_pi,
            'is_loco': is_loco,
            'is_object': is_object,
            'obj_bps_data': obj_bps_data,
            'obj_rot_mat_ref': object_rot_mat_ref.astype(np.float32),
            'rest_pose_obj_nn_pts': rest_pose_obj_nn_pts,
            'rest_pose_obj_normals': rest_pose_obj_normals,
            'transformed_obj_verts': transformed_obj_verts,
            'object_points': object_points,
            'seq_name': self.scene_name[origin_sequence_idx],
            'contact_label': contact_label.astype(np.float32),
            'joints_gt': self.joints[start_idx: end_idx],
            'global_rot_6d_gt': global_rot_6d,
            'object_trans_gt': self.object_trans[start_idx: end_idx] if self.object_trans is not None else np.zeros((48, 3)),
            'object_rot_mat_gt': self.object_rot_mat[start_idx: end_idx] if self.object_rot_mat is not None else np.zeros((48, 3, 3)),
            'rest_human_offsets': self.rest_human_offsets[origin_sequence_idx].astype(np.float32),
            'transl': transl,
            'betas': self.betas[origin_sequence_idx],
            'gender': self.gender[origin_sequence_idx],
            'seg_len': seq_len,
            'end_pi': min(pi + self.max_window_size * self.step, self.ori_sequence_end_idx[origin_sequence_idx] - self.ori_sequence_start_idx[origin_sequence_idx]),
            'object_name': object_name
        }
        
        return info
        # return joints.astype(np.float32), mat.astype(np.float32), object_trans.astype(np.float32), object_rot_mat.astype(np.float32), \
        #         scene_flag, text_clip_embedding, pelvis_goal.astype(np.float32), hand_goal.astype(np.float32), object_goal.astype(np.float32), \
        #         is_pick, need_scene, need_pelvis_dir, int(pi), need_pi, is_loco, is_object, obj_bps_data, object_rot_mat_ref.astype(np.float32)

    def get_pene_occ_count(self, points, scene_flag):
        occ = (self.scene_occ[scene_flag]).to(dtype=torch.int8).clone().to(dtype=torch.int8)

        T, N = points.shape[0], points.shape[1]
        points = points.reshape(-1, 3)
        voxel_size = torch.div(self.scene_grid_torch[3: 6] - self.scene_grid_torch[:3], self.scene_grid_torch[6:])
        voxel = torch.div((points - self.scene_grid_torch[:3]), voxel_size) # [T * N, 3]
        voxel = voxel.to(dtype=torch.long)
        # voxel = rearrange(voxel, 'b p c -> (b p) c')
        lb = torch.all(voxel >= 0, dim=-1)
        ub = torch.all(voxel < self.scene_grid_torch[6:] - 0, dim=-1)
        in_bound = torch.logical_and(lb, ub)
        voxel[torch.logical_not(in_bound)] = 0
        voxel = voxel.reshape(T, N, -1)
        
        t_idx = torch.arange(T, device=occ.device).unsqueeze(1).expand(T, N)
        # Find all positions with value 1 and set them to 3

        if not self.vis:
            mask = (occ[t_idx, voxel[..., 0], voxel[..., 1], voxel[..., 2]] == 1)
            occ[t_idx[mask], voxel[..., 0][mask], voxel[..., 1][mask], voxel[..., 2][mask]] = 3
            # Count the number of 3s in occ (number of penetrating occ)
            pene_count = torch.sum(occ == 3, dim=(1, 2, 3)).cpu().numpy()
        else:
            mask = (occ[voxel[..., 0], voxel[..., 1], voxel[..., 2]] == 1)
            occ[voxel[..., 0][mask], voxel[..., 1][mask], voxel[..., 2][mask]] = 3
            # Count the number of 3s in occ (number of penetrating occ)
            pene_count = torch.sum(occ == 3, dim=(1, 2)).cpu().numpy()
        
        # mask = (occ[0, voxel[..., 0], voxel[..., 1], voxel[..., 2]] == 1)
        # occ[0, voxel[..., 0][mask], voxel[..., 1][mask], voxel[..., 2][mask]] = 3
        # pene_count = torch.sum(occ == 3).cpu().numpy()

        return pene_count

    def add_object_points(self, points, occ):
        points = points.reshape(-1, 3)
        voxel_size = torch.div(self.scene_grid_torch[3: 6] - self.scene_grid_torch[:3], self.scene_grid_torch[6:])
        voxel = torch.div((points - self.scene_grid_torch[:3]), voxel_size)
        voxel = voxel.to(dtype=torch.long)
        # voxel = rearrange(voxel, 'b p c -> (b p) c')
        lb = torch.all(voxel >= 0, dim=-1)
        ub = torch.all(voxel < self.scene_grid_torch[6:] - 0, dim=-1)
        in_bound = torch.logical_and(lb, ub)
        voxel[torch.logical_not(in_bound)] = 0
        if self.train:
            voxel = torch.cat([self.batch_id_obj, voxel], dim=-1)
        # voxel = voxel[in_bound]
        if self.train:
            occ[voxel[:, 0], voxel[:, 1], voxel[:, 2], voxel[:, 3]] = 2 # 2 represents object (todo: object index?)
        else:
            if self.vis:
                occ[0, voxel[:, 0], voxel[:, 1], voxel[:, 2]] = 2
            else:
                voxel = torch.cat([self.batch_id_obj, voxel], dim=-1)
                occ[voxel[:, 0], voxel[:, 1], voxel[:, 2], voxel[:, 3]] = 2
                # occ = occ.unsqueeze(0)
                # occ[0, voxel[:, 0], voxel[:, 1], voxel[:, 2]] = 2

    def get_occ_for_points(self, points, obj_points, scene_flag):
        batch_size = points.shape[0]
        seq_len = points.shape[1]
        points = points.reshape(-1, 3)
        voxel_size = torch.div(self.scene_grid_torch[3: 6] - self.scene_grid_torch[:3], self.scene_grid_torch[6:]) # 0.02
        voxel = torch.div((points - self.scene_grid_torch[:3]), voxel_size)
        voxel = voxel.to(dtype=torch.long)
        lb = torch.all(voxel >= 0, dim=-1)
        ub = torch.all(voxel < self.scene_grid_torch[6:] - 0, dim=-1)
        in_bound = torch.logical_and(lb, ub)
        voxel[torch.logical_not(in_bound)] = 0

        self.batch_id = torch.linspace(0, batch_size - 1, batch_size).tile((self.nb_voxels[0]*self.nb_voxels[1]*self.nb_voxels[2], 1)).T \
            .reshape(-1, 1).to(device=points.device, dtype=torch.long)
        
        if self.train:
            voxel = torch.cat([self.batch_id, voxel], dim=1)

        occ = (self.scene_occ[scene_flag]).to(dtype=torch.int8)

        if self.load_object_goal:
            self.batch_id_obj = torch.linspace(0, batch_size - 1, batch_size).tile((1024, 1)).T \
                .reshape(-1, 1).to(device=points.device, dtype=torch.long)

        if obj_points is not None:
            self.add_object_points(obj_points, occ)

        if self.train:
            occ_for_points = occ[voxel[:, 0], voxel[:, 1], voxel[:, 2], voxel[:, 3]]
        else:
            if self.vis:
                occ_for_points = occ[0, voxel[:, 0], voxel[:, 1], voxel[:, 2]]
            else:
                voxel = torch.cat([self.batch_id, voxel], dim=1)
                occ_for_points = occ[voxel[:, 0], voxel[:, 1], voxel[:, 2], voxel[:, 3]]
                # occ = occ.unsqueeze(0)
                # occ_for_points = occ[0, voxel[:, 0], voxel[:, 1], voxel[:, 2]]
        occ_for_points[torch.logical_not(in_bound)] = 1 # 1 represents occupied
        occ_for_points = occ_for_points.reshape(batch_size, seq_len, -1)
        
        return occ_for_points

    def get_nearest_free_voxel(self, points, scene_flag):
        with torch.no_grad():
            occ = self.scene_occ[scene_flag]
            occ_ref = self.scene_occ_ref[scene_flag]

        original_shape = points.shape[:-1]
        batch_size = points.shape[0]
        seq_len = points.shape[1]
        N = points.shape[2]

        points_flat = points.reshape(-1, 3)
        
        voxel_size = torch.div(self.scene_grid_torch[3: 6] - self.scene_grid_torch[:3], self.scene_grid_torch[6:])
        voxel_indices = torch.div(points_flat - self.scene_grid_torch[:3], voxel_size).long()

        valid_mask = torch.all((voxel_indices >= 0) & (voxel_indices < self.scene_grid_torch[6:] - 0), dim=-1)
        voxel_indices[torch.logical_not(valid_mask)] = 0
        voxel_indices = voxel_indices.reshape(batch_size, seq_len*N, 3)

        b_idx = torch.arange(batch_size, device=occ.device).unsqueeze(1).expand(batch_size, seq_len*N)

        is_penetrating = (occ[b_idx, voxel_indices[..., 0], voxel_indices[..., 1], voxel_indices[..., 2]] == 1)
        valid_mask = valid_mask.reshape(is_penetrating.shape)

        nearest_free_points = points_flat.clone().reshape(batch_size, seq_len*N, 3)

        # For penetrating points, get the displacement and compute the safe position
        penetrating_mask = valid_mask & is_penetrating
        if penetrating_mask.any():
            pen_indices = voxel_indices[penetrating_mask]
            # Get the displacement vector directly from the occ_ref tensor
            displacements = occ_ref[b_idx[penetrating_mask], pen_indices[:, 0], pen_indices[:, 1], pen_indices[:, 2]]
            # Compute the safe position
            nearest_free_points[penetrating_mask] = (pen_indices + displacements) * voxel_size + self.scene_grid_torch[:3]
            # import pdb; pdb.set_trace()
        
        return is_penetrating.reshape(original_shape), nearest_free_points.reshape(*original_shape, 3)

    def create_meshgrid(self, batch_size=1):
        bbox = self.mesh_grid
        size = (self.nb_voxels[0], self.nb_voxels[1], self.nb_voxels[2])
        x = torch.linspace(bbox[0], bbox[1], size[0])
        y = torch.linspace(bbox[2], bbox[3], size[1])
        z = torch.linspace(bbox[4], bbox[5], size[2])
        xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
        grid = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
        grid = grid.repeat(batch_size, 1, 1)

        return grid

    def __len__(self):
        return len(self.start_ind)

    def normalize(self, data, is_object=False):
        shape_orig = data.shape
        data = data.reshape((-1, 3))
        if is_object:
            data = -1. + 2. * (data - self.obj_min) / (self.obj_max - self.obj_min)
        else:
            data = -1. + 2. * (data - self.min) / (self.max - self.min)
        data = data.reshape(shape_orig)

        return data

    def normalize_torch(self, data, is_object=False):
        shape_orig = data.shape
        data = data.reshape((-1, 3))
        if is_object:
            data = -1. + 2. * (data - self.obj_min_torch) / (self.obj_max_torch - self.obj_min_torch)
        else:
            data = -1. + 2. * (data - self.min_torch) / (self.max_torch - self.min_torch)
        data = data.reshape(shape_orig)

        return data

    def denormalize(self, data, is_object=False):
        shape_orig = data.shape
        data = data.reshape((-1, 3))
        if is_object:
            data = (data + 1.) * (self.obj_max - self.obj_min) / 2. + self.obj_min
        else:
            data = (data + 1.) * (self.max - self.min) / 2. + self.min
        data = data.reshape(shape_orig)

        return data
    
    def denormalize_torch(self, data, is_object=False):
        shape_orig = data.shape
        data = data.reshape((-1, 3))
        if is_object:
            data = (data + 1.) * (self.obj_max_torch - self.obj_min_torch) / 2. + self.obj_min_torch
        else:
            data = (data + 1.) * (self.max_torch - self.min_torch) / 2. + self.min_torch
        data = data.reshape(shape_orig)

        return data

    # load rest pose object geometry from chois dataset
    def load_rest_pose_object_geometry_and_transform(self, object_name, obj_rot, obj_com_pos):
        rest_obj_path = os.path.join(self.rest_object_geo_folder, object_name+".ply")
        
        mesh = trimesh.load_mesh(rest_obj_path)
        rest_verts = np.asarray(mesh.vertices) # Nv X 3
        obj_mesh_faces = np.asarray(mesh.faces) # Nf X 3
    
        rest_verts = rest_verts[None].repeat(obj_rot.shape[0], 1, 1)
        transformed_obj_verts = obj_rot.bmm(rest_verts.transpose(1, 2)) + obj_com_pos[:, :, None]
        transformed_obj_verts = transformed_obj_verts.transpose(1, 2) # T X Nv X 3 

        return rest_verts, obj_mesh_faces, transformed_obj_verts
    
    def prep_rel_obj_rot_mat_w_reference_mat(self, obj_rot_mat, ref_rot_mat):
        # obj_rot_mat: T X 3 X 3 / BS X T X 3 X 3 
        # ref_rot_mat: BS X 1 X 3 X 3/ 1 X 3 X 3 
        obj_rot_mat = torch.tensor(obj_rot_mat)
        ref_rot_mat = torch.tensor(ref_rot_mat)
        if obj_rot_mat.dim() == 4:
            timesteps = obj_rot_mat.shape[1]

            init_obj_rot_mat = ref_rot_mat.repeat(1, timesteps, 1, 1) # BS X T X 3 X 3
            rel_rot_mat = torch.matmul(obj_rot_mat, init_obj_rot_mat.transpose(2, 3)) # BS X T X 3 X 3
        else:
            timesteps = obj_rot_mat.shape[0]

            # Compute relative rotation matrix with respect to the first frame's object geometry. 
            init_obj_rot_mat = ref_rot_mat.repeat(timesteps, 1, 1) # T X 3 X 3
            # R_rel = R_obj @ R_ref^T
            rel_rot_mat = torch.matmul(obj_rot_mat, init_obj_rot_mat.transpose(1, 2)) # T X 3 X 3
        return rel_rot_mat.cpu().numpy()
    
    def local2global_pose(self, local_pose):
        # local_pose: T X J X 3 X 3 
        kintree = self.parents_22 

        bs = local_pose.shape[0]

        local_pose = local_pose.view(bs, -1, 3, 3)

        global_pose = local_pose.clone()

        for jId in range(len(kintree)):
            parent_id = kintree[jId]
            if parent_id >= 0:
                global_pose[:, jId] = torch.matmul(global_pose[:, parent_id], global_pose[:, jId])

        return global_pose # T X J X 3 X 3 

    def quat_ik_torch(self, grot_mat):
        # grot: T X J X 3 X 3 
        parents = self.parents_22 

        grot = transforms.matrix_to_quaternion(grot_mat) # T X J X 4 

        res = torch.cat(
                [
                    grot[..., :1, :],
                    transforms.quaternion_multiply(transforms.quaternion_invert(grot[..., parents[1:], :]), \
                    grot[..., 1:, :]),
                ],
                dim=-2) # T X J X 4 

        res_mat = transforms.quaternion_to_matrix(res) # T X J X 3 X 3 

        return res_mat
    
    def quat_fk_torch(self, lrot_mat, lpos, use_joints24=True):
        # lrot: N X J X 3 X 3 (local rotation with reprect to its parent joint)
        # lpos: N X J/(J+2) X 3 (root joint is in global space, the other joints are offsets relative to its parent in rest pose)
        if use_joints24:
            parents = self.parents_24
        else:
            parents = self.parents_22 

        lrot = transforms.matrix_to_quaternion(lrot_mat)

        gp, gr = [lpos[..., :1, :]], [lrot[..., :1, :]]
        for i in range(1, len(parents)):
            gp.append(
                transforms.quaternion_apply(gr[parents[i]], lpos[..., i : i + 1, :]) + gp[parents[i]]
            )
            if i < lrot.shape[-2]:
                gr.append(transforms.quaternion_multiply(gr[parents[i]], lrot[..., i : i + 1, :]))

        res = torch.cat(gr, dim=-2), torch.cat(gp, dim=-2)

        return res

    def compute_occ_ref(self, occ):
        """Compute the reference position from each occupied voxel to the nearest free voxel in the scene

        Args:
            occ (torch.Tensor): scene occupancy grid of shape [B, W, H, D], 1 means occupied, 0 means free

        Returns:
            torch.Tensor: tensor of shape [B, W, H, D, 3], storing the displacement vector from each voxel to the nearest free voxel
        """
        # Compute the distance field to the nearest free voxel
        device = occ.device
        occ = occ.cpu().numpy()

        # Process each batch separately
        batch_size = occ.shape[0]
        batch_displacements = []

        for b in range(batch_size):
            dist_transform = distance_transform_edt(occ[b], return_distances=True, return_indices=True)
            indices = np.array(dist_transform[1])  # [3, W, H, D]

            # Create grid coordinates
            w, h, d = occ[b].shape
            x, y, z = np.meshgrid(np.arange(w), np.arange(h), np.arange(d), indexing='ij')
            coords = np.stack([x, y, z], axis=0)  # [3, W, H, D]

            # Compute the displacement vector for each position
            displacements = indices - coords  # [3, W, H, D]

            # Convert to the required output format [W, H, D, 3]
            displacements = np.transpose(displacements, (1, 2, 3, 0))
            batch_displacements.append(displacements)

        # Stack the results of all batches [B, W, H, D, 3]
        batch_displacements = np.stack(batch_displacements, axis=0)

        # Convert to a torch tensor
        occ_ref = torch.from_numpy(batch_displacements).to(device=device, dtype=torch.int16)

        return occ_ref