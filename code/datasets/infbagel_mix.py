import torch
import numpy as np
from torch.utils.data import Dataset
from datasets.infbagel import InfBaGelDataset
import os

class InfBaGelMixDataset(Dataset):
    """Mixed dataset class, supports joint training of OMOMO and LINGO datasets"""
    
    def __init__(self, omomo_folder, lingo_folder, device, mesh_grid, batch_size, step, nb_voxels, train=True,
                 load_scene=True, load_language=True, load_pelvis_goal=False, load_scene_goal=False,
                 load_object_goal=False, use_random_frame_bps=False, use_object_keypoints=False,
                 max_window_size=16,
                 use_pi=True,
                 vis=True,
                 start_type='stand',
                 test_scene_name=None,
                 human_only_ratio=0.5,
                 lingo_scene_num=111,
                 lingo_data_ratio=1.0,
                 empty_omomo_scene=False,
                 lingo_only=False,
                 random_seed=42,
                 **kwargs):
        
        # Initialize the two datasets
        self.omomo_dataset = InfBaGelDataset(
            folder=omomo_folder,
            device=device,
            mesh_grid=mesh_grid,
            batch_size=batch_size,
            step=step,
            nb_voxels=nb_voxels,
            train=train,
            load_scene=load_scene,
            load_language=load_language,
            load_pelvis_goal=load_pelvis_goal,
            load_scene_goal=load_scene_goal,
            load_object_goal=True,  # OMOMO dataset loads objects
            use_random_frame_bps=use_random_frame_bps,
            use_object_keypoints=use_object_keypoints,
            max_window_size=max_window_size,
            use_pi=use_pi,
            vis=vis,
            start_type=start_type,
            test_scene_name=test_scene_name,
            **kwargs
        )
        
        self.lingo_dataset = InfBaGelDataset(
            folder=lingo_folder,
            device=device,
            mesh_grid=mesh_grid,
            batch_size=batch_size,
            step=step,
            nb_voxels=nb_voxels,
            train=train,
            load_scene=load_scene,
            load_language=load_language,
            load_pelvis_goal=load_pelvis_goal,
            load_scene_goal=load_scene_goal,
            load_object_goal=False,  # Human-only dataset does not load objects
            use_random_frame_bps=False,
            use_object_keypoints=False,
            max_window_size=max_window_size,
            use_pi=use_pi,
            vis=vis,
            start_type=start_type,
            test_scene_name=test_scene_name,
            **kwargs
        )
        
        # Extend the need_object field for the human-only dataset
        if hasattr(self.lingo_dataset, 'start_ind'):
            self.lingo_dataset.need_object = [False] * len(self.lingo_dataset.start_ind)

        # Save experiment configuration parameters
        self.lingo_scene_num = lingo_scene_num
        self.lingo_data_ratio = lingo_data_ratio
        self.empty_omomo_scene = empty_omomo_scene
        self.random_seed = random_seed
        self.human_only_ratio = human_only_ratio
        self.omomo_size = len(self.omomo_dataset)
        self.lingo_size = len(self.lingo_dataset)
        self.lingo_only = lingo_only

        # Save configuration parameters
        self.train = train
        self.device = device
        self.load_scene = load_scene
        self.nb_voxels = nb_voxels
        self.mesh_grid = mesh_grid
        self.batch_size = batch_size
        self.step = step
        self.load_scene = load_scene
        self.load_language = load_language
        self.load_pelvis_goal = load_pelvis_goal
        self.load_scene_goal = load_scene_goal
        self.load_object_goal = load_object_goal
        self.use_random_frame_bps = use_random_frame_bps
        self.use_object_keypoints = use_object_keypoints
        self.max_window_size = max_window_size
        self.use_pi = use_pi
        self.vis = vis
        self.test_scene_name = test_scene_name

        # Handle object-related normalization parameters
        if not hasattr(self.lingo_dataset, 'obj_min'):
            self.lingo_dataset.obj_min = self.omomo_dataset.obj_min
            self.lingo_dataset.obj_max = self.omomo_dataset.obj_max
            self.lingo_dataset.obj_min_torch = self.omomo_dataset.obj_min_torch
            self.lingo_dataset.obj_max_torch = self.omomo_dataset.obj_max_torch

        # Compute unified normalization parameters
        self._compute_unified_normalization_params(omomo_folder, lingo_folder)

        # Create the unified scene encoding mapping system
        self._create_unified_scene_mapping()

        # Create mixed indices
        self._create_mixed_indices()

    def _compute_unified_normalization_params(self, omomo_folder, lingo_folder):
        """Compute unified normalization parameters, ensuring all data lies in the same space"""

        if not self.lingo_only:
            # Compute unified human-data normalization parameters (take the min and max of the two datasets)
            # self.unified_min = np.minimum(self.omomo_dataset.min, self.lingo_dataset.min)
            # self.unified_max = np.maximum(self.omomo_dataset.max, self.lingo_dataset.max)
            # self.unified_min_torch = torch.minimum(self.omomo_dataset.min_torch, self.lingo_dataset.min_torch)
            # self.unified_max_torch = torch.maximum(self.omomo_dataset.max_torch, self.lingo_dataset.max_torch)

            self.unified_min = self.omomo_dataset.min
            self.unified_max = self.omomo_dataset.max
            self.unified_min_torch = self.omomo_dataset.min_torch
            self.unified_max_torch = self.omomo_dataset.max_torch

            # Object data uses the OMOMO dataset parameters (LINGO dataset has no objects)
            self.unified_obj_min = self.omomo_dataset.obj_min
            self.unified_obj_max = self.omomo_dataset.obj_max
            self.unified_obj_min_torch = self.omomo_dataset.obj_min_torch
            self.unified_obj_max_torch = self.omomo_dataset.obj_max_torch
        else:
            # When using only the LINGO dataset
            # self.unified_min = self.lingo_dataset.min
            # self.unified_max = self.lingo_dataset.max
            # self.unified_min_torch = self.lingo_dataset.min_torch
            # self.unified_max_torch = self.lingo_dataset.max_torch
            norm = np.load(os.path.join(lingo_folder, 'norm_inter_and_loco__16frames.npy'))

            self.unified_min = norm[0].astype(np.float32)
            self.unified_max = norm[1].astype(np.float32)
            self.unified_min_torch = torch.tensor(self.unified_min).to(self.lingo_dataset.device)
            self.unified_max_torch = torch.tensor(self.unified_max).to(self.lingo_dataset.device)

            # (LINGO dataset has no objects) keep the default values
            self.unified_obj_min = self.lingo_dataset.obj_min
            self.unified_obj_max = self.lingo_dataset.obj_max
            self.unified_obj_min_torch = self.lingo_dataset.obj_min_torch
            self.unified_obj_max_torch = self.lingo_dataset.obj_max_torch

    def _create_unified_scene_mapping(self):
        """Create the unified scene encoding mapping system"""
        self.unified_scene_dict = {}
        self.scene_flag_mapping = {}
        self.unified_scene_source = {}  # Added: record the dataset source of each unified code

        current_flag = 0

        if not self.lingo_only:
            # First add the scenes from the OMOMO dataset
            for scene_name, flag in self.omomo_dataset.scene_dict.items():
                unified_flag = current_flag
                self.unified_scene_dict[scene_name] = unified_flag
                self.scene_flag_mapping[('omomo', flag)] = unified_flag
                self.unified_scene_source[unified_flag] = 'omomo'  # Mark as OMOMO source
                current_flag += 1

        # Then add the scenes from the LINGO dataset (avoiding duplicates)
        for scene_name, flag in self.lingo_dataset.scene_dict.items():
            unified_flag = current_flag
            self.unified_scene_dict[scene_name] = unified_flag
            self.scene_flag_mapping[('lingo', flag)] = unified_flag
            self.unified_scene_source[unified_flag] = 'lingo'  # Mark as LINGO source
            current_flag += 1


        # Merge scene occupancy data
        self._merge_scene_data()

        if not self.lingo_only:
            # Create the OMOMO scene boolean lookup table
            self._create_omomo_scene_mask()
    
    def _merge_scene_data(self):
        """Merge the scene occupancy data of the two datasets"""
        # Create the merged scene occupancy data list
        self.merged_scene_occ = []

        # Merge scene data in unified-code order
        scene_name_to_data = {}

        if not self.lingo_only:
            # Collect scene data from the OMOMO dataset
            for scene_name, original_flag in self.omomo_dataset.scene_dict.items():
                scene_data = self.omomo_dataset.scene_occ[original_flag]
                scene_name_to_data[scene_name] = scene_data

        # Collect scene data from the LINGO dataset (if not duplicated)
        for scene_name, original_flag in self.lingo_dataset.scene_dict.items():
            if scene_name not in scene_name_to_data:
                scene_data = self.lingo_dataset.scene_occ[original_flag]
                scene_name_to_data[scene_name] = scene_data

        # Organize data in unified-code order
        for unified_flag in range(len(self.unified_scene_dict)):
            scene_name = None
            for name, flag in self.unified_scene_dict.items():
                if flag == unified_flag:
                    scene_name = name
                    break
            
            if scene_name and scene_name in scene_name_to_data:
                self.merged_scene_occ.append(scene_name_to_data[scene_name])
        
        # Convert to torch.stack format
        if self.merged_scene_occ:
            self.merged_scene_occ = torch.stack(self.merged_scene_occ)

        if not self.lingo_only:
            # Release the original dataset scene data to save GPU memory
            if hasattr(self.omomo_dataset, 'scene_occ'):
                del self.omomo_dataset.scene_occ
                self.omomo_dataset.scene_occ = None  # Ensure it is deleted

        if hasattr(self.lingo_dataset, 'scene_occ'):
            del self.lingo_dataset.scene_occ
            self.lingo_dataset.scene_occ = None  # Ensure it is deleted

        # Force garbage collection to release GPU memory immediately
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    
    def _create_omomo_scene_mask(self):
        """Create the boolean lookup table for OMOMO scenes"""
        if len(self.unified_scene_source) == 0:
            self._omomo_scene_mask = torch.zeros(0, dtype=torch.bool, device=self.device)
            return
        
        max_scene_idx = max(self.unified_scene_source.keys()) + 1
        self._omomo_scene_mask = torch.zeros(max_scene_idx, dtype=torch.bool, device=self.device)
        
        for idx, source in self.unified_scene_source.items():
            if source == 'omomo':
                self._omomo_scene_mask[idx] = True
    

    def _filter_is_pick_entries(self, indices):
        """
        Filter out data entries with is_pick=True and entries with sequence length <= 48

        Args:
            indices: list of data indices to filter

        Returns:
            filtered_indices: list of indices after filtering
        """
        # Convert to a numpy array for vectorized operations
        indices_np = np.array(indices)

        # Batch-fetch left_hand_inter_frame and right_hand_inter_frame
        left_hand_frames = self.lingo_dataset.left_hand_inter_frame[indices_np]
        right_hand_frames = self.lingo_dataset.right_hand_inter_frame[indices_np]

        # Vectorized check: when either hand interaction frame is not -1, is_pick=True
        is_pick_mask = (left_hand_frames != -1) | (right_hand_frames != -1)

        # Get the original sequence length for each index
        origin_sequence_idx = self.lingo_dataset.ori_sequence_idx[indices_np]
        seq_lengths = self.lingo_dataset.ori_sequence_end_idx[origin_sequence_idx] - self.lingo_dataset.ori_sequence_start_idx[origin_sequence_idx]

        # Sequence length filter: keep sequences longer than 48
        seq_length_mask = seq_lengths <= 48

        # Combine the two filter conditions
        filter_mask = is_pick_mask | seq_length_mask

        # Keep the entries that meet the conditions
        filtered_indices = indices_np[~filter_mask].tolist()
        filtered_pick_count = np.sum(is_pick_mask)
        filtered_seq_count = np.sum(seq_length_mask)

        return filtered_indices

    def _create_mixed_indices(self):
        """Create the index mapping for the mixed dataset, supporting scene and data-ratio filtering"""
        omomo_size = len(self.omomo_dataset)

        # 1. Scene filtering (for the LINGO dataset)
        lingo_indices = list(range(len(self.lingo_dataset)))

        if self.lingo_scene_num < 111:
            # Get all unique scenes and their corresponding data indices
            scene_to_indices = {}
            for idx in lingo_indices:
                scene_flag = self.lingo_dataset.scene_dict[self.lingo_dataset.scene_name[int(self.lingo_dataset.start_ind[idx])]]
                if scene_flag not in scene_to_indices:
                    scene_to_indices[scene_flag] = []
                scene_to_indices[scene_flag].append(idx)

            # Select scenes in order (take the first n scenes)
            all_scenes = list(scene_to_indices.keys())
            n_total_scenes = len(all_scenes)
            # n_keep_scenes = max(1, int(n_total_scenes * self.lingo_scene_ratio))
            n_keep_scenes = self.lingo_scene_num

            # Select the first n scenes in order to ensure experiment stability
            selected_scenes = all_scenes[:n_keep_scenes]

            # Keep only the data indices of the selected scenes
            lingo_indices = []
            for scene_flag in selected_scenes:
                lingo_indices.extend(scene_to_indices[scene_flag])

            print(f"LINGO scene selection: Select {n_keep_scenes} scenes from {n_total_scenes} scenes")
            print(f"The selected scene contains {len(lingo_indices)} entries")

        # 2. is_pick filtering (filter out data entries with is_pick=True)
        lingo_indices = self._filter_is_pick_entries(lingo_indices)

        if not self.lingo_only:
            target_lingo_size = int(omomo_size * self.lingo_data_ratio)

            if target_lingo_size < len(lingo_indices):
                # Further filtering is needed to satisfy the data ratio
                np.random.seed(self.random_seed + 1)
                lingo_indices = np.random.choice(lingo_indices, target_lingo_size, replace=False).tolist()
                print(f"LINGO data proportion selection: Select {target_lingo_size} records from the data")
        else:
            omomo_size = 0

        # 3. Create the final index mapping
        indices = []

        if not self.lingo_only:
            # Add OMOMO data indices (dataset_id=0)
            for i in range(omomo_size):
                indices.append((0, i))

        # Add the filtered LINGO data indices (dataset_id=1)
        for idx in lingo_indices:
            indices.append((1, idx))

        # 4. Shuffle the indices (ensure randomness during training)
        np.random.seed(self.random_seed + 2)
        np.random.shuffle(indices)

        self.indices = indices

        print(f"Mixed dataset creation completed:")
        print(f"  - OMOMO: {omomo_size} seqs")
        print(f"  - LINGO: {len(lingo_indices)} seqs (scene: {self.lingo_scene_num}, proportion: {self.lingo_data_ratio})")
        print(f"  - All: {len(indices)} seqs")
    
    def __getitem__(self, idx):
        dataset_id, sample_idx = self.indices[idx]
        
        if dataset_id == 0:
            # Fetch from the OMOMO dataset
            info = self.omomo_dataset[sample_idx]
            original_scene_flag = info['scene_flag']
            # Convert to the unified scene encoding
            info['scene_flag'] = self.scene_flag_mapping[('omomo', original_scene_flag)]
        else:
            # Fetch from the LINGO dataset
            info = self.lingo_dataset[sample_idx]
            original_scene_flag = info['scene_flag']
            # Convert to the unified scene encoding
            info['scene_flag'] = self.scene_flag_mapping[('lingo', original_scene_flag)]

            # Ensure the LINGO-specific fields are set correctly
            info['is_object'] = False
            info['object_name'] = 'none'

            # Set object points to out-of-bound coordinates, relying on bound checks to filter them automatically
            if 'object_points' in info and info['object_points'] is not None:
                info['object_points'] = np.full_like(info['object_points'], 999.0).astype(np.float32)

        # Add the dataset identifier
        info['dataset_type'] = 'omomo' if dataset_id == 0 else 'lingo'
        return info
    
    def __len__(self):
        return len(self.indices)
    
    def create_meshgrid(self, batch_size):
        """Proxy to the create_meshgrid method of omomo_dataset"""
        return self.omomo_dataset.create_meshgrid(batch_size)
    
    def normalize(self, data, is_object=False, is_omomo=None):
        """
        Normalize data using the unified normalization parameters
        Args:
            data: input data
            is_object: whether this is object data (uses obj_min/obj_max)
            is_omomo: kept for backward compatibility, no longer used
        """
        # Ignore the is_omomo argument, use the unified normalization parameters
        _ = is_omomo
        shape_orig = data.shape
        data = data.reshape((-1, 3))

        if is_object:
            if self.unified_obj_min is not None and self.unified_obj_max is not None:
                data = -1. + 2. * (data - self.unified_obj_min) / (self.unified_obj_max - self.unified_obj_min)
            else:
                raise ValueError("Object normalization parameters are not set, please check the dataset configuration.")
        else:
            data = -1. + 2. * (data - self.unified_min) / (self.unified_max - self.unified_min)

        data = data.reshape(shape_orig)
        return data
    
    def normalize_torch(self, data, is_object=False, is_omomo=None):
        """
        Normalize data using the unified normalization parameters (PyTorch version)
        Args:
            data: input data
            is_object: whether this is object data (uses obj_min/obj_max)
            is_omomo: kept for backward compatibility, no longer used
        """
        # Ignore the is_omomo argument, use the unified normalization parameters
        _ = is_omomo
        shape_orig = data.shape
        data = data.reshape((-1, 3))

        if is_object:
            if self.unified_obj_min_torch is not None and self.unified_obj_max_torch is not None:
                data = -1. + 2. * (data - self.unified_obj_min_torch) / (self.unified_obj_max_torch - self.unified_obj_min_torch)
            else:
                raise ValueError("Object normalization parameters are not set, please check the dataset configuration.")
        else:
            data = -1. + 2. * (data - self.unified_min_torch) / (self.unified_max_torch - self.unified_min_torch)

        data = data.reshape(shape_orig)
        return data
    
    def denormalize(self, data, is_object=False, is_omomo=None):
        """
        Denormalize data using the unified normalization parameters
        Args:
            data: input data
            is_object: whether this is object data (uses obj_min/obj_max)
            is_omomo: kept for backward compatibility, no longer used
        """
        # Ignore the is_omomo argument, use the unified normalization parameters
        _ = is_omomo
        shape_orig = data.shape
        data = data.reshape((-1, 3))

        if is_object:
            if self.unified_obj_min is not None and self.unified_obj_max is not None:
                data = (data + 1.) * (self.unified_obj_max - self.unified_obj_min) / 2. + self.unified_obj_min
            else:
                raise ValueError("Object normalization parameters are not set, please check the dataset configuration.")
        else:
            data = (data + 1.) * (self.unified_max - self.unified_min) / 2. + self.unified_min

        data = data.reshape(shape_orig)
        return data
    
    def denormalize_torch(self, data, is_object=False, is_omomo=None):
        """
        Denormalize data using the unified normalization parameters (PyTorch version)
        Args:
            data: input data
            is_object: whether this is object data (uses obj_min/obj_max)
            is_omomo: kept for backward compatibility, no longer used
        """
        # Ignore the is_omomo argument, use the unified normalization parameters
        _ = is_omomo
        shape_orig = data.shape
        data = data.reshape((-1, 3))

        if is_object:
            if self.unified_obj_min_torch is not None and self.unified_obj_max_torch is not None:
                data = (data + 1.) * (self.unified_obj_max_torch - self.unified_obj_min_torch) / 2. + self.unified_obj_min_torch
            else:
                raise ValueError("Object normalization parameters are not set, please check the dataset configuration.")
        else:
            data = (data + 1.) * (self.unified_max_torch - self.unified_min_torch) / 2. + self.unified_min_torch

        data = data.reshape(shape_orig)
        return data
    
    def quat_ik_torch(self, grot_mat):
        """Proxy to the quat_ik_torch method of omomo_dataset"""
        return self.omomo_dataset.quat_ik_torch(grot_mat)
    
    def quat_fk_torch(self, lrot_mat, lpos, use_joints24=True):
        """Proxy to the quat_fk_torch method of omomo_dataset"""
        return self.omomo_dataset.quat_fk_torch(lrot_mat, lpos, use_joints24)
    
    def add_object_points(self, points, occ):
        points = points.reshape(-1, 3)
        voxel_size = torch.div(self.omomo_dataset.scene_grid_torch[3: 6] - self.omomo_dataset.scene_grid_torch[:3], self.omomo_dataset.scene_grid_torch[6:])
        voxel = torch.div((points - self.omomo_dataset.scene_grid_torch[:3]), voxel_size)
        voxel = voxel.to(dtype=torch.long)
        # voxel = rearrange(voxel, 'b p c -> (b p) c')
        lb = torch.all(voxel >= 0, dim=-1)
        ub = torch.all(voxel < self.omomo_dataset.scene_grid_torch[6:] - 0, dim=-1)
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
        voxel_size = torch.div(self.omomo_dataset.scene_grid_torch[3: 6] - self.omomo_dataset.scene_grid_torch[:3], self.omomo_dataset.scene_grid_torch[6:]) # 0.02
        voxel = torch.div((points - self.omomo_dataset.scene_grid_torch[:3]), voxel_size)
        voxel = voxel.to(dtype=torch.long)
        lb = torch.all(voxel >= 0, dim=-1)
        ub = torch.all(voxel < self.omomo_dataset.scene_grid_torch[6:] - 0, dim=-1)
        in_bound = torch.logical_and(lb, ub)
        voxel[torch.logical_not(in_bound)] = 0

        self.batch_id = torch.linspace(0, batch_size - 1, batch_size).tile((self.omomo_dataset.nb_voxels[0]*self.omomo_dataset.nb_voxels[1]*self.omomo_dataset.nb_voxels[2], 1)).T \
            .reshape(-1, 1).to(device=points.device, dtype=torch.long)
        
        if self.train:
            voxel = torch.cat([self.batch_id, voxel], dim=1)

        # Get the scene occupancy data
        occ = (self.merged_scene_occ[scene_flag]).to(dtype=torch.int8).clone()

        # Check whether the OMOMO scene needs to be cleared
        if self.empty_omomo_scene:
            # Use the precomputed lookup table to batch-determine which samples need clearing
            need_clear = self._omomo_scene_mask[scene_flag]

            # Batch-clear the OMOMO scene
            occ[need_clear] = 0

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