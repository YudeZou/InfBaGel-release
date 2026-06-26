import numpy as np
import os

SMPLH_PATH = "/cpfs04/shared/sport/zouyude/code/chois_release/processed_data/smpl_all_models/smplh_amass"

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

import time
def get_occupancy_from_npy(data):
    # data = np.load(npy_path, allow_pickle=True).item()
    # Unpack bit data
    data = np.array(data)
    start_time = time.time()
    bs = data.shape[0]
    shape = [300, 100, 400]
    unpacked = np.unpackbits(data, axis=1)
    # Take only the required length (shape[0]*shape[1]*shape[2]) and reshape to 3D
    total_size = shape[0] * shape[1] * shape[2]
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    return 1 - unpacked[:, :total_size].reshape(bs, shape[0], shape[1], shape[2])

def get_smpl_parents(use_joints24=True):
    bm_path = os.path.join(SMPLH_PATH, 'male/model.npz')
    npz_data = np.load(bm_path)
    ori_kintree_table = npz_data['kintree_table'] # 2 X 52 

    if use_joints24:
        parents = ori_kintree_table[0, :23] # 23 
        parents[0] = -1 # Assign -1 for the root joint's parent idx.

        parents_list = parents.tolist()
        parents_list.append(ori_kintree_table[0][37])
        parents = np.asarray(parents_list) # 24 
    else:
        parents = ori_kintree_table[0, :22] # 22 
        parents[0] = -1 # Assign -1 for the root joint's parent idx.
    
    return parents