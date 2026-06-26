# InfBaGel: Human-Object-Scene Interaction Generation with Dynamic Perception and Iterative Refinement

<center><img src="assets/teaser.png" alt="HOSI motion synthesis" style="zoom:40%;" /></center>

#### This is the official code repository of **InfBaGel: Human-Object-Scene Interaction Generation with Dynamic Perception and Iterative Refinement** at **ICLR 2026**
#### [arXiv](https://arxiv.org/abs/2604.04843) | [Paper](https://openreview.net/pdf?id=TeyHNq4WlI) | [Project Page](https://yudezou.github.io/InfBaGel-page/) | [Dataset](https://huggingface.co/datasets/xdzouyd/infbagel-data)

Yude Zou, Junji Gong, Xing Gao✉, Zixuan Li, Tianxing Chen, Guanjie Zheng

*Shanghai Jiao Tong University, Shanghai Artificial Intelligence Laboratory, Sichuan University, Shenzhen University, The University of Hong Kong*

## Abstract

Human-object-scene interactions (HOSI) generation has broad applications in embodied AI, simulation, and animation. Unlike human-object interaction (HOI) and human-scene interaction (HSI), HOSI generation requires reasoning over dynamic object-scene changes, yet suffers from limited annotated data. To address these issues, we propose a coarse-to-fine instruction-conditioned interaction generation framework that is explicitly aligned with the iterative denoising process of a consistency model. In particular, we adopt a dynamic perception strategy that leverages trajectories from the preceding refinement to update scene context and condition subsequent refinement at each denoising step of consistency model, yielding consistent interactions. To further reduce physical artifacts, we introduce a bump-aware guidance that mitigates collisions and penetrations during sampling without requiring fine-grained scene geometry, enabling real-time generation. To overcome data scarcity, we design a hybrid training strategy that synthesizes pseudo-HOSI samples by injecting voxelized scene occupancy into HOI datasets and jointly trains with high-fidelity HSI data, allowing interaction learning while preserving realistic scene awareness.


# Dataset

The dataset is expected at `data/` (relative to the project root). The directory structure is as follows:

**`data/train/`** — OMOMO training set:
- **human_orient.npy:** — SMPL-X `global_orient` per frame.
- **human_pose.npy:** — SMPL-X `body_pose` per frame.
- **transl_aligned.npy:** — SMPL-X `transl` per frame.
- **betas.npy:** Shape parameters per sequence.
- **gender.pkl:** Gender per sequence.
- **human_joints_aligned.npy:** — joint positions (y-up).
- **start_idx.npy:** — start frame of each segment.
- **end_idx.npy:** — end frame of each segment.
- **rest_human_offsets_aligned.npy:** Rest-pose human joint offsets.
- **norm.npy:** Normalization parameters.
- **object_rot_mat.npy:** Object rotation matrix per frame.
- **object_trans.npy:** Object translation per frame.
- **object_points.npy:** Object surface point cloud.
- **object_name.pkl:** Sequence → object name mapping.
- **scene_name.pkl:** Frame → scene name.
- **scene_name2file.pkl:** Scene name → occupancy file mapping.
- **clip_features.npy:** Precomputed CLIP features for text annotations.
- **text2features_idx.pkl:** Text → CLIP feature index mapping.
- **rest_object_geo/:** Rest-pose object meshes (one `.npy` per object).
- **cano_object_bps_npy_files_joints24_120/:** Precomputed BPS features per sequence.
- **contact_label_npy_files/:** Per-frame hand-object contact labels per sequence.
- **language_motion_dict/:** Language annotation dictionaries.
- **Scene/:** Scene occupancy grids.

**`data/test/`** — OMOMO test set (same structure as `train/`, with `Scene_vis/` added for evaluation scenes).

**`data/dataset/`** — LINGO dataset (HSI only, no object motion):
- **human_orient.npy, human_pose.npy, transl_aligned.npy, betas.npy, gender.pkl**
- **human_joints_aligned.npy, start_idx.npy, end_idx.npy**
- **rest_human_offsets_aligned.npy, norm.npy**
- **scene_name.pkl, clip_features.npy, text2features_idx.pkl**
- **left_hand_inter_frame.npy:** Frame index of left hand-object contact per segment.
- **right_hand_inter_frame.npy:** Frame index of right hand-object contact per segment.
- **text_aug.pkl:** Augmented text annotations.
- **language_motion_dict/:** Language annotation dictionaries.
- **Scene/:** Scene occupancy grids (training).
- **Scene_vis/:** Scene occupancy grids (evaluation).

**`data/object/`** — Shared object geometry:
- **rest_object_geo/:** Rest-pose meshes (`.npy` / `.ply` / `.json` per object).
- **rest_object_sdf_256_npy_files/:** SDF grids for objects (256³).

**`data/hosi_test/`** — HOSI evaluation test cases:
- **Scene_sdf/:** Scene SDF files for test scenes.
- **data/:** Test case JSON files (one per sequence).
- **vis/:** Visualization.

# Prerequisites

- Python 3.8+
- CUDA-capable GPU (training uses 4 GPUs by default)
- Required Python packages (specified in `requirements.txt`)

## Installation

1. **Prepare Data and Model Files**:

    Place the following directories at the project root:
    - `data/` — download from [Hugging Face Dataset](https://huggingface.co/datasets/xdzouyd/infbagel-data) and place contents here
    - `checkpoint/` — pretrained consistency model checkpoints (trained on OMOMO dataset only); download [checkpoint.tar.gz](https://drive.google.com/open?id=1h6k38uogfQ9V_z4ZO2kfyi03EXTYsoCm) and extract
    - `smpl_models/` — SMPL-X body model files; download [smpl_models.tar.gz](https://drive.google.com/open?id=1IQGdCSd8HwTwPS-noBdBIf-smI79yZ2K) and extract

2. **Set Up Conda Environment**:

    ```sh
    conda create -n infbagel python=3.8 -y
    conda activate infbagel
    ```

    Install PyTorch with CUDA 11.7 wheels (compatible with CUDA 11.x / 12.x systems):
    ```sh
    pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 \
        --extra-index-url https://download.pytorch.org/whl/cu117
    ```

    Install PyTorch3D 0.7.8 from source (no pre-built wheel; requires a C++ compiler and may take 15–30 minutes):
    ```sh
    pip install "git+https://github.com/facebookresearch/pytorch3d.git@6020323d94675f67860f702c35cc34c3eccc48da"
    ```

3. **Install Python Packages**:
    ```sh
    pip install -r requirements.txt
    ```

## Training

Navigate to the `code` directory:

```bash
cd code
```

### Diffusion Model

Train the InfBaGel diffusion model:

```bash
python train_infbagel.py
```

The training script loads configuration from `config/config_train_infbagel.yaml`. Key hyperparameters (editable in the config):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 512 | Training batch size |
| `lr` | 1e-4 | Learning rate |
| `epochs` | 501 | Number of training epochs |
| `num_gpus` | 4 | Number of GPUs for DDP training |
| `ckpt_interval` | 20 | Checkpoint save interval (epochs) |

To train with mixed dataset:

```bash
python train_infbagel.py --config-name config_train_infbagel_mix
```

### Consistency Model

The consistency model is trained via consistency distillation from a pretrained diffusion model. First train the diffusion model, then set the checkpoint path in `config/config_train_infbagel_cm.yaml`:

```yaml
ckpt_path: "/path/to/diffusion/checkpoint.pth"
```

Then start distillation (fewer epochs, faster inference):

```bash
python train_infbagel.py --config-name config_train_infbagel_cm
```

The consistency model config (`config/config_train_infbagel_cm.yaml`) uses `sample_type: consistency`. Key hyperparameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 512 | Training batch size |
| `lr` | 1e-4 | Learning rate |
| `epochs` | 201 | Number of training epochs |
| `num_gpus` | 4 | Number of GPUs for DDP training |
| `ckpt_interval` | 20 | Checkpoint save interval (epochs) |
| `load_state_dict` | `true` | Load weights from `ckpt_path` (must be `true` for distillation) |
| `ckpt_path` | `""` | Path to pretrained diffusion model checkpoint |

To train with mixed dataset:

```bash
python train_infbagel.py --config-name config_train_infbagel_mix_cm
```

## Evaluation

Sampling behavior is controlled by `config/config_sample_infbagel.yaml`. Two key guidance parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `w` | `1` | Classifier-free guidance weight for scene voxel conditioning |
| `guidance_weight` | `1` | Classifier guidance weight applied during sampling |

### HOSI Evaluation (Human-Object-Scene Interaction)

```bash
cd code
python test_infbagel_hosi.py
```

### HOI Evaluation (Human-Object Interaction)

```bash
cd code
python test_infbagel_hoi.py
```

The `render_mesh_from_params.py` script restores human and object meshes from saved motion parameter pickle files and optionally renders them into a video via Blender.

```bash
cd code
python render_mesh_from_params.py --param_file <path/to/params.pkl> --output_dir ./restored_meshes
```

To also render a video (requires [Blender 3.6.3](https://www.blender.org/download/releases/3-6/) installed):

```bash
python render_mesh_from_params.py --param_file <path/to/params.pkl> --output_dir ./restored_meshes --render_video
```

> **Note:** Update `BLENDER_PATH` at the top of `render_mesh_from_params.py` to point to your local Blender executable before running with `--render_video`.


# Citation
```
@inproceedings{zou2026infbagel,
    title={InfBaGel: Human-Object-Scene Interaction Generation with Dynamic Perception and Iterative Refinement},
    author={Yude Zou and Junji Gong and Xing Gao and Zixuan Li and Tianxing Chen and Guanjie Zheng},
    booktitle={The Fourteenth International Conference on Learning Representations},
    year={2026},
    url={https://openreview.net/forum?id=TeyHNq4WlI}
}

@article{li2023omomo,
    title={Object Motion Guided Human Motion Synthesis},
    author={Li, Jiaman and Wu, Jiajun and Liu, C. Karen},
    journal={ACM Transactions on Graphics},
    volume={42},
    number={6},
    pages={1--11},
    year={2023},
    publisher={ACM New York, NY, USA}
}

@inproceedings{jiang2024lingo,
    title={Autonomous Character-Scene Interaction Synthesis from Text Instruction},
    author={Jiang, Nan and He, Zimo and Wang, Zi and Li, Hongjie and Chen, Yixin and Huang, Siyuan and Zhu, Yixin},
    booktitle={SIGGRAPH Asia 2024 Conference Papers},
    year={2024},
    doi={10.1145/3680528.3687595}
}

@inproceedings{jiang2024trumans,
    title={Scaling Up Dynamic Human-Scene Interaction Modeling},
    author={Jiang, Nan and Zhang, Zhiyuan and Li, Hongjie and Ma, Xiaoxuan and Wang, Zan and Chen, Yixin and Liu, Tengyu and Zhu, Yixin and Huang, Siyuan},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    pages={1477--1487},
    year={2024}
}
```
