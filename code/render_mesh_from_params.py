#!/usr/bin/env python3
"""
Script to restore human and object meshes from saved motion parameters and visualize as video.

Usage:
    python render_mesh_from_params.py --param_file path/to/motion_params.pkl --output_dir ./output
"""

import os
import argparse
import pickle
import numpy as np
import torch
import trimesh
import smplx
import subprocess
import imageio
import shutil
import glob

# Import necessary functions from the original codebase
from constants import SMPL_DIR
from utils import yup_to_zup, zup_to_yup, load_object_geometry_w_rest_geo

# Blender rendering configuration (modify paths as needed)
BLENDER_PATH = "/cpfs04/shared/sport/zouyude/jjgong/blender-3.6.3-linux-x64/blender"  # Adjust to your blender executable path
BLENDER_UTILS_ROOT_FOLDER = "/cpfs04/shared/sport/zouyude/code/chois_release/manip/vis"
BLENDER_SCENE_FOLDER = "/cpfs04/shared/sport/zouyude/code/chois_release/processed_data/blender_files"
DEFAULT_SCENE_BLEND = os.path.join(BLENDER_SCENE_FOLDER, "floor_colorful_mat.blend")


def run_smplx_model(pose_pred, transl, betas, gender, joints_ind=None):
    """
    Run SMPL-X model to generate human mesh vertices and joints.
    
    Args:
        pose_pred: [T, 22, 3] - SMPL body pose in axis-angle representation
        transl: [T, 3] - root translation
        betas: [16] - SMPL shape parameters  
        gender: str - gender ('male', 'female', 'neutral')
        joints_ind: list - joint indices to extract
        
    Returns:
        vertices: [T, V, 3] - mesh vertices
        joints: [T, J, 3] - joint positions
    """
    if joints_ind is None:
        joints_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 28, 43]
    
    device = pose_pred.device if torch.is_tensor(pose_pred) else torch.device('cpu')
    
    # Convert to tensor if needed
    if not torch.is_tensor(pose_pred):
        pose_pred = torch.from_numpy(pose_pred).float().to(device)
    if not torch.is_tensor(transl):
        transl = torch.from_numpy(transl).float().to(device)
    if not torch.is_tensor(betas):
        betas = torch.from_numpy(betas).float().to(device)
    
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
    
    smpl_output = smpl_model(transl=transl, 
                           body_pose=pose_pred[:, 1:], 
                           global_orient=pose_pred[:, :1], 
                           betas=betas[None].repeat(pose_pred.shape[0], 1),
                           return_verts=True)
    
    return smpl_output.vertices, smpl_output.joints[:, joints_ind].reshape(pose_pred.shape[0], -1, 3)


def load_object_rest_geometry(rest_verts_root="/cpfs04/shared/sport/zouyude/code/chois_release/processed_data/rest_object_geo"):
    """
    Load object rest geometry (vertices and faces) from PLY files.
    
    Args:
        rest_verts_root: str - path to directory containing object PLY files
        
    Returns:
        obj_rest_data: dict - mapping from object names to mesh data
    """
    obj_rest_data = {}
    
    if not os.path.exists(rest_verts_root):
        print(f"Warning: Object geometry directory {rest_verts_root} not found!")
        return obj_rest_data
        
    for file in os.listdir(rest_verts_root):
        if not file.endswith('.ply'):
            continue
        obj_name = file.split('.')[0]
        rest_obj_path = os.path.join(rest_verts_root, file)
        
        try:
            mesh = trimesh.load_mesh(rest_obj_path)
            rest_verts = np.asarray(mesh.vertices)  # [V, 3]
            faces = np.asarray(mesh.faces)  # [F, 3]
            
            obj_rest_data[obj_name] = {
                'vertices': torch.from_numpy(zup_to_yup(rest_verts)).float(),
                'faces': faces
            }
            print(f"Loaded object {obj_name}: {rest_verts.shape[0]} vertices, {faces.shape[0]} faces")
        except Exception as e:
            print(f"Error loading object {obj_name}: {e}")
            
    return obj_rest_data


def restore_human_mesh(human_params, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Restore human mesh from saved parameters.
    
    Args:
        human_params: dict - contains pose_pred, root_trans, betas, gender
        device: str - device to run computation on
        
    Returns:
        vertices: [T, V, 3] - human mesh vertices
        joints: [T, J, 3] - human joint positions  
        faces: [F, 3] - mesh face connectivity
    """
    pose_pred = torch.from_numpy(human_params['pose_pred']).float().to(device)
    root_trans = torch.from_numpy(human_params['root_trans']).float().to(device)
    betas = torch.from_numpy(human_params['betas']).float().to(device)
    gender = human_params['gender']
    
    print(f"Restoring human mesh: {pose_pred.shape[0]} frames, gender: {gender}")
    
    vertices, joints = run_smplx_model(pose_pred.reshape(-1, 22, 3), root_trans, betas, gender)
    
    # Get face connectivity from SMPL model
    device_cpu = torch.device('cpu')
    smpl_model = smplx.create(SMPL_DIR, model_type='smplx',
                              gender=gender, ext='npz',
                              num_betas=16,
                              use_pca=False,
                              batch_size=1).to(device_cpu)
    faces = smpl_model.faces.astype(np.int32)
    
    return vertices.detach().cpu().numpy(), joints.detach().cpu().numpy(), faces


def restore_object_mesh(object_params, obj_rest_data, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Restore object mesh from saved parameters.
    
    Args:
        object_params: dict - contains obj_trans, obj_rot_mat, obj_name
        obj_rest_data: dict - object rest geometry data
        device: str - device to run computation on
        
    Returns:
        vertices: [T, V, 3] - object mesh vertices
        faces: [F, 3] - mesh face connectivity
    """
    obj_name = object_params['obj_name']
    obj_trans = torch.from_numpy(object_params['obj_trans']).float().to(device).reshape(-1, 3)
    obj_rot_mat = torch.from_numpy(object_params['obj_rot_mat']).float().to(device).reshape(-1, 3, 3)
    
    if obj_name not in obj_rest_data:
        print(f"Warning: Object {obj_name} not found in rest geometry data!")
        return None, None
        
    obj_rest_verts = obj_rest_data[obj_name]['vertices'].to(device)
    faces = obj_rest_data[obj_name]['faces']
    
    print(f"Restoring object mesh: {obj_name}, {obj_trans.shape[0]} frames")
    
    # Apply transformation to get object vertices for each frame
    vertices = load_object_geometry_w_rest_geo(obj_rot_mat, obj_trans, obj_rest_verts)

    vertices = vertices[:, :, [0, 2, 1]]  # swap y and z coordinates
    vertices[:, :, 1] = -vertices[:, :, 1]  # negate y coordinate
    
    # return yup_to_zup(vertices.detach().cpu().numpy()), faces # ours
    return vertices.detach().cpu().numpy(), faces # lingo


def save_mesh_sequence(vertices, faces, output_path, format='ply'):
    """
    Save mesh sequence to files.
    
    Args:
        vertices: [T, V, 3] - mesh vertices for each frame
        faces: [F, 3] - mesh face connectivity
        output_path: str - output file path prefix
        format: str - output format ('obj', 'ply')
    """
    T = vertices.shape[0]
    
    for t in range(T):
        if format.lower() == 'obj':
            filepath = f"{output_path}_frame_{t:04d}.obj"
            mesh = trimesh.Trimesh(vertices=vertices[t], faces=faces)
            mesh.export(filepath)
        elif format.lower() == 'ply':
            filepath = f"{output_path}_frame_{t:05d}.ply"
            mesh = trimesh.Trimesh(vertices=vertices[t], faces=faces)
            mesh.export(filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    print(f"Saved {T} frames to {output_path}_frame_*.{format}")


def save_verts_faces_to_mesh_files(mesh_verts, mesh_faces, obj_verts, obj_faces, save_mesh_folder):
    """
    Save human and object mesh sequence to PLY files for Blender rendering.
    
    Args:
        mesh_verts: [T, V, 3] - human mesh vertices
        mesh_faces: [F, 3] - human mesh faces
        obj_verts: [T, V, 3] - object mesh vertices  
        obj_faces: [F, 3] - object mesh faces
        save_mesh_folder: str - output folder path
    """
    if not os.path.exists(save_mesh_folder):
        os.makedirs(save_mesh_folder)

    num_meshes = mesh_verts.shape[0]
    for idx in range(num_meshes):
        # Save human mesh
        human_mesh = trimesh.Trimesh(vertices=mesh_verts[idx], faces=mesh_faces)
        human_mesh_path = os.path.join(save_mesh_folder, f"{idx:05d}.ply")
        human_mesh.export(human_mesh_path)
        
        # Save object mesh
        if obj_verts is not None and obj_faces is not None:
            obj_mesh = trimesh.Trimesh(vertices=obj_verts[idx], faces=obj_faces)
            obj_mesh_path = os.path.join(save_mesh_folder, f"{idx:05d}_object.ply")
            obj_mesh.export(obj_mesh_path)


def images_to_video_w_imageio(img_folder, output_vid_file, fps=30):
    """
    Convert images to video using imageio.
    
    Args:
        img_folder: str - folder containing images
        output_vid_file: str - output video file path
        fps: int - frames per second
    """
    img_files = [f for f in os.listdir(img_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not img_files:
        print(f"No image files found in {img_folder}")
        return
        
    img_files.sort()
    im_arr = []
    
    for img_name in img_files:
        img_path = os.path.join(img_folder, img_name)
        try:
            im = imageio.imread(img_path)
            im_arr.append(im)
        except Exception as e:
            print(f"Error reading image {img_path}: {e}")
    
    if im_arr:
        im_arr = np.asarray(im_arr)
        imageio.mimwrite(output_vid_file, im_arr, fps=fps, quality=8)
        print(f"Video saved to {output_vid_file}")
    else:
        print("No valid images found to create video")


def cleanup_intermediate_files(output_dir, seq_name, keep_intermediate=False):
    """
    Clean up intermediate PLY and PNG files, keeping only video files.
    
    Args:
        output_dir: str - base output directory
        seq_name: str - sequence name for file patterns
        keep_intermediate: bool - if True, keep all intermediate files
    """
    if keep_intermediate:
        print("Keeping all intermediate files as requested")
        return
        
    # List of directories and file patterns to clean up
    cleanup_patterns = [
        f"{seq_name}_human_frame_*.ply",
        f"{seq_name}_joints_frame_*.ply", 
        f"{seq_name}_object_frame_*.ply",
        f"{seq_name}_meshes/",
        f"{seq_name}_rendered/",
        f"{seq_name}_simple_video_temp_imgs/"
    ]
    
    files_removed = 0
    dirs_removed = 0
    
    for pattern in cleanup_patterns:
        full_pattern = os.path.join(output_dir, pattern)
        
        if pattern.endswith('/'):
            # Directory cleanup
            dir_path = full_pattern.rstrip('/')
            if os.path.exists(dir_path):
                try:
                    shutil.rmtree(dir_path)
                    dirs_removed += 1
                    print(f"Removed directory: {os.path.basename(dir_path)}")
                except Exception as e:
                    print(f"Error removing directory {dir_path}: {e}")
        else:
            # File pattern cleanup
            matching_files = glob.glob(full_pattern)
            for file_path in matching_files:
                try:
                    os.remove(file_path)
                    files_removed += 1
                except Exception as e:
                    print(f"Error removing file {file_path}: {e}")
    
    print(f"Cleanup completed: removed {files_removed} files and {dirs_removed} directories")


def run_blender_rendering(obj_folder_path, out_folder_path, out_vid_path, 
                         scene_blend_path=DEFAULT_SCENE_BLEND, 
                         vis_object=True, vis_human=True, fps=30):
    """
    Run Blender rendering and save to video.
    
    Args:
        obj_folder_path: str - folder containing mesh files
        out_folder_path: str - output folder for rendered images
        out_vid_path: str - output video file path
        scene_blend_path: str - path to Blender scene file
        vis_object: bool - whether to visualize object
        vis_human: bool - whether to visualize human
        fps: int - video frame rate
    """
    # Create output directories
    os.makedirs(out_folder_path, exist_ok=True)
    os.makedirs(os.path.dirname(out_vid_path), exist_ok=True)
    
    # Check if Blender utilities exist
    if vis_object and vis_human:
        blender_utils_path = os.path.join(BLENDER_UTILS_ROOT_FOLDER, "blender_vis_utils.py")
    elif vis_human:
        blender_utils_path = os.path.join(BLENDER_UTILS_ROOT_FOLDER, "blender_vis_human_utils.py")
    else:
        print("At least human or object visualization must be enabled")
        return False
    
    if not os.path.exists(blender_utils_path):
        print(f"Blender utils not found: {blender_utils_path}")
        return False
    
    if not os.path.exists(scene_blend_path):
        print(f"Blender scene file not found: {scene_blend_path}")
        return False
    
    # Run Blender rendering
    try:
        blender_command = f'{BLENDER_PATH} -P {blender_utils_path} -b -- --folder {obj_folder_path} --scene {scene_blend_path} --out-folder {out_folder_path}'
        print(f"Running Blender command: {blender_command}")
        result = subprocess.call(blender_command, shell=True)
        
        if result != 0:
            print(f"Blender command failed with return code {result}")
            return False
        
        # Convert rendered images to video
        images_to_video_w_imageio(out_folder_path, out_vid_path, fps=fps)
        return True
        
    except Exception as e:
        print(f"Error running Blender rendering: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Restore and visualize meshes from motion parameters")
    parser.add_argument('--param_file', type=str, required=True,
                       help='Path to motion parameters pickle file')
    parser.add_argument('--output_dir', type=str, default='./restored_meshes',
                       help='Output directory for mesh files and video')
    parser.add_argument('--format', type=str, default='ply', choices=['obj', 'ply'],
                       help='Output mesh format')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--rest_verts_root', type=str, 
                       default="/cpfs04/shared/sport/zouyude/code/chois_release/processed_data/rest_object_geo",
                       help='Path to object rest geometry directory')
    parser.add_argument('--fps', type=int, default=30,
                       help='Video frame rate')
    parser.add_argument('--render_video', action='store_true',
                       help='Render video using Blender')
    parser.add_argument('--blender_scene', type=str, default=DEFAULT_SCENE_BLEND,
                       help='Path to Blender scene file')
    parser.add_argument('--vis_human', action='store_true', default=True,
                       help='Visualize human')
    parser.add_argument('--vis_object', action='store_true', default=True,
                       help='Visualize object')
    parser.add_argument('--cleanup', action='store_true', default=True,
                       help='Clean up intermediate files (PLY, PNG) after video creation')
    parser.add_argument('--keep_intermediate', action='store_true', default=False,
                       help='Keep intermediate files (overrides --cleanup)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load motion parameters
    print(f"Loading motion parameters from {args.param_file}")
    try:
        with open(args.param_file, 'rb') as f:
            motion_params = pickle.load(f)
    except Exception as e:
        print(f"Error loading parameters: {e}")
        return
    
    seq_name = motion_params['seq_name']
    human_params = motion_params['human_motion']
    object_params = motion_params['object_motion']
    
    print(f"Restoring sequence: {seq_name}")
    
    human_verts = None
    human_faces = None
    object_verts = None
    object_faces = None
    
    # Restore human mesh
    if args.vis_human:
        print("\n=== Restoring Human Mesh ===")
        human_verts, human_joints, human_faces = restore_human_mesh(human_params, device)
        print(f"Human mesh restored: {human_verts.shape[0]} frames, {human_verts.shape[1]} vertices")
        
        # Save human mesh sequence
        human_output_path = os.path.join(args.output_dir, f"{seq_name}_human")
        save_mesh_sequence(human_verts, human_faces, human_output_path, args.format)
        
        # Also save joints as point clouds
        joints_output_path = os.path.join(args.output_dir, f"{seq_name}_joints")
        for t in range(human_joints.shape[0]):
            joints_mesh = trimesh.points.PointCloud(vertices=human_joints[t])
            joints_mesh.export(f"{joints_output_path}_frame_{t:05d}.ply")
            
    # Restore object mesh
    if args.vis_object:
        print("\n=== Restoring Object Mesh ===")
        try:
            obj_rest_data = load_object_rest_geometry(args.rest_verts_root)
            
            if obj_rest_data:
                object_verts, object_faces = restore_object_mesh(object_params, obj_rest_data, device)
                
                if object_verts is not None:
                    print(f"Object mesh restored: {object_verts.shape[0]} frames, {object_verts.shape[1]} vertices")
                    
                    # Save object mesh sequence
                    object_output_path = os.path.join(args.output_dir, f"{seq_name}_object")
                    save_mesh_sequence(object_verts, object_faces, object_output_path, args.format)
                else:
                    print("Failed to restore object mesh")
            else:
                print("No object geometry data found")
                args.vis_object = False
                
        except Exception as e:
            print(f"Error restoring object mesh: {e}")
            args.vis_object = False
    
    # Create video visualization
    if args.render_video and (args.vis_human or args.vis_object):
        print("\n=== Creating Video Visualization ===")
        
        # Try Blender rendering first
        blender_success = False
        
        if human_verts is not None:
            try:
                # Create mesh folder for Blender rendering
                mesh_folder = os.path.join(args.output_dir, f"{seq_name}_meshes")
                rendered_images_folder = os.path.join(args.output_dir, f"{seq_name}_rendered")
                video_output_path = os.path.join(args.output_dir, f"{seq_name}_video.mp4")
                
                # Save meshes in format compatible with Blender scripts
                if args.vis_human and args.vis_object and human_verts is not None and object_verts is not None:
                    save_verts_faces_to_mesh_files(human_verts, human_faces, object_verts, object_faces, mesh_folder)
                elif args.vis_human and human_verts is not None:
                    # Save only human meshes
                    save_verts_faces_to_mesh_files(human_verts, human_faces, None, None, mesh_folder)
                
                # Run Blender rendering
                blender_success = run_blender_rendering(
                    obj_folder_path=mesh_folder,
                    out_folder_path=rendered_images_folder,
                    out_vid_path=video_output_path,
                    scene_blend_path=args.blender_scene,
                    vis_object=args.vis_object,
                    vis_human=args.vis_human,
                    fps=args.fps
                )
                
                if blender_success:
                    print(f"Blender video created at: {video_output_path}")
                    
            except Exception as e:
                print(f"Error with Blender rendering: {e}")
    
    print(f"\nMesh restoration completed! Output saved to {args.output_dir}")
    
    # Cleanup intermediate files if requested
    should_cleanup = args.cleanup and not args.keep_intermediate
    if should_cleanup:
        print("\n=== Cleaning Up Intermediate Files ===")
        cleanup_intermediate_files(args.output_dir, seq_name, keep_intermediate=args.keep_intermediate)
        print("Note: Use --keep_intermediate to preserve PLY and PNG files")
    elif args.keep_intermediate:
        print("Intermediate files preserved as requested")


if __name__ == "__main__":
    main()