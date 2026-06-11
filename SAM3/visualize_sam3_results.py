## This script visualizes the SAM3 segmentation results by overlaying the predicted masks on the original images from the NuScenes dataset

## It creates both individual frame visualizations and a video output for each prompt
## The script also supports a combined visualization mode where multiple prompts are overlaid with priority-based resolution to handle overlapping masks
## For single prompt visualization, it ensures consistent coloring of the same object across frames by building a mapping of SAM3 IDs to new consistent IDs based on their temporal presence
## The script is designed to be flexible, allowing users to specify a frame range for visualization and handles cases where results may be missing for certain frames or prompts gracefully
#!/usr/bin/env python3
"""
Visualize SAM3 segmentation results by overlaying masks on original frames
and creating video/image outputs.
"""

import os
import numpy as np
from PIL import Image, ImageDraw
import cv2
from pathlib import Path
from nuscenes.nuscenes import NuScenes
import argparse
from tqdm import tqdm



# Configuration
data_root = '/media/lleba/ECP_Nuscenes_01/output2/ecp2nuscenes'
scene_token_filter = 'scene-euro-citystrasbourg-scenariolatesession-00002_1'
results_root = '/media/lleba/ECP_Nuscenes_01/SAM3_Visualizations/Strassbourg'
output_viz_root = '/media/lleba/ECP_Nuscenes_01/SAM3_Visualizations/Strassbourg'

# Colors for different prompts (BGR format for OpenCV)
COLORS = {
    'car': (0, 165, 255),         # Orange
    'pedestrian': (0, 0, 255),    # Red
    'bicycle and cyclist': (238, 130, 238),   # Violet
    'motorcycle': (0, 255, 0),    # Green for motorcycle
}

# Utility function for robust color lookup

def get_prompt_color(prompt):
    # Try full prompt, then first word, then default grey
    if prompt in COLORS:
        return COLORS[prompt]
    key = prompt.split()[0]
    if key in COLORS:
        return COLORS[key]
    return (128, 128, 128)

# ===== USER-DEFINED FRAME RANGE =====
# Set these to restrict visualization to a specific frame range
START_FRAME = 500
END_FRAME = 800
# ====================================

def load_scene_images(nusc, scene_idx, cam_idx=0):
    """Load all images for a specific camera in a scene using NuScenes API."""
    scene = nusc.scene[scene_idx]
    cam_names = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']
    cam_name = cam_names[cam_idx]
    
    images = []
    first_sample = nusc.get('sample', scene['first_sample_token'])
    current_token = first_sample['data'][cam_name]
    
    while current_token != '':
        img_path = nusc.get_sample_data_path(current_token)
        image = np.array(Image.open(img_path))
        images.append(image)
        
        rec = nusc.get('sample_data', current_token)
        current_token = rec['next']
    
    return images

def overlay_mask_on_image(image, mask, label, color, alpha=0.4):
    """
    Overlay a segmentation mask on an image.
    
    Args:
        image: numpy array (H, W, 3) RGB
        mask: numpy array (H, W) binary mask
        label: string label for the object
        color: tuple (B, G, R) for OpenCV
        alpha: transparency factor
    
    Returns:
        image_with_mask: numpy array (H, W, 3) with overlay
    """
    image_copy = image.copy()
    
    # Convert to BGR if needed
    if isinstance(image_copy, np.ndarray):
        image_copy_bgr = cv2.cvtColor(image_copy, cv2.COLOR_RGB2BGR)
    
    # Create a colored mask
    colored_mask = np.zeros_like(image_copy_bgr)
    colored_mask[mask > 0] = color
    
    # Blend mask with image
    image_with_mask = cv2.addWeighted(image_copy_bgr, 1 - alpha, colored_mask, alpha, 0)
    
    # Add label text
    cv2.putText(image_with_mask, label, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    return image_with_mask

def visualize_single_prompt(scene_idx, cam_idx, prompt, nusc):
    """
    Create visualization for a single prompt across all frames.
    """
    # Load results
    results_file = os.path.join(results_root, f'scene_{scene_idx}_cam_{cam_idx}_sam3_outputs__{prompt}.npy')
    
    if not os.path.exists(results_file):
        print(f"Results file not found: {results_file}")
        return
    
    print(f"Loading results for prompt: {prompt}")
    outputs = np.load(results_file, allow_pickle=True).item()

    mapping = build_consistent_id_mapping(outputs)
    
    # Load images
    print(f"Loading images...")
    loader_module_path = '/home/lleba/Thesis/Development/get_sam3_masks_for_nuscenes_trainval__bus.py'
    # We'll reload using the NuScenes API directly
    scene = nusc.scene[scene_idx]
    
    # Get first sample
    first_sample = nusc.get('sample', scene['first_sample_token'])
    cam_names = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']
    cam_name = cam_names[cam_idx]
    
    # Create output directory
    output_dir = os.path.join(output_viz_root, f'scene_{scene_idx}_cam_{cam_idx}_{prompt}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Process and save frames
    current_token = first_sample['data'][cam_name]
    frame_idx = 0
    color = get_prompt_color(prompt)
    print(f"Processing frames...")
    frames_for_video = []
    # Only process frames in the user-defined range
    while current_token != '':
        if START_FRAME is not None and frame_idx < START_FRAME:
            rec = nusc.get('sample_data', current_token)
            current_token = rec['next']
            frame_idx += 1
            continue
        if END_FRAME is not None and frame_idx > END_FRAME:
            break
        # Load image
        img_path = nusc.get_sample_data_path(current_token)
        image = np.array(Image.open(img_path))
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Get mask for this frame
        if frame_idx in outputs:
            # mask_data = outputs[frame_idx]['out_binary_masks']  # Shape: (num_objects, height, width)
            
            # # Combine all masks for this frame (if multiple instances)
            # if isinstance(mask_data, np.ndarray) and len(mask_data.shape) == 3:
            #     # Multiple masks per frame
            #     combined_mask = np.zeros(mask_data.shape[1:], dtype=bool)
            #     for m in mask_data:
            #         combined_mask |= m.astype(bool)

            ids = outputs[frame_idx]['out_obj_ids']
            masks = outputs[frame_idx]['out_binary_masks']

            image_with_mask = image_bgr.copy()

            for i, sam3_id in enumerate(ids):

                if (frame_idx, sam3_id) not in mapping:
                    continue

                new_id = mapping[(frame_idx, sam3_id)]
                color = get_color_for_id(new_id)

                mask = masks[i].astype(bool)

                colored_mask = np.zeros_like(image_with_mask)
                colored_mask[mask] = color

                image_with_mask = cv2.addWeighted(image_with_mask, 1.0, colored_mask, 0.4, 0)

                # draw ID
                y = 30 + i * 20
                cv2.putText(image_with_mask, f"ID {new_id}", (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # elif isinstance(mask_data, np.ndarray) and len(mask_data.shape) == 2:
            #     # Single mask
            #     combined_mask = mask_data.astype(bool)
            # else:
            #     combined_mask = np.zeros(image.shape[:2], dtype=bool)
            
            # # Overlay mask
            # image_with_mask = overlay_mask_on_image(image, combined_mask, prompt, color)
        else:
            image_with_mask = image
        
        # Save individual frame
        output_frame_path = os.path.join(output_dir, f'frame_{frame_idx:05d}.jpg')
        cv2.imwrite(output_frame_path, image_with_mask)
        frames_for_video.append(cv2.cvtColor(image_with_mask, cv2.COLOR_BGR2RGB))
        
        if (frame_idx + 1) % 100 == 0:
            print(f"  Processed {frame_idx + 1} frames")
        
        # Get next frame
        rec = nusc.get('sample_data', current_token)
        current_token = rec['next']
        frame_idx += 1
    
    # Create video
    print(f"Creating video...")
    video_path = os.path.join(output_dir, f'segmentation_{prompt}.mp4')
    height, width = frames_for_video[0].shape[:2]
    fps = 20
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    
    for frame in frames_for_video:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    out.release()
    print(f"Video saved: {video_path}")
    print(f"Individual frames saved to: {output_dir}")


def visualize_combined_prompts(scene_idx, cam_idx, prompts_priority, nusc, start_frame=None, end_frame=None):
    """
    Create combined visualization for multiple prompts with priority-based overlap resolution.
    Higher priority prompts override lower ones.
    """
    # Load all results
    all_outputs = {}
    for prompt in prompts_priority:
        results_file = os.path.join(results_root, f'scene_{scene_idx}_cam_{cam_idx}_sam3_outputs__{prompt}.npy')
        if os.path.exists(results_file):
            outputs = np.load(results_file, allow_pickle=True).item()
            all_outputs[prompt] = outputs
        else:
            all_outputs[prompt] = {}  # Ensure key exists, even if no file
    
    if not all_outputs:
        print("No results found for any prompts")
        return
    
    # Get scene info
    scene = nusc.scene[scene_idx]
    first_sample = nusc.get('sample', scene['first_sample_token'])
    cam_names = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']
    cam_name = cam_names[cam_idx]
    
    # Create output directory
    output_dir = os.path.join(output_viz_root, f'scene_{scene_idx}_cam_{cam_idx}_combined')
    os.makedirs(output_dir, exist_ok=True)
    # Initialize processed_frame_indices before the loop
    processed_frame_indices = []
    # Process frames (original loop, now restricted by frame range)
    current_token = first_sample['data'][cam_name]
    frame_idx = 0
    frames_for_video = []
    print("Processing combined frames with priority resolution...")
    while current_token != '':
        if start_frame is not None and frame_idx < start_frame:
            rec = nusc.get('sample_data', current_token)
            current_token = rec['next']
            frame_idx += 1
            continue
        if end_frame is not None and frame_idx > end_frame:
            break
        # Load image
        img_path = nusc.get_sample_data_path(current_token)
        image = np.array(Image.open(img_path))
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Create combined mask overlay
        overlay = np.zeros_like(image_bgr)
        labels_present = []
        
        for prompt in prompts_priority:
            if prompt in all_outputs and frame_idx in all_outputs[prompt]:
                mask_data = all_outputs[prompt][frame_idx]['out_binary_masks']
                
                if isinstance(mask_data, np.ndarray) and len(mask_data.shape) == 3:
                    # Multiple masks per frame
                    combined_mask = np.zeros(mask_data.shape[1:], dtype=bool)
                    for m in mask_data:
                        combined_mask |= m.astype(bool)
                elif isinstance(mask_data, np.ndarray) and len(mask_data.shape) == 2:
                    # Single mask
                    combined_mask = mask_data.astype(bool)
                else:
                    continue
                
                # Apply to overlay where not already set (priority: higher priority first)
                color = get_prompt_color(prompt)
                mask_pixels = combined_mask > 0
                unset_pixels = np.all(overlay == 0, axis=2)
                apply_mask = mask_pixels & unset_pixels
                overlay[apply_mask] = color
                
                if np.any(apply_mask):
                    labels_present.append(prompt)
        
        # Blend overlay with image
        if np.any(overlay > 0):
            image_with_mask = cv2.addWeighted(image_bgr, 0.6, overlay, 0.4, 0)
            
            # Add labels for present classes
            y_offset = 30
            for label in labels_present:
                color = get_prompt_color(label)
                cv2.putText(image_with_mask, label, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                y_offset += 30
        else:
            image_with_mask = image_bgr
        
        # Save frame
        output_frame_path = os.path.join(output_dir, f'frame_{frame_idx:05d}.jpg')
        cv2.imwrite(output_frame_path, image_with_mask)
        frames_for_video.append(cv2.cvtColor(image_with_mask, cv2.COLOR_BGR2RGB))
        
        if (frame_idx + 1) % 100 == 0:
            print(f"  Processed {frame_idx + 1} frames")
        
        # Save frame index for no-mask video
        processed_frame_indices.append(frame_idx)
        rec = nusc.get('sample_data', current_token)
        current_token = rec['next']
        frame_idx += 1
    
    # Create video
    print("Creating combined video...")
    video_path = os.path.join(output_dir, 'segmentation_combined.mp4')
    height, width = frames_for_video[0].shape[:2]
    fps = 20
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    for frame in frames_for_video:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()
    print(f"Combined video saved: {video_path}")
    print(f"Individual frames saved to: {output_dir}")

def build_consistent_id_mapping(outputs):
    id_presence = {}

    # collect frames per SAM3 id
    for frame_idx in outputs:
        ids = outputs[frame_idx]['out_obj_ids']
        for obj_id in ids:
            id_presence.setdefault(obj_id, []).append(frame_idx)

    def split_into_segments(frame_list):
        segments = []
        current = [frame_list[0]]

        for i in range(1, len(frame_list)):
            if frame_list[i] == frame_list[i-1] + 1:
                current.append(frame_list[i])
            else:
                segments.append(current)
                current = [frame_list[i]]
        segments.append(current)
        return segments

    mapping = {}
    new_id_counter = 0

    for obj_id in sorted(id_presence.keys()):
        frames_seen = sorted(id_presence[obj_id])
        segments = split_into_segments(frames_seen)

        for seg in segments:
            for f in seg:
                mapping[(f, obj_id)] = new_id_counter
            new_id_counter += 1

    print(f"Total consistent IDs: {new_id_counter}")
    return mapping

def get_color_for_id(idx):
    np.random.seed(idx)
    return tuple(np.random.randint(0, 255, 3).tolist())


def create_video_without_masks(nusc, scene_idx, cam_idx, frame_indices, output_dir):
    cam_names = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']
    cam_name = cam_names[cam_idx]
    scene = nusc.scene[scene_idx]
    first_sample = nusc.get('sample', scene['first_sample_token'])
    frames_for_video = []
    for frame_idx in frame_indices:
        current_token = first_sample['data'][cam_name]
        for _ in range(frame_idx):
            rec = nusc.get('sample_data', current_token)
            current_token = rec['next']
        img_path = nusc.get_sample_data_path(current_token)
        image = np.array(Image.open(img_path))
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        output_frame_path = os.path.join(output_dir, f'frame_{frame_idx:05d}_nomask.jpg')
        cv2.imwrite(output_frame_path, image_bgr)
        frames_for_video.append(image_bgr)
    if frames_for_video:
        height, width = frames_for_video[0].shape[:2]
        fps = 20
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_path = os.path.join(output_dir, 'video_nomask.mp4')
        out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        for frame in frames_for_video:
            out.write(frame)
        out.release()
        print(f"Video without masks saved: {video_path}")
    else:
        print("No frames for video without masks.")


def create_original_video_from_range(nusc, scene_idx, cam_idx, start_frame, end_frame, output_dir):
    cam_names = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']
    cam_name = cam_names[cam_idx]
    scene = nusc.scene[scene_idx]
    first_sample = nusc.get('sample', scene['first_sample_token'])
    frames_for_video = []
    for frame_idx in range(start_frame, end_frame + 1):
        current_token = first_sample['data'][cam_name]
        for _ in range(frame_idx):
            rec = nusc.get('sample_data', current_token)
            current_token = rec['next']
        img_path = nusc.get_sample_data_path(current_token)
        image = np.array(Image.open(img_path))
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        output_frame_path = os.path.join(output_dir, f'frame_{frame_idx:05d}_original.jpg')
        cv2.imwrite(output_frame_path, image_bgr)
        frames_for_video.append(image_bgr)
    if frames_for_video:
        height, width = frames_for_video[0].shape[:2]
        fps = 20
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_path = os.path.join(output_dir, 'video_original.mp4')
        out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        for frame in frames_for_video:
            out.write(frame)
        out.release()
        print(f"Original video (no masks) saved: {video_path}")
    else:
        print("No frames for original video.")


def main():
    # Initialize NuScenes
    nusc = NuScenes(version='v1.0-trainval', dataroot=data_root, verbose=False)
    
    # Find scene index
    scene_idx = None
    for idx, scene in enumerate(nusc.scene):
        if scene['name'] == scene_token_filter:
            scene_idx = idx
            break
    
    if scene_idx is None:
        print(f"Scene not found: {scene_token_filter}")
        return
    
    print(f"Found scene at index: {scene_idx}")
    
    # Combined visualization with priority (highest first)
    prompts_priority = ['bicycle and cyclist', 'motorcycle', 'pedestrian', 'car']
    
    # visualize_combined_prompts(scene_idx, 0, prompts_priority, nusc, START_FRAME, END_FRAME)
    visualize_single_prompt(scene_idx, 0, 'car', nusc)
    # Also create a video from original frames in the same range (no masks)
    # create_original_video_from_range(nusc, scene_idx, 0, START_FRAME, END_FRAME, output_viz_root)
    
    print(f"\nVisualization complete!")
    print(f"Output saved to: {output_viz_root}")


if __name__ == '__main__':
    main()
