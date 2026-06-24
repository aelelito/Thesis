# %%


## Script is able to create SAM3 segmentation masks for single prompts
## script needs to be run for each prompt separately
## to reduce GPU memory, frames are processed in chunks (configurable in the code) and saved to disk immediately after processing
## outputs are appended to the same file for each prompt, so you can have all frames in one file per prompt at the end

import os


os.environ['CUDA_DEVICE_ORDER']       = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']    = '0'                                             # GPU index (use 0 for first/available GPU).
os.environ['MKL_NUM_THREADS']         = '8'                                             # Num of threads.
os.environ['NUMEXPR_NUM_THREADS']     = '8'                                             # Num of threads.
os.environ['OMP_NUM_THREADS']         = '8'                                             # Num of threads.

# %%
results_root = '/media/lleba/ECP_Nuscenes_01/SAM3_Visualizations/Strassbourg'
data_root    = '/media/lleba/ECP_Nuscenes_01/output2/ecp2nuscenes'
scene_token_filter = 'scene-euro-citystrasbourg-scenariolatesession-00002_1'  # Only process this scene

# Create results directory if it doesn't exist
os.makedirs(results_root, exist_ok=True)

assert results_root != 'PUT_YOUR_DIRECTORY_HERE', print('Folder for storing UNION results. Change to directory in your file system!')
assert data_root    != 'PUT_YOUR_DIRECTORY_HERE', print('Directory to nuScenes dataset. Change to directory in your file system!')


# %%
import numpy as np
from PIL import Image
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix


class RawSequenceLoader:
    """
    A lightweight wrapper around a nuScenes sequence.
    Stores file paths and pre-computed transforms to the sequence Base Frame.
    Lazy-loads images and point clouds to save RAM.
    """
    def __init__(
            self,
            nusc: NuScenes,
            scene_idx: int,
            lidar_period: float = 0.05   # Default 20 Hz for nuScenes.
            ):
        """
        Initializes the RawSequenceLoader.

        Args:
            nusc (NuScenes): The NuScenes dataset object.
            scene_idx (int): Index of the scene to load.
            lidar_period (float): Time interval between LiDAR frames. Default is 0.05s (20Hz).
        """
        self.nusc = nusc
        self.scene = nusc.scene[scene_idx]
        self.cam_names = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']
        self.lidar_name = 'LIDAR_TOP'
        self.lidar_period = lidar_period
        
        # Core data structure (lightweight metadata).
        self.data = {
            'image_paths': {i: [] for i in range(len(self.cam_names))}, 
            'lidar_paths': {0: []},
            'T_lidar_cam': {i: None for i in range(len(self.cam_names))},   # Shapes (4x4).
            'T_base_cam': {i: [] for i in range(len(self.cam_names))},   # Shapes (4x4).
            'T_base_lidar': {0: []},   # Shapes (4x4).
            'K_intrinsics': {i: None for i in range(len(self.cam_names))},   # Shapes (3x3).
            'cam_timestamps': {i: [] for i in range(len(self.cam_names))},   # Unit: seconds.
            'lidar_timestamps': [],   # Unit: seconds.
            'time_offsets': {i: 0.0 for i in range(len(self.cam_names))},   # Unit: seconds.
            'corrected_cam_timestamps': {i: [] for i in range(len(self.cam_names))},   # cam_timestamp + time_offset.
        }
        
        self._init_base_frame()

        self._extract_sensor_stream(self.lidar_name, is_lidar=True)
        for i, cam_name in enumerate(self.cam_names):
            self._extract_sensor_stream(cam_name, is_lidar=False, cam_idx=i)

        self._fill_time_offsets()
        self._correct_cam_timestamps()

    def _init_base_frame(
            self,
            ):
        """
        Defines the T_base_world from the very first LiDAR timestamp of the scene.
        """
        first_sample = self.nusc.get('sample', self.scene['first_sample_token'])
        first_lidar_token = first_sample['data'][self.lidar_name]
        first_lidar = self.nusc.get('sample_data', first_lidar_token)
        first_ego = self.nusc.get('ego_pose', first_lidar['ego_pose_token'])
        
        T_world_base = transform_matrix(first_ego['translation'], Quaternion(first_ego['rotation']), inverse=False)
        self.T_base_world = np.linalg.inv(T_world_base)

    def _extract_sensor_stream(
            self, sensor_name,
            is_lidar=False,
            cam_idx=None,
            ):
        """
        Traverses the linked list of sample_data to get EVERY frame between scene start and scene end.

        Args:
            sensor_name (str): Sensor name, e.g. 'LIDAR_TOP' or 'CAM_FRONT'.
            is_lidar (bool): Whether this stream is LiDAR or Camera (for storing in correct dict keys).
            cam_idx (int): If camera, the index for storing in dict keys (0-5 for 6 cameras).
        """
        # Step 1: Identify Start and End tokens for this specific sensor in this scene.
        first_sample = self.nusc.get('sample', self.scene['first_sample_token'])
        last_sample = self.nusc.get('sample', self.scene['last_sample_token'])
        
        # Check if this sensor exists in the scene
        if sensor_name not in first_sample['data'] or first_sample['data'][sensor_name] == '':
            print(f"  Warning: {sensor_name} not found in scene {self.scene['name']}")
            return
        
        current_token = first_sample['data'][sensor_name]
        end_token = last_sample['data'][sensor_name]
        
        # Step 2: Get static extrinsics (Sensor -> Vehicle).
        # We grab this from the first frame, assuming rigid mounting for the scene duration.
        sd = self.nusc.get('sample_data', current_token)
        cs = self.nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
        T_vehicle_sensor = transform_matrix(cs['translation'], Quaternion(cs['rotation']))
        
        if not is_lidar:
            self.data['K_intrinsics'][cam_idx] = np.array(cs['camera_intrinsic'])

        # Step 3: Traverse the linked list.
        while True:
            # Load metadata.
            rec = self.nusc.get('sample_data', current_token)
            ego = self.nusc.get('ego_pose', rec['ego_pose_token'])
            
            # Compute full pose: Base <- World <- Vehicle(t) <- Sensor
            T_world_vt = transform_matrix(ego['translation'], Quaternion(ego['rotation']))
            T_base_sensor = self.T_base_world @ T_world_vt @ T_vehicle_sensor
            
            # Store data.
            if is_lidar:
                self.data['lidar_paths'][0].append(self.nusc.get_sample_data_path(current_token))
                self.data['T_base_lidar'][0].append(T_base_sensor)
                self.data['lidar_timestamps'].append(rec['timestamp'] / 1e6)   # Convert to seconds.
            else:
                self.data['image_paths'][cam_idx].append(self.nusc.get_sample_data_path(current_token))
                self.data['T_base_cam'][cam_idx].append(T_base_sensor)
                self.data['cam_timestamps'][cam_idx].append(rec['timestamp'] / 1e6)   # Convert to seconds.

            # Check termination.
            if current_token == end_token or rec['next'] == '':
                break
            current_token = rec['next']

    def _fill_time_offsets(
            self,
            ):
        """
        Hardcoded time offsets for each camera.
        LiDAR timestamp is at the END of scan (Back cam / -x direction).
        Cameras trigger at different points in scan based on their position.

        The firing order is Clockwise:
        - Start of Sweep (~ -0.050s)
        - CAM_FRONT_LEFT (~ -0.043s) -> First to fire.
        - CAM_FRONT (~ -0.036s)
        - CAM_FRONT_RIGHT (~ -0.028s)
        - CAM_BACK_RIGHT (~ -0.020s)
        - CAM_BACK (~ -0.011s)
        - CAM_BACK_LEFT (~ -0.001s) -> Last to fire
        - End of Sweep (~ 0.000s)
        """
        offsets = {
            'CAM_FRONT': -0.03583,
            'CAM_FRONT_RIGHT': -0.02789,
            'CAM_BACK_RIGHT': -0.02031,
            'CAM_BACK': -0.01072,
            'CAM_BACK_LEFT': -0.00084,
            'CAM_FRONT_LEFT': -0.04345,
        }
        for i, cam_name in enumerate(self.cam_names):
             self.data['time_offsets'][i] = offsets[cam_name]

    def _correct_cam_timestamps(
            self,
            ):
        """
        Applies the time offsets to the camera timestamps to get corrected timestamps.
        """
        for i in range(len(self.cam_names)):
            offset = self.data['time_offsets'][i]
            self.data['corrected_cam_timestamps'][i] = [ts + offset for ts in self.data['cam_timestamps'][i]]
            
    def get_lidar(
            self,
            idx,
            ) -> np.ndarray:
        """
        Fast binary load for LiDAR.
        
        Args:
            idx (int) : Index of the frame to load (0-based).

        Returns:
            pc_lidar (np.ndarray) : Point cloud of shape (N, 3) for the requested frame.
        """
        path = self.data['lidar_paths'][0][idx]
        pc_lidar = np.fromfile(path, dtype=np.float32).reshape(-1, 5)[:,:3]
        return pc_lidar

    def get_image(
            self,
            cam_idx,
            idx,
            ) -> np.ndarray:
        """
        Fast load for camera images.

        Args:
            cam_idx (int) : Camera index (0-5).
            idx (int) : Frame index (0-based).

        Returns:
            image (np.ndarray) : Image array of shape (H, W, 3) for the requested frame.
        """
        path = self.data['image_paths'][cam_idx][idx]
        image = np.array(Image.open(path))
        return image

    def get_stats(
            self,
            ) -> dict:
        """
        Get statistics about the loaded data.

        Returns:
            stats (dict) : Dictionary containing statistics such as number of frames for LiDAR and each camera.
        """
        stats = {
            'Lidar Frames': len(self.data['lidar_paths'][0]),
            'Camera Frames': {cam_idx: len(paths) for cam_idx, paths in self.data['image_paths'].items()}
        }
        return stats
    
    def get_match(
            self,
            cam_name: str,
            tolerance: float = 0.025,
            ) -> list:
        matches = []
        len_lidar = len(t_lidar)
        for i, idx in enumerate(indices):
            # Case A: Insertion at index 0 (Target is before the first LiDAR frame).
            if idx == 0:
                best_idx = 0
            # Case B: Insertion at end (Target is after the last LiDAR frame).
            elif idx == len_lidar:
                best_idx = len_lidar - 1
            # Case C: Target is between idx-1 and idx. Check which is closer.
        """
        Finds the closest LiDAR frame index for every frame of the specified camera.

        Args:
            cam_name (str): Name of the camera (e.g. 'CAM_FRONT').
            tolerance (float): Maximum allowed time difference in seconds for a valid match.

        Returns:
            matches (list): A list of integer indices.
        """
        if cam_name not in self.cam_names:
            raise ValueError(f'Camera {cam_name} not found! Available: {self.cam_names}')
        cam_idx = self.cam_names.index(cam_name)
        
        t_cam = np.array(self.data['corrected_cam_timestamps'][cam_idx])
        t_lidar = np.array(self.data['lidar_timestamps'])
        
        indices = np.searchsorted(t_lidar, t_cam)
        
        matches = []
        len_lidar = len(t_lidar)
        for i, idx in enumerate(indices):
            # Case A: Insertion at index 0 p"](Target is before the first LiDAR frame).
            if idx == 0:
                best_idx = 0
            # Case B: Insertion at end (Target is after the last LiDAR frame).
            elif idx == len_lidar:
                best_idx = len_lidar - 1
            # Case C: Target is between idx-1 and idx. Check which is closer.
            else:
                dt_left = abs(t_cam[i] - t_lidar[idx - 1])
                dt_right = abs(t_cam[i] - t_lidar[idx])
                if dt_left < dt_right:
                    best_idx = idx - 1
                else:
                    best_idx = idx
            matches.append(int(best_idx))

        bool_valid = [abs(t_cam[i] - t_lidar[matches[i]]) <= tolerance for i in range(len(matches))]
        matches = [matches[i] if bool_valid[i] else None for i in range(len(matches))]
        return matches


nusc = NuScenes(version='v1.0-trainval', dataroot=data_root, verbose=False)   #'v1.0-mini'   # 'v1.0-trainval'


# %%
import os
import shutil
import numpy as np
import torch
from PIL import Image
import gc
# Ensure you are importing the builder correctly
from sam3.model_builder import build_sam3_video_predictor

# Set environment variable for better memory management
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# --- 1. INITIALIZE MODEL ONCE (OUTSIDE LOOPS) ---
print("Initializing SAM3 model... (this happens only once)")
from huggingface_hub import hf_hub_download
SAM3_CKPT = hf_hub_download(repo_id="facebook/sam3", filename="sam3.pt")
gpus = range(torch.cuda.device_count())
predictor = build_sam3_video_predictor(gpus_to_use=gpus, checkpoint_path=SAM3_CKPT)
print("Model initialized.")

# Optional: Set precision for A100 speedup
torch.set_float32_matmul_precision('high')

# ===== CONFIGURE PROMPTS HERE =====
# Set to process only specific prompts to save memory
TEXT_PROMPTS = ["car"]  # Change this for each run: "car", "pedestrian", "bicycle and cyclist", "motorcycle"
# ===================================

for scene_idx in range(len(nusc.scene)):
    # Only process the filtered scene
    if nusc.scene[scene_idx]['name'] != scene_token_filter:
        continue
    
    print(f"\n{'='*60}")
    print(f"Processing Scene: {scene_token_filter} (index {scene_idx})")
    print(f"{'='*60}")
    
    loader = RawSequenceLoader(nusc, scene_idx=scene_idx)
    
    # Only process CAM_FRONT (camera index 0)
    cam_idx = 0
    print(f"Processing Camera: {loader.cam_names[cam_idx]}...")

    # Config
    CAM_IDX = cam_idx
    TEMP_VIDEO_DIR = f'./temp_nuscenes_sam3_input_s{scene_idx}_c{cam_idx}'

    # Prep data (Save images to disk) - ONCE FOR ALL PROMPTS
    if os.path.exists(TEMP_VIDEO_DIR):
        shutil.rmtree(TEMP_VIDEO_DIR)
    os.makedirs(TEMP_VIDEO_DIR)
    
    # Get frames for this specific camera
    num_frames = len(loader.data['image_paths'][CAM_IDX])
    print(f"Number of frames: {num_frames}")
    
    # Set start frame for processing (change this to process different segments)
    start_frame = 500  # Start from frame 0
    
    # For testing/visualization, process only a chunk of frames to save memory
    # Reduce this number if OOM errors occur
    num_frames_to_process = min(num_frames - start_frame, 300)
    print(f"Processing {num_frames_to_process} frames starting from frame {start_frame} (total frames: {num_frames})")
    
    for i in range(start_frame, start_frame + num_frames_to_process):
        img_array = loader.get_image(cam_idx=CAM_IDX, idx=i)
        save_path = os.path.join(TEMP_VIDEO_DIR, f"{i:05d}.jpg")
        Image.fromarray(img_array).save(save_path)
        if (i + 1) % 50 == 0:
            print(f"  Saved {i + 1}/{num_frames_to_process} images")

    print(f"All images saved to {TEMP_VIDEO_DIR}")

    # --- 2. INFERENCE WITH MULTIPLE PROMPTS ---
    for prompt_idx, TEXT_PROMPT in enumerate(TEXT_PROMPTS):
        print(f"\n--- Processing prompt ({prompt_idx + 1}/{len(TEXT_PROMPTS)}): '{TEXT_PROMPT}' ---")
        
        # Start the session using the existing predictor
        response = predictor.handle_request(request=dict(
            type='start_session', 
            resource_path=TEMP_VIDEO_DIR
        ))
        session_id = response['session_id']
        
        # Add prompt
        _ = predictor.handle_request(request=dict(
            type='add_prompt', 
            session_id=session_id, 
            frame_index=0, 
            text=TEXT_PROMPT
        ))
        
        outputs_per_frame = {}
        
        # Run propagation with autocast for A100 speed
        print(f"Propagating through video for '{TEXT_PROMPT}'...")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for response in predictor.handle_stream_request(request=dict(
                type='propagate_in_video', 
                session_id=session_id
            )):
                outputs_per_frame[start_frame + response['frame_index']] = response['outputs']
                if (response['frame_index'] + 1) % 100 == 0:
                    print(f"  Processed frame {start_frame + response['frame_index'] + 1}")
            
        # Close the session
        predictor.handle_request(request=dict(
            type='close_session', 
            session_id=session_id
        ))

        # Save results
        save_file = os.path.join(results_root, f'scene_{scene_idx}_cam_{CAM_IDX}_sam3_outputs__{TEXT_PROMPT}.npy')
        if os.path.exists(save_file):
            existing_outputs = np.load(save_file, allow_pickle=True).item()
            existing_outputs.update(outputs_per_frame)
            outputs_per_frame = existing_outputs
        with open(save_file, 'wb') as f:
            np.save(f, outputs_per_frame)
        print(f"Results saved to {save_file}")
        
        # Clear memory between prompts
        del outputs_per_frame
        gc.collect()
        torch.cuda.empty_cache()
        print(f"Memory cleared. GPU memory available: {torch.cuda.mem_get_info()[0] / 1e9:.2f} GB")
        
    # Cleanup temp images AFTER ALL PROMPTS (once at the end)
    shutil.rmtree(TEMP_VIDEO_DIR)
    print("Temporary files cleaned up.")
    print(f"Scene processing complete!")
