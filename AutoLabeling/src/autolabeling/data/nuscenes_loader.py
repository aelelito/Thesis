"""
nuScenes data loader.

Iterates keyframes (samples) within scenes. Only keyframes are yielded because
only keyframes have GT annotations and are expected in the submission JSON.

`frame_start` / `frame_end` are keyframe indices within a scene (0-based),
not sample_data indices.

Images are NOT loaded at collection time — FrameRecord only stores the path
and metadata. Call frame.load_images() when the image is actually needed.
This keeps RAM flat regardless of how many frames are collected.

NOTE — keyframes vs. all frames
--------------------------------
nuScenes scenes are ~20 s long. Cameras run at 12 Hz (~240 frames/scene) but
only every 6th frame is a keyframe (2 Hz, ~40/scene) with GT annotations.
This loader iterates nusc.sample (keyframes only).

When adding temporal context or tracking in a later pipeline stage, switch to
iterating nusc.sample_data filtered to the relevant camera channel so that ALL
~240 frames per scene are processed, not just the 40 annotated keyframes.
"""
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import cv2
import numpy as np
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion


@dataclass
class FrameRecord:
    """
    Lightweight metadata record for one camera keyframe.

    Images are loaded on demand via load_images() to avoid holding all frames
    in RAM simultaneously.
    """
    sample_token: str          # nuScenes sample (keyframe) token
    img_path: str
    K: np.ndarray              # (3, 3) float64  camera intrinsics
    R_c2e: np.ndarray          # (3, 3) float64  rotation camera → ego
    t_c2e: np.ndarray          # (3,)   float64  translation camera → ego
    R_e2g: np.ndarray          # (3, 3) float64  rotation ego → global
    t_e2g: np.ndarray          # (3,)   float64  translation ego → global
    scene_name: str
    frame_idx: int             # keyframe index within scene (0-based)
    # LiDAR fields — None when the dataset has no LIDAR_TOP channel
    lidar_path: Optional[str] = None
    R_l2e: Optional[np.ndarray] = None      # (3, 3) float64  rotation LiDAR → ego
    t_l2e: Optional[np.ndarray] = None      # (3,)   float64  translation LiDAR → ego
    lidar_sd_token: Optional[str] = None    # sample_data token for LIDAR_TOP anchor sweep (enables multi-sweep aggregation)

    def load_images(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load and return (img_rgb, img_bgr) as uint8 arrays. Called per-frame during inference."""
        img_bgr = cv2.imread(self.img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_rgb, img_bgr


def _load_frame(nusc: NuScenes, sample_token: str, camera: str, frame_idx: int) -> FrameRecord:
    sample   = nusc.get('sample', sample_token)
    sd_token = sample['data'][camera]
    sd       = nusc.get('sample_data', sd_token)
    cal      = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
    scene    = nusc.get('scene', sample['scene_token'])

    ego_pose = nusc.get('ego_pose', sd['ego_pose_token'])

    K     = np.array(cal['camera_intrinsic'], dtype=np.float64)
    R_c2e = Quaternion(cal['rotation']).rotation_matrix.astype(np.float64)
    t_c2e = np.array(cal['translation'], dtype=np.float64)
    R_e2g = Quaternion(ego_pose['rotation']).rotation_matrix.astype(np.float64)
    t_e2g = np.array(ego_pose['translation'], dtype=np.float64)

    # LiDAR calibration — only available when the dataset has LIDAR_TOP
    lidar_path     = None
    R_l2e          = None
    t_l2e          = None
    lidar_sd_token = None
    if 'LIDAR_TOP' in sample['data']:
        lidar_sd_token = sample['data']['LIDAR_TOP']
        lidar_sd       = nusc.get('sample_data', lidar_sd_token)
        lidar_cal      = nusc.get('calibrated_sensor', lidar_sd['calibrated_sensor_token'])
        lidar_path     = nusc.get_sample_data_path(lidar_sd_token)
        R_l2e          = Quaternion(lidar_cal['rotation']).rotation_matrix.astype(np.float64)
        t_l2e          = np.array(lidar_cal['translation'], dtype=np.float64)

    return FrameRecord(
        sample_token=sample_token,
        img_path=nusc.get_sample_data_path(sd_token),
        K=K,
        R_c2e=R_c2e,
        t_c2e=t_c2e,
        R_e2g=R_e2g,
        t_e2g=t_e2g,
        scene_name=scene['name'],
        frame_idx=frame_idx,
        lidar_path=lidar_path,
        R_l2e=R_l2e,
        t_l2e=t_l2e,
        lidar_sd_token=lidar_sd_token,
    )


def iter_scene_frames(
    nusc: NuScenes,
    scene_name: str,
    camera: str = 'CAM_FRONT',
    frame_start: int = 0,
    frame_end: Optional[int] = None,
) -> Iterator[FrameRecord]:
    """
    Yield FrameRecord for each keyframe in a scene within [frame_start, frame_end).

    Parameters
    ----------
    nusc        : NuScenes instance
    scene_name  : e.g. "scene-0001"
    camera      : camera channel, default CAM_FRONT
    frame_start : first keyframe index to include (0-based, inclusive)
    frame_end   : last keyframe index to exclude; None = until end of scene
    """
    scene = next((s for s in nusc.scene if s['name'] == scene_name), None)
    if scene is None:
        raise ValueError(f'Scene not found: {scene_name}')

    token = scene['first_sample_token']
    idx   = 0
    while token:
        if frame_end is not None and idx >= frame_end:
            break
        if idx >= frame_start:
            yield _load_frame(nusc, token, camera, idx)
        token = nusc.get('sample', token)['next']
        idx  += 1


def collect_frames(
    nusc: NuScenes,
    scene_names: Optional[List[str]],
    camera: str = 'CAM_FRONT',
    frame_start: int = 0,
    frame_end: Optional[int] = None,
) -> List[FrameRecord]:
    """
    Collect all FrameRecords for the given scenes and frame range.

    Fast and memory-light: only paths and camera metadata are stored,
    no images are loaded here.

    Parameters
    ----------
    nusc         : NuScenes instance
    scene_names  : list of scene names to process; None = all scenes in dataset
    camera       : camera channel
    frame_start  : keyframe index start (inclusive, applies per-scene)
    frame_end    : keyframe index end   (exclusive, applies per-scene); None = end of scene
    """
    if scene_names is None:
        scene_names = [s['name'] for s in nusc.scene]

    frames = []
    for name in scene_names:
        scene_frames = list(iter_scene_frames(nusc, name, camera, frame_start, frame_end))
        print(f'  {name}: {len(scene_frames)} keyframe(s) (camera={camera})')
        frames.extend(scene_frames)

    return frames
