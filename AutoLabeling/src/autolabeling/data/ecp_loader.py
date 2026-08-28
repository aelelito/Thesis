"""
ECP data loader.

ECP data is stored in nuScenes format (ecp2nuscenes conversion), so the
underlying loading logic is identical to the nuScenes loader. This module
exists as a separate entry point to make the dataset clear in configs and
to allow ECP-specific adjustments in the future (e.g. handling unannotated
frames differently).

Note: ECP uses 'v1.0-trainval' and its own data root. All other logic
(scene iteration, camera intrinsics, extrinsics) is the same as nuScenes.
"""
from typing import Iterator, List, Optional

from nuscenes.nuscenes import NuScenes

from .nuscenes_loader import (
    FrameRecord, _load_frame, collect_frames, collect_frames_multi_cam,
    iter_scene_frames,
)

# Re-export so callers can import from either loader module uniformly
__all__ = ['FrameRecord', 'iter_scene_frames', 'collect_frames',
           'collect_frames_multi_cam',
           'get_annotated_scene_names', 'collect_annotated_frames',
           'collect_annotated_frames_multi_cam']


def collect_annotated_frames_multi_cam(
    nusc: NuScenes,
    cameras: List[str],
) -> dict:
    """
    Return {camera_name: [FrameRecord]} for annotated keyframes only.

    Identical frame ordering for every camera so that frames_per_cam[cam_A][i]
    and frames_per_cam[cam_B][i] correspond to the same keyframe.
    """
    frames_per_cam = {}
    for cam in cameras:
        frames_per_cam[cam] = collect_annotated_frames(nusc, cam)
    return frames_per_cam


def collect_annotated_frames(nusc: NuScenes, camera: str = 'CAM_FRONT') -> List[FrameRecord]:
    """
    Return FrameRecords for only the annotated keyframes across all ECP scenes.

    ECP annotates a sparse subset of keyframes (~33 out of thousands).
    Use this instead of collect_frames when you only need to generate
    pseudo-labels for frames that have GT to evaluate against.
    """
    annotated_tokens = {ann['sample_token'] for ann in nusc.sample_annotation}
    frames = []
    for sample in sorted(nusc.sample, key=lambda s: s['timestamp']):
        if sample['token'] in annotated_tokens:
            scene = nusc.get('scene', sample['scene_token'])
            # frame_idx within scene (used for display only)
            frame_idx = 0
            t = scene['first_sample_token']
            while t and t != sample['token']:
                t = nusc.get('sample', t)['next']
                frame_idx += 1
            frames.append(_load_frame(nusc, sample['token'], camera, frame_idx))
    print(f'ECP annotated-only: {len(frames)} keyframe(s) across '
          f'{len({f.scene_name for f in frames})} scene(s).')
    return frames


def get_annotated_scene_names(nusc: NuScenes) -> List[str]:
    """
    Return scene names that have at least one ground-truth annotation.

    ECP only annotates a subset of keyframes; this filters to scenes that
    have any annotation at all (matches the logic in evaluation_ecp.py).
    """
    annotated_sample_tokens = {ann['sample_token'] for ann in nusc.sample_annotation}
    annotated_scene_names   = set()
    for sample in nusc.sample:
        if sample['token'] in annotated_sample_tokens:
            scene = nusc.get('scene', sample['scene_token'])
            annotated_scene_names.add(scene['name'])
    return sorted(annotated_scene_names)
