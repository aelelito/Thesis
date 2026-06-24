"""
Write pipeline detections to the nuScenes submission JSON format.

Both nuScenes and ECP use the same format (ECP data is converted to nuScenes
format, so sample_token exists for both).

Submission schema
-----------------
{
    "split":        "train",
    "mapping_name": "8class",
    "meta":         {"use_camera": true, ...},
    "results": {
        "<sample_token>": [
            {
                "sample_token":    "<sample_token>",
                "translation":     [x, y, z],          # ego frame, box center
                "size":            [width, length, height],  # metres (nuScenes convention)
                "rotation":        [w, x, y, z],        # quaternion, ego frame
                "velocity":        [0.0, 0.0],
                "detection_name":  "pedestrian",
                "detection_score": 0.87,
                "attribute_name":  ""
            }
        ]
    }
}

Size convention
---------------
nuScenes [width, length, height]:
  - width  = lateral dimension (y in box-local frame)
  - length = longitudinal dimension (x in box-local frame)
  - height = vertical dimension (z)

Pedestrian OBB dims [W, H, D] (camera-space right/up/fwd axes):
  → [width=W, length=D, height=H]

Object OBB dims [length, width, height] (ego-frame PCA axes):
  → [width=dims[1], length=dims[0], height=dims[2]]

Rotation quaternion
-------------------
For a box with yaw θ in ego frame (rotation around z-up axis):
  Quaternion [w, x, y, z] = [cos(θ/2), 0, 0, sin(θ/2)]
"""
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml
from pyquaternion import Quaternion

from ..utils.geometry import cam_to_ego, ego_to_global

# submission.py → writers → autolabeling → src → AutoLabeling → configs/class_mapping
_CLASS_MAPPING_DIR = Path(__file__).parent.parent.parent.parent / 'configs' / 'class_mapping'

# Maps SAM3 text prompt → nuScenes 8class detection name
PROMPT_TO_CLASS = {
    'pedestrian':           'pedestrian',
    'car':                  'car',
    'bicycle':              'bicycle',
    'motorcycle':           'motorcycle',
    'bus':                  'bus',
    'truck':                'truck',
    'construction vehicle': 'construction_vehicle',
    'trailer':              'trailer',
}

_META = {
    'use_camera':   True,
    'use_lidar':    False,
    'use_radar':    False,
    'use_map':      False,
    'use_external': False,
}


def _yaw_to_quat(yaw_rad: float) -> List[float]:
    """Convert a yaw angle (rotation around z-up) to [w, x, y, z] quaternion."""
    q = Quaternion(axis=[0.0, 0.0, 1.0], angle=float(yaw_rad))
    return [q.w, q.x, q.y, q.z]


def _pedestrian_box(r: dict, R_c2e: np.ndarray, t_c2e: np.ndarray,
                    R_e2g: np.ndarray, t_e2g: np.ndarray, sample_token: str) -> dict:
    """
    Convert a pedestrian body result to a nuScenes submission box.

    OBB and orientation are in camera space → ego → global.
    nuScenes evaluation expects translations in global frame.
    """
    # Center: camera → ego → global
    center_cam = r['obb_center'].reshape(1, 3).astype(np.float64)
    center_ego = cam_to_ego(center_cam, R_c2e, t_c2e)
    center_global = ego_to_global(center_ego, R_e2g, t_e2g)[0]

    # Dims: [W, H, D] (right, up, fwd in camera) → [width, length, height]
    dims = r['obb_dims'].astype(np.float64)
    width, height, depth = dims[0], dims[1], dims[2]
    size = [float(width), float(depth), float(height)]

    # Yaw: camera → ego → global, then arctan2 in global XY plane
    fwd_cam = r['orientation_fwd'].astype(np.float64)
    fwd_ego = R_c2e @ fwd_cam
    fwd_global = R_e2g @ fwd_ego
    yaw_global = float(np.arctan2(fwd_global[1], fwd_global[0]))
    rotation = _yaw_to_quat(yaw_global)

    return {
        'sample_token':    sample_token,
        'translation':     center_global.tolist(),
        'size':            size,
        'rotation':        rotation,
        'velocity':        [0.0, 0.0],
        'detection_name':  'pedestrian',
        'detection_score': float(r.get('score', 1.0)),
        'attribute_name':  '',
    }


def _object_box(r: dict, R_e2g: np.ndarray, t_e2g: np.ndarray, sample_token: str) -> dict:
    """
    Convert an object result to a nuScenes submission box.

    Object OBB center and yaw are in ego frame → transformed to global.
    """
    # Center: ego → global
    center_ego = r['obb_center'].reshape(1, 3).astype(np.float64)
    center_global = ego_to_global(center_ego, R_e2g, t_e2g)[0]

    # Dims: [length, width, height] → [width, length, height]
    dims = r['obb_dims'].astype(np.float64)
    size = [float(dims[1]), float(dims[0]), float(dims[2])]

    # Yaw: ego → global (rotate the forward direction, recompute yaw)
    yaw_ego = float(r['obb_yaw'])
    fwd_ego = np.array([np.cos(yaw_ego), np.sin(yaw_ego), 0.0])
    fwd_global = R_e2g @ fwd_ego
    yaw_global = float(np.arctan2(fwd_global[1], fwd_global[0]))
    rotation = _yaw_to_quat(yaw_global)

    det_name = PROMPT_TO_CLASS.get(r['prompt'], r['prompt'])

    return {
        'sample_token':    sample_token,
        'translation':     center_global.tolist(),
        'size':            size,
        'rotation':        rotation,
        'velocity':        [0.0, 0.0],
        'detection_name':  det_name,
        'detection_score': float(r.get('score', 1.0)),
        'attribute_name':  '',
    }


def build_submission(
    frames,
    body_results: Dict[int, list],
    obj_results: Dict[int, list],
    split: str,
    mapping_name: str,
) -> dict:
    """
    Assemble the full submission dict from per-frame pipeline results.

    Parameters
    ----------
    frames       : list of FrameRecord
    body_results : {frame_idx: [pedestrian result dicts]}
    obj_results  : {frame_idx: [object result dicts]}
    split        : "train" or "val"
    mapping_name : "8class" / "3class" / "1class"

    Returns
    -------
    submission dict ready for json.dump
    """
    results = {}
    for i, frame in enumerate(frames):
        token = frame.sample_token
        boxes = []

        for r in body_results.get(i, []):
            boxes.append(_pedestrian_box(r, frame.R_c2e, frame.t_c2e,
                                         frame.R_e2g, frame.t_e2g, token))

        for r in obj_results.get(i, []):
            boxes.append(_object_box(r, frame.R_e2g, frame.t_e2g, token))

        results[token] = boxes

    return {
        'split':        split,
        'mapping_name': mapping_name,
        'meta':         _META,
        'results':      results,
    }


def remap_submission(submission_8class: dict, target_mapping: str) -> dict:
    """
    Derive a coarser submission from an 8class one by remapping detection_name
    values using configs/class_mapping/<target_mapping>.yaml.

    Boxes whose 8class name is not present in the remap table are dropped
    (shouldn't happen with the current prompt set, but safe to guard).
    """
    remap_path = _CLASS_MAPPING_DIR / f'{target_mapping}.yaml'
    with open(remap_path, encoding='utf-8') as f:
        remap = yaml.safe_load(f)['remap']

    new_results = {}
    for token, boxes in submission_8class['results'].items():
        remapped = []
        for box in boxes:
            new_name = remap.get(box['detection_name'])
            if new_name is not None:
                remapped.append({**box, 'detection_name': new_name})
        new_results[token] = remapped

    return {**submission_8class, 'results': new_results, 'mapping_name': target_mapping}


def write_submission(
    submission: dict,
    output_path: Path,
) -> None:
    """Write a submission dict to disk as JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(submission, f)
    n_frames = len(submission['results'])
    n_boxes  = sum(len(v) for v in submission['results'].values())
    print(f'Submission written: {output_path}  ({n_frames} frames, {n_boxes} boxes)')
