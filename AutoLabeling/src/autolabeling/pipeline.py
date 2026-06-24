"""
Pipeline orchestrator — stage-batched model loading with per-frame checkpointing.

Memory profile
--------------
Images are NOT stored in FrameRecord. Each model wrapper calls frame.load_images()
at the start of run_frame() and the arrays are released when the function returns.
When checkpointing is enabled, SAM3 results for stage 2/3 are also loaded one
frame at a time from disk rather than kept in RAM.

Peak RAM at any point:
  - One frame's images (~8 MB)
  - One frame's SAM3 masks (~1–3 MB sparse)
  - Active model weights (GPU VRAM, not RAM)
  → Flat profile regardless of scene count or number of frames.

Checkpoint layout
-----------------
<checkpoint_dir>/
    sam3/     000000.pkl.gz   000001.pkl.gz   ...
    body/     000000.pkl.gz   ...
    objects/  000000.pkl.gz   ...

Stage order
-----------
1. SAM3 segmentation     (all frames, GPU)
2. SAM3D Body            (all frames, GPU, pedestrians)
3. SAM3D Objects + MoGe  (all frames, GPU, all other classes)
4. Orientation + OBB     (CPU, per frame)
"""
import gc
import gzip
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from .fitting.obb import compute_obb_gravity_aligned, compute_obb_pedestrian
from .models.sam3_segmentor import SAM3Segmentor
from .models.sam3d_body import SAM3DBodyModel
from .models.sam3d_objects import SAM3DObjectsModel
from .orientation.pedestrian import facing_direction


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _ckpt_path(checkpoint_dir: Optional[Path], stage: str, idx: int) -> Optional[Path]:
    if checkpoint_dir is None:
        return None
    return checkpoint_dir / stage / f'{idx:06d}.pkl.gz'


def _save(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'wb', compresslevel=3) as f:
        pickle.dump(data, f, protocol=4)


def _load(path: Path):
    try:
        with gzip.open(path, 'rb') as f:
            return pickle.load(f)
    except (EOFError, OSError, pickle.UnpicklingError) as e:
        # Checkpoint was truncated mid-write (e.g. disk full). Delete it so
        # the frame is re-processed on the next run.
        print(f'  [warn] corrupt checkpoint {path.name} ({e}), deleting and re-processing.')
        path.unlink(missing_ok=True)
        return None


# ── Sparse mask encoding (SAM3 checkpoints) ───────────────────────────────────

def _mask_sparse(mask: np.ndarray) -> dict:
    return {'idx': np.where(mask.ravel())[0].astype(np.uint32), 'shape': mask.shape}


def _mask_dense(sparse: dict) -> np.ndarray:
    out = np.zeros(sparse['shape'][0] * sparse['shape'][1], dtype=bool)
    out[sparse['idx']] = True
    return out.reshape(sparse['shape'])


def _sam3_to_ckpt(frame_sam3: dict) -> dict:
    return {
        prompt: [{'mask_sparse': _mask_sparse(d['binary_mask']),
                  'score': d['score'], 'prompt': d['prompt']}
                 for d in dets]
        for prompt, dets in frame_sam3.items()
    }


def _sam3_from_ckpt(ckpt: dict) -> dict:
    return {
        prompt: [{'binary_mask': _mask_dense(d['mask_sparse']),
                  'score': d['score'], 'prompt': d['prompt']}
                 for d in dets]
        for prompt, dets in ckpt.items()
    }


# ── Body / object checkpoint encoding ────────────────────────────────────────

def _body_to_ckpt(body: list) -> dict:
    """Faces are shared topology — store once per frame."""
    if not body:
        return {'faces': None, 'people': []}
    return {
        'faces': body[0]['faces'],
        'people': [{'vertices': r['vertices'], 'joints_3d': r['joints_3d'],
                    'cam_t': r['cam_t'], 'score': r['score']}
                   for r in body],
    }


def _body_from_ckpt(ckpt: dict) -> list:
    faces = ckpt['faces']
    return [{'vertices': p['vertices'], 'faces': faces,
             'joints_3d': p['joints_3d'], 'cam_t': p['cam_t'], 'score': p['score']}
            for p in ckpt['people']]


def _obj_to_ckpt(objs: list) -> list:
    return [{'vertices': r['vertices'], 'faces': r['faces'],
             'score': r['score'], 'prompt': r['prompt']}
            for r in objs]


_obj_from_ckpt = lambda ckpt: ckpt  # already the right format


# ── SAM3 result accessor ──────────────────────────────────────────────────────

def _get_sam3(i: int, sam3_mem: dict, checkpoint_dir: Optional[Path]) -> dict:
    """
    Return SAM3 results for frame i.

    When checkpointing is enabled, results are loaded from disk one frame at a
    time so the full results dict never accumulates in RAM.
    When checkpointing is disabled, results come from the in-memory dict.
    """
    if checkpoint_dir is not None:
        return _sam3_from_ckpt(_load(_ckpt_path(checkpoint_dir, 'sam3', i)))
    return sam3_mem[i]


# ── Postprocess (CPU) ─────────────────────────────────────────────────────────

def _postprocess_frame(frame, body: list, objs: list) -> None:
    """Orientation estimation + OBB fitting for one frame. Modifies lists in-place."""
    for r in body:
        fwd = facing_direction(r['joints_3d'])
        r['orientation_fwd'] = fwd
        corners, center, dims = compute_obb_pedestrian(r['vertices'], fwd)
        r['obb_corners'] = corners
        r['obb_center']  = center   # camera space
        r['obb_dims']    = dims
        r['obb_yaw']     = float(np.arctan2(fwd[0], fwd[2]))

    for r in objs:
        r['orientation_fwd'] = None
        corners, center, dims, yaw = compute_obb_gravity_aligned(
            r['vertices'], frame.R_c2e, frame.t_c2e, ground_z=None
        )
        r['obb_corners'] = corners
        r['obb_center']  = center   # ego space
        r['obb_dims']    = dims
        r['obb_yaw']     = yaw


# ── Rider OBB merge ───────────────────────────────────────────────────────────

_RIDER_CLASSES = {'bicycle', 'motorcycle'}


def _merge_rider_obbs(
    frame, body_list: list, obj_list: list, dist_thresh: float = 1.5
) -> tuple:
    """
    Merge pedestrian body OBBs with co-located bicycle / motorcycle OBBs.

    When a cyclist or motorcyclist is detected, SAM3 produces two masks:
      • 'pedestrian'  → SAM3D Body  (high-quality body mesh, obb_center in camera space)
      • 'bicycle' or 'motorcycle' → SAM3D Objects  (vehicle mesh, obb_center in ego space)

    If the two OBB centers are within dist_thresh metres of each other in ego
    space, the results are merged:
      • A single OBB is recomputed from the combined camera-space vertices.
      • The merged entry keeps the vehicle prompt label (bicycle / motorcycle).
      • The pedestrian body entry is removed from body_list.

    Each body result is matched to at most one vehicle; each vehicle is matched
    to at most one body (closest wins if multiple candidates exist).

    Parameters
    ----------
    frame       : FrameRecord  (needs R_c2e, t_c2e)
    body_list   : list of body result dicts (modified in-place for matched entries)
    obj_list    : list of object result dicts (modified in-place for matched entries)
    dist_thresh : maximum ego-space distance [m] to trigger a merge (default 1.5 m)

    Returns
    -------
    new_body_list : body_list with matched pedestrian entries removed
    obj_list      : same list reference (matched entries updated in-place)
    """
    body_used = set()

    for bi, br in enumerate(body_list):
        # body obb_center is in camera space — transform to ego for comparison
        bc_ego = frame.R_c2e @ br['obb_center'].astype(np.float64) + frame.t_c2e

        best_oi, best_d = None, float('inf')
        for oi, or_ in enumerate(obj_list):
            if or_['prompt'] not in _RIDER_CLASSES:
                continue
            d = float(np.linalg.norm(bc_ego - np.array(or_['obb_center'], dtype=np.float64)))
            if d <= dist_thresh and d < best_d:
                best_d, best_oi = d, oi

        if best_oi is None:
            continue

        or_ = obj_list[best_oi]
        combined = np.concatenate([br['vertices'], or_['vertices']], axis=0).astype(np.float32)
        corners, center, dims, yaw = compute_obb_gravity_aligned(
            combined, frame.R_c2e, frame.t_c2e, ground_z=None
        )
        or_['vertices']    = combined
        or_['obb_corners'] = corners
        or_['obb_center']  = center
        or_['obb_dims']    = dims
        or_['obb_yaw']     = yaw
        body_used.add(bi)
        print(f'    [rider merge] pedestrian → "{or_["prompt"]}"  (d={best_d:.2f} m)')

    new_body = [br for bi, br in enumerate(body_list) if bi not in body_used]
    return new_body, obj_list


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(
    cfg,
    frames: list,
    device: Optional[str] = None,
    checkpoint_dir: Optional[Path] = None,
) -> tuple:
    """
    Run the full auto-labeling pipeline on a list of FrameRecords.

    Parameters
    ----------
    cfg            : config namespace (from load_config in run_pipeline.py)
    frames         : list of FrameRecord  (images NOT pre-loaded)
    device         : override device; defaults to 'cuda' if available
    checkpoint_dir : directory for per-frame checkpoints; None = no checkpointing

    Returns
    -------
    body_results : {frame_idx: [pedestrian dicts with obb_* fields]}
    obj_results  : {frame_idx: [object dicts with obb_* fields]}
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dev_root     = Path(cfg.models.dev_root)
    sam3_ckpt    = cfg.models.sam3_ckpt or _hf_download_sam3()
    body_repo    = (Path(cfg.models.sam3d_body_repo) if cfg.models.sam3d_body_repo
                    else dev_root / 'SAM3D' / 'sam-3d-body')
    obj_cfg_path = (Path(cfg.models.sam3d_obj_cfg) if cfg.models.sam3d_obj_cfg
                    else dev_root / 'SAM3D' / 'sam-3d-objects' / 'checkpoints' / 'hf' / 'pipeline.yaml')
    obj_repo     = obj_cfg_path.parent.parent.parent

    # sam3_mem is only populated when checkpointing is disabled (fits in RAM).
    # When checkpointing is enabled, each stage loads results per-frame from disk.
    sam3_mem = {}

    # ── SAM3 segmentation ─────────────────────────────────────────────────────
    print('\n[SAM3 segmentation]')
    pending = []
    for i, frame in enumerate(frames):
        p = _ckpt_path(checkpoint_dir, 'sam3', i)
        if p is not None and p.exists():
            # Validate now — corrupt SAM3 checkpoints would otherwise only be
            # discovered mid-way through the body/objects stage when the SAM3
            # model is no longer loaded, making recovery impossible.
            if _load(p) is None:
                pending.append((i, frame))  # corrupt, will be re-processed
            # else: valid, will be loaded on demand per-frame in later stages
        else:
            pending.append((i, frame))

    n_cached = len(frames) - len(pending)
    if n_cached:
        print(f'  Resuming: {n_cached} frame(s) cached, {len(pending)} to process.')

    if pending:
        _dedup_cfg  = getattr(cfg, 'cross_class_dedup', None)
        _dedup_iou  = getattr(_dedup_cfg, 'iou_thresh', 0.5) if _dedup_cfg else 0.5
        _dedup_prio = vars(getattr(_dedup_cfg, 'priority', None) or {}) if _dedup_cfg else {}
        segmentor = SAM3Segmentor(
            checkpoint_path=sam3_ckpt,
            prompts=vars(cfg.prompts),
            score_thresh=cfg.thresholds.sam3_score,
            min_mask_px=cfg.thresholds.min_mask_px,
            device=device,
            cross_class_iou_thresh=_dedup_iou,
            cross_class_priority=_dedup_prio,
        )
        segmentor.load()
        bar = tqdm(pending, total=len(frames), initial=len(frames) - len(pending),
                   desc='SAM3', unit='frame')
        for i, frame in bar:
            bar.set_postfix_str(f'{frame.scene_name}  frame {frame.frame_idx}')
            result = segmentor.run_frame(frame)
            p = _ckpt_path(checkpoint_dir, 'sam3', i)
            if p:
                _save(p, _sam3_to_ckpt(result))
            else:
                sam3_mem[i] = result
        segmentor.unload()
    else:
        print(f'  All {len(frames)} frame(s) loaded from cache.')

    _print_gpu(device)

    # ── SAM3D Body ────────────────────────────────────────────────────────────
    print('\n[SAM3D Body]')
    body_results = {}
    pending = []
    for i, frame in enumerate(frames):
        p = _ckpt_path(checkpoint_dir, 'body', i)
        if p is not None and p.exists():
            data = _load(p)
            if data is not None:
                body_results[i] = _body_from_ckpt(data)
            else:
                pending.append((i, frame))  # corrupt checkpoint, re-process
        else:
            pending.append((i, frame))

    if pending:
        n_cached = len(frames) - len(pending)
        if n_cached:
            print(f'  Resuming: {n_cached} frame(s) cached, {len(pending)} to process.')
        body_model = SAM3DBodyModel(
            repo_path=str(body_repo),
            checkpoint_path=str(body_repo / 'checkpoints' / 'sam-3d-body-dinov3' / 'model.ckpt'),
            mhr_path=str(body_repo / 'checkpoints' / 'sam-3d-body-dinov3' / 'assets' / 'mhr_model.pt'),
            bbox_thresh=cfg.thresholds.body_bbox,
            iou_merge_thresh=cfg.thresholds.sam3_iou_merge,
            device=device,
        )
        body_model.load()
        bar = tqdm(pending, total=len(frames), initial=len(frames) - len(pending),
                   desc='SAM3D Body', unit='frame')
        for i, frame in bar:
            bar.set_postfix_str(f'{frame.scene_name}  frame {frame.frame_idx}')
            ped_dets = _get_sam3(i, sam3_mem, checkpoint_dir).get('pedestrian', [])
            result = body_model.run_frame(frame, ped_dets)
            body_results[i] = result
            p = _ckpt_path(checkpoint_dir, 'body', i)
            if p:
                _save(p, _body_to_ckpt(result))
        body_model.unload()
    else:
        print(f'  All {len(frames)} frame(s) loaded from cache.')

    _print_gpu(device)

    # ── SAM3D Objects ─────────────────────────────────────────────────────────
    _lidar_cfg   = getattr(cfg, 'lidar', None)
    pointmap_mode = getattr(_lidar_cfg, 'pointmap_mode', 'baseline')
    _mode_labels = {
        'baseline':       'MoGe baseline (non-metric)',
        'o1_lidar':       'O1 — sparse LiDAR pointmap',
        'o2_moge_affine': 'O2 — MoGe + global affine calibration',
    }
    print(f'\n[SAM3D Objects — {_mode_labels.get(pointmap_mode, pointmap_mode)}]')
    obj_results = {}
    pending = []
    for i, frame in enumerate(frames):
        p = _ckpt_path(checkpoint_dir, 'objects', i)
        if p is not None and p.exists():
            data = _load(p)
            if data is not None:
                obj_results[i] = _obj_from_ckpt(data)
            else:
                pending.append((i, frame))  # corrupt checkpoint, re-process
        else:
            pending.append((i, frame))

    if pending:
        n_cached = len(frames) - len(pending)
        if n_cached:
            print(f'  Resuming: {n_cached} frame(s) cached, {len(pending)} to process.')
        obj_model = SAM3DObjectsModel(
            repo_path=str(obj_repo),
            config_path=str(obj_cfg_path),
            prompts=vars(cfg.prompts),
            device=device,
            pointmap_mode=pointmap_mode,
        )
        obj_model.load()
        bar = tqdm(pending, total=len(frames), initial=len(frames) - len(pending),
                   desc='SAM3D Objects', unit='frame')
        for i, frame in bar:
            bar.set_postfix_str(f'{frame.scene_name}  frame {frame.frame_idx}')
            frame_sam3 = _get_sam3(i, sam3_mem, checkpoint_dir)
            result = obj_model.run_frame(frame, frame_sam3)
            obj_results[i] = result
            p = _ckpt_path(checkpoint_dir, 'objects', i)
            if p:
                _save(p, _obj_to_ckpt(result))
        obj_model.unload()
    else:
        print(f'  All {len(frames)} frame(s) loaded from cache.')

    _print_gpu(device)

    # ── Orientation + OBB (CPU) ───────────────────────────────────────────────
    print('\nOrientation estimation + OBB fitting...')
    for i, frame in enumerate(frames):
        _postprocess_frame(frame, body_results[i], obj_results[i])
        body_results[i], obj_results[i] = _merge_rider_obbs(
            frame, body_results[i], obj_results[i]
        )
    print('  Done.')

    return body_results, obj_results


def _print_gpu(device: str) -> None:
    if device == 'cuda':
        print(f'  GPU free: {torch.cuda.mem_get_info()[0] / 1024**3:.1f} GB')


def _hf_download_sam3() -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id='facebook/sam3', filename='sam3.pt')
