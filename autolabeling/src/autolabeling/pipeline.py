"""
Pipeline orchestrator — stage-batched model loading with per-frame checkpointing.

Stage order
-----------
1. SAM3 segmentation            (all frames, GPU)
2. TerraSeg ground removal      (all frames, GPU)
   Run PER SWEEP, in each sweep's own ego frame, BEFORE aggregating (TerraSeg is trained
   on single independent scans, height/range relative to that scan's own sensor origin).
   Produces the ground-free aggregated cloud `pts_ego`, cached per keyframe (shared across
   cameras) and used by B1 and by every pointmap mode.
3. PseudoLabeler pre-fitting    (only when B2 is on)
   Fitted on the FULL aggregated cloud (ground included) -- it must see ground to find it,
   and as an offline per-frame fit it benefits from the extra density.
4. SAM3D Body + B1 (depth correction) + B2 (ground anchoring), independently toggleable
5. SAM3D Objects, pointmap mode 1-11 (see models/sam3d_objects.py)
6. Orientation + OBB + rider merge (CPU). No OBB filter of any kind.

The only LiDAR pre-filter is the ego-body point filter, applied per sweep in that sweep's
own ego frame before any transform. There is no range filter and no OBB size/volume/ego
filter.

Memory profile
--------------
Images are NOT stored in FrameRecord (models call frame.load_images()). SAM3 masks and the
ground-free clouds are read from disk one frame at a time, so RAM stays flat regardless of
scene count.

Checkpoint layout
-----------------
<checkpoint_dir>/
    sam3__<hash>/                 000000.pkl.gz ...
    body__<hash>/                 000000.pkl.gz ...
    objects__<mode>__<hash>/      000000.pkl.gz ...
    pl_states__<hash>.pt
<lidar_cache_dir>/nonground__<hash>/<sample_token>.npy     (float32, shared across cameras)

Every stage directory carries a short hash of exactly the config that stage depends on, so
changing the pointmap mode, aggregation window, ego box, thresholds, ... starts a fresh
directory instead of silently resuming stale results from an earlier configuration.
"""
import contextlib
import gzip
import hashlib
import json
import os
import pickle
import sys
import tempfile
import zlib
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional

import numpy as np
import torch
from sklearn.linear_model import RANSACRegressor
from tqdm import tqdm

from .fitting.obb import compute_obb_gravity_aligned, compute_obb_pedestrian, disambiguate_yaw
from .models.sam3_segmentor import SAM3Segmentor
from .models.sam3d_body import SAM3DBodyModel
from .models.sam3d_objects import POINTMAP_MODES, SAM3DObjectsModel, resolve_pointmap_mode, _FULL_CLOUD_MODES
from .utils.meshes import attach_full_obbs, slim_object_results
from .orientation.pedestrian import facing_direction
from .utils.lidar import (
    filter_inmask_lidar_hdbscan,
    load_lidar_pts_aggregated,
    load_lidar_pts_nonground_aggregated,
    project_lidar_to_camera,
)
from .utils.terraseg import TerraSegGroundFilter


# ── Config hashing (stage-directory fingerprints) ─────────────────────────────

def _to_plain(x):
    if isinstance(x, SimpleNamespace):
        return {k: _to_plain(v) for k, v in vars(x).items()}
    if isinstance(x, dict):
        return {str(k): _to_plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_plain(v) for v in x]
    return x


def _hash(*parts) -> str:
    s = json.dumps([_to_plain(p) for p in parts], sort_keys=True, default=str)
    return hashlib.md5(s.encode()).hexdigest()[:8]


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
    except (EOFError, OSError, pickle.UnpicklingError, zlib.error) as e:
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
# Masks are persisted (sparse) with the results: cross-camera merge needs each
# detection's binary_mask, and a result restored from a checkpoint without one would
# be silently skipped as a merge candidate.

def _body_to_ckpt(body: list) -> dict:
    """Faces are shared topology — store once per frame."""
    if not body:
        return {'faces': None, 'people': []}
    return {
        'faces': body[0]['faces'],
        'people': [{'vertices': r['vertices'], 'joints_3d': r['joints_3d'],
                    'cam_t': r['cam_t'], 'score': r['score'],
                    'sam3_mask_idx': r.get('sam3_mask_idx'),
                    'mask_sparse': (None if r.get('binary_mask') is None
                                    else _mask_sparse(r['binary_mask']))}
                   for r in body],
    }


def _body_from_ckpt(ckpt: dict) -> list:
    faces = ckpt['faces']
    return [{'vertices': p['vertices'], 'faces': faces,
             'joints_3d': p['joints_3d'], 'cam_t': p['cam_t'], 'score': p['score'],
             'sam3_mask_idx': p.get('sam3_mask_idx'),
             'binary_mask': (None if p.get('mask_sparse') is None
                             else _mask_dense(p['mask_sparse']))}
            for p in ckpt['people']]


def _obj_to_ckpt(objs: list) -> list:
    return [{'vertices': r['vertices'], 'faces': r['faces'], 'obb_raw': r.get('obb_raw'),
             'score': r['score'], 'prompt': r['prompt'], 'o3_mode': r.get('o3_mode'),
             'sam3_index': r.get('sam3_index'), 'ssi_scale': r.get('ssi_scale'), 'ssi_shift': r.get('ssi_shift'),
             'fit_ab': r.get('fit_ab'), 'bg_ab': r.get('bg_ab'),
             'mask_sparse': (None if r.get('binary_mask') is None
                             else _mask_sparse(r['binary_mask']))}
            for r in objs]


def _obj_from_ckpt(ckpt: list) -> list:
    return [{'vertices': r['vertices'], 'faces': r['faces'], 'obb_raw': r.get('obb_raw'),
             'score': r['score'], 'prompt': r['prompt'], 'o3_mode': r.get('o3_mode'),
             'sam3_index': r.get('sam3_index'), 'ssi_scale': r.get('ssi_scale'), 'ssi_shift': r.get('ssi_shift'),
             'fit_ab': r.get('fit_ab'), 'bg_ab': r.get('bg_ab'),
             'binary_mask': (None if r.get('mask_sparse') is None
                             else _mask_dense(r['mask_sparse']))}
            for r in ckpt]


# ── SAM3 result accessor ──────────────────────────────────────────────────────

def _get_sam3(i: int, sam3_mem: dict, checkpoint_dir: Optional[Path], stage: str) -> dict:
    """
    Return SAM3 results for frame i.

    When checkpointing is enabled, results are loaded from disk one frame at a
    time so the full results dict never accumulates in RAM.
    When checkpointing is disabled, results come from the in-memory dict.
    """
    if checkpoint_dir is not None:
        return _sam3_from_ckpt(_load(_ckpt_path(checkpoint_dir, stage, i)))
    return sam3_mem[i]


# ── Ground-free cloud cache (one .npy per keyframe, shared across cameras) ────

def _save_npy_atomic(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'wb') as f:
        np.save(f, arr)
    os.replace(tmp, path)


# ── Ground estimation helpers (PseudoLabeler) ─────────────────────────────────

_B2_RANSAC_Z_MIN    = -1.0
_B2_RANSAC_Z_MAX    =  0.5
_B2_RANSAC_RESIDUAL =  0.10


def _fit_pseudolabeler(pts_ego: np.ndarray, device: str, dev_root: Path):
    """
    Fit a PseudoLabeler MLP on the raw LiDAR sweep for one frame.

    The MLP learns gθ: R² → R  (x,y) → z, finding the ground surface using
    an asymmetric loss (no ground labels needed).  Once fitted it can be queried
    at any (x,y) via pl_model(xy_tensor).

    Returns the fitted model in eval mode, or None if fitting fails / no pts.
    """
    _pl_path = str(dev_root / 'models' / 'TerraSeg' / 'PseudoLabeler_scripts')
    if _pl_path not in sys.path:
        sys.path.insert(0, _pl_path)
    from pseudolabeler_model import PseudoLabeler
    from pseudolabeler_loss  import pseudolabeler_loss as _pl_loss

    _dev     = torch.device(device)
    pl_model = PseudoLabeler().to(_dev)
    pl_model.train()

    _pc = torch.tensor(pts_ego, dtype=torch.float32).to(_dev)
    _pc = pl_model.remove_ego_points(_pc)
    _pc = pl_model.preprocess_denoise_pc(_pc)
    if len(_pc) == 0:
        return None

    _n_steps  = 2500
    _n_warmup = 200   # steps before early stopping kicks in — avoids the random-init
                      # baseline: pred≈0 at init is already a decent ground approx,
                      # so loss_0 is low; lr=1e-2 overshoots on step 1 and loss rises
                      # for ~100-200 steps before dropping below loss_0.  The warmup
                      # window lets the optimizer pass through that noisy phase first.
    _patience = 300   # stop if loss hasn't improved for this many steps after warmup
    print(f'    [PseudoLabeler] fitting on {len(_pc):,} pts...', end=' ', flush=True)
    _opt = torch.optim.AdamW(pl_model.parameters(), lr=1e-2, weight_decay=1e-4)
    _sch = torch.optim.lr_scheduler.CosineAnnealingLR(_opt, T_max=_n_steps, eta_min=1e-4)
    _best_loss, _best_state, _no_improve = float('inf'), None, 0

    for _step in range(_n_steps):
        _opt.zero_grad()
        _pred = pl_model(_pc)
        _loss = _pl_loss(_pred, _pc[:, 2])
        _l = _loss.item()
        # Only track best and count patience AFTER warmup.  Reason: random init gives
        # pred≈0, and for flat terrain (nuScenes) ground is also near z=0, so loss_0 is
        # accidentally low.  If we track from step 0, the step-1 overshoot (lr=1e-2)
        # raises loss above loss_0 and the recovered model never beats it — patience
        # fires at exactly warmup+patience steps.  By delaying tracking to post-warmup,
        # best_loss is set from the actual trained state (step >= 200), not random init.
        if _step >= _n_warmup:
            if _l < _best_loss:
                _best_loss  = _l
                _best_state = {k: v.detach().cpu() for k, v in pl_model.state_dict().items()}
                _no_improve = 0
            else:
                _no_improve += 1
                if _no_improve >= _patience:
                    break
        _loss.backward()
        torch.nn.utils.clip_grad_norm_(pl_model.parameters(), 5.0)
        _opt.step()
        _sch.step()

    if _best_state:
        pl_model.load_state_dict(_best_state)
    pl_model.eval().to(_dev)
    print(f'done  (loss={_best_loss:.5f}, steps={_step + 1})')
    return pl_model


def _restore_pseudolabeler(state_dict: Optional[dict], device: str, dev_root: Path):
    """
    Reconstruct a PseudoLabeler from a cached state_dict.

    Returns None if state_dict is None (fitting failed or not needed for this frame).
    Ensures the PseudoLabeler module is importable by adding the scripts path to sys.path.
    """
    if state_dict is None:
        return None
    _pl_path = str(dev_root / 'models' / 'TerraSeg' / 'PseudoLabeler_scripts')
    if _pl_path not in sys.path:
        sys.path.insert(0, _pl_path)
    from pseudolabeler_model import PseudoLabeler
    _dev     = torch.device(device)
    pl_model = PseudoLabeler().to(_dev)
    pl_model.load_state_dict({k: v.to(_dev) for k, v in state_dict.items()})
    pl_model.eval()
    return pl_model


# ── B1 — depth correction ─────────────────────────────────────────────────────

def _apply_b1_depth_correction(body_results: list, frame, ped_dets: list,
                               pts_ego_vis, u_vis, v_vis, Z_vis,
                               H: int, W: int,
                               hdbscan_kwargs: dict = None) -> None:
    """
    Stage 1: override tz for each pedestrian with HDBSCAN-filtered median LiDAR depth.
    Recomputes tx, ty from mask centroid. Shifts vertices by delta; cam_t updated.
    joints_3d are body-relative and are NOT shifted.
    Modifies body_results in-place.

    pts_ego_vis / u_vis / v_vis / Z_vis are the camera projection of the GROUND-FREE
    aggregated cloud, so ground/feet returns never enter the pedestrian's cluster.
    """
    fx, fy, cx, cy = frame.K[0, 0], frame.K[1, 1], frame.K[0, 2], frame.K[1, 2]
    u_int = np.round(u_vis).astype(int).clip(0, W - 1)
    v_int = np.round(v_vis).astype(int).clip(0, H - 1)

    for i, r in enumerate(body_results):
        # ── Resolve mask: use sam3_mask_idx for direct association ───────────
        # sam3_mask_idx is set by SAM3DBodyModel.run_frame for SAM3-sourced boxes
        # (None for ViTDet-only detections that have no corresponding SAM3 mask).
        sam3_mask_idx = r.get('sam3_mask_idx')
        if sam3_mask_idx is not None and sam3_mask_idx < len(ped_dets):
            mask     = ped_dets[sam3_mask_idx]['binary_mask']
            mask_src = 'sam3_mask'
        else:
            # ViTDet-only: fall back to bbox rect
            bbox = r.get('bbox')
            if bbox is None:
                r['b1_mode'] = 'no_bbox'
                continue
            x1, y1, x2, y2 = [int(v) for v in bbox]
            mask = np.zeros((H, W), dtype=bool)
            mask[max(0, y1):min(H, y2+1), max(0, x1):min(W, x2+1)] = True
            mask_src = 'bbox_rect'

        tz_pred = float(r['cam_t'][2])

        in_mask    = mask[v_int, u_int]
        pts_inmask = pts_ego_vis[in_mask]
        Z_inmask   = Z_vis[in_mask]

        if len(pts_inmask) == 0:
            r['b1_mode'] = 'no_lidar'
            continue

        keep = filter_inmask_lidar_hdbscan(pts_inmask, **(hdbscan_kwargs or {}))

        if keep is None:
            tz_lidar = float(np.percentile(Z_inmask, 15))
            b1_mode  = f'{mask_src}+p15_fallback'
        else:
            tz_lidar = float(np.median(Z_inmask[keep]))
            b1_mode  = f'{mask_src}+hdbscan'

        ys, xs = np.where(mask)
        u_cen  = float(xs.mean())
        v_cen  = float(ys.mean())
        tx_new = (u_cen - cx) / fx * tz_lidar
        ty_new = (v_cen - cy) / fy * tz_lidar

        cam_t_new     = np.array([tx_new, ty_new, tz_lidar], dtype=np.float32)
        delta         = cam_t_new - r['cam_t']
        r['vertices'] = r['vertices'] + delta[None, :]
        r['cam_t']    = cam_t_new
        r['b1_mode']  = b1_mode
        print(f'    [B1-tz ] ped {i}: {tz_pred:.2f}→{tz_lidar:.2f} m  [{b1_mode}]')


# ── B2 — ground anchoring ─────────────────────────────────────────────────────

def _apply_b2_ground_anchoring(body_results: list, frame,
                                pts_ego: np.ndarray,
                                pl_model, device: str,
                                _stats: Optional[list] = None) -> None:
    """
    Stage 2: shift each pedestrian mesh so its lowest vertex sits on the ground.

    Primary path  (pl_model not None): query fitted PseudoLabeler MLP at ped (x,y).
    Fallback      (pl_model is None):  global RANSAC plane on raw LiDAR.
    joints_3d are body-relative and are NOT shifted.
    Modifies body_results in-place.
    """
    R_c2e, t_c2e = frame.R_c2e, frame.t_c2e
    _dev          = torch.device(device)
    _ransac       = None   # lazy-init only if needed

    def _get_z_ransac(ped_xy):
        nonlocal _ransac
        if _ransac is None:
            pts_f   = pts_ego.astype(np.float64)
            g_cands = pts_f[(pts_f[:, 2] >= _B2_RANSAC_Z_MIN) &
                            (pts_f[:, 2] <= _B2_RANSAC_Z_MAX)]
            _ransac = RANSACRegressor(min_samples=3,
                                      residual_threshold=_B2_RANSAC_RESIDUAL,
                                      max_trials=500, random_state=42)
            _ransac.fit(g_cands[:, :2], g_cands[:, 2])
        return float(_ransac.predict([ped_xy])[0]), 'ransac_global'

    for i, r in enumerate(body_results):
        cam_t_ego = R_c2e @ r['cam_t'].astype(np.float64) + t_c2e
        ped_xy    = cam_t_ego[:2]

        if pl_model is not None:
            try:
                xy_t = torch.tensor([[ped_xy[0], ped_xy[1]]],
                                    dtype=torch.float32).to(_dev)
                with torch.no_grad():
                    z_ground = float(pl_model(xy_t).item())
                method = 'pseudolabeler'
            except Exception as e:
                print(f'    [B2] ped {i}: PseudoLabeler query failed ({e}), '
                      f'falling back to RANSAC.')
                z_ground, method = _get_z_ransac(ped_xy)
        else:
            z_ground, method = _get_z_ransac(ped_xy)

        verts_ego = (R_c2e @ r['vertices'].astype(np.float64).T).T + t_c2e
        z_foot    = float(verts_ego[:, 2].min())
        z_shift   = z_ground - z_foot

        delta_cam     = (R_c2e.T @ np.array([0., 0., z_shift])).astype(np.float32)
        r['vertices'] = r['vertices'] + delta_cam[None, :]
        r['cam_t']    = r['cam_t']    + delta_cam
        r['b2_mode']  = method
        print(f'    [B2] ped {i}: foot={z_foot:.2f}→ground={z_ground:.2f} m  '
              f'shift={z_shift:+.2f} m  [{method}]')
        if _stats is not None:
            _stats.append({
                'frame_idx':     frame.frame_idx,
                'scene':         frame.scene_name,
                'ped_idx':       i,
                'z_foot_before': round(z_foot, 4),
                'z_ground':      round(z_ground, 4),
                'z_shift':       round(z_shift, 4),
                'method':        method,
            })


# ── Postprocess (CPU) ─────────────────────────────────────────────────────────

def _postprocess_frame(frame, body: list, objs: list) -> None:
    """
    Orientation estimation + OBB fitting for one frame. Modifies lists in-place.

    Pedestrians: heading from the MHR70 skeleton (no 180° ambiguity).
    Objects:     PCA footprint yaw, disambiguated toward the ego heading (parked
                 vehicles are road-aligned -> the candidate closest to ego travel
                 direction).
    velocity_mps / is_dynamic are always 0 / False (no motion estimation).
    """
    # Ego heading from R_e2g: forward axis [1,0,0] rotated to global XY
    ego_fwd   = frame.R_e2g @ np.array([1.0, 0.0, 0.0])
    ego_heading_rad = float(np.arctan2(ego_fwd[1], ego_fwd[0]))

    for r in body:
        fwd = facing_direction(r['joints_3d'])
        r['orientation_fwd'] = fwd
        corners, center, dims = compute_obb_pedestrian(r['vertices'], fwd)
        r['obb_corners'] = corners
        r['obb_center']  = center   # camera space
        r['obb_dims']    = dims
        r['obb_yaw']     = float(np.arctan2(fwd[0], fwd[2]))
        r['velocity_mps'] = 0.0
        r['is_dynamic']   = False

    for r in objs:
        r['orientation_fwd'] = None
        raw = r.get('obb_raw')      # fitted on the full mesh before it was reduced (missing in old checkpoints)
        if raw is not None:
            corners, center, dims, yaw = raw['corners'], raw['center'], raw['dims'], raw['yaw']
        else:
            corners, center, dims, yaw = compute_obb_gravity_aligned(
                r['vertices'], frame.R_c2e, frame.t_c2e, ground_z=None
            )
        r['obb_corners'] = corners
        r['obb_center']  = center   # ego space
        r['obb_dims']    = dims
        r['obb_yaw']     = disambiguate_yaw(yaw, ego_heading_rad)
        r['velocity_mps'] = 0.0
        r['is_dynamic']   = False


# ── Rider OBB merge ───────────────────────────────────────────────────────────

_RIDER_CLASSES = {'bicycle', 'motorcycle'}


def _merge_rider_obbs(
    frame, body_list: list, obj_list: list, dist_thresh: float = 1
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
        # OBB is fitted to combined vertices; individual meshes are kept separate
        or_['rider_vertices'] = br['vertices']
        or_['rider_faces']    = br['faces']
        or_['obb_corners'] = corners
        or_['obb_center']  = center
        or_['obb_dims']    = dims
        or_['obb_yaw']     = yaw
        body_used.add(bi)
        print(f'    [rider merge] {frame.scene_name} frame {frame.frame_idx}: '
              f'pedestrian → "{or_["prompt"]}"  (d={best_d:.2f} m)')

    new_body = [br for bi, br in enumerate(body_list) if bi not in body_used]
    return new_body, obj_list


# ── Config sanity ─────────────────────────────────────────────────────────────

_LEGACY_MODE_HINT = {
    'o1_lidar': 1, 'o3_local_affine': 2, 'o5_mask_hdbscan': 4,
}


def _resolve_mode(value):
    try:
        return resolve_pointmap_mode(value)
    except ValueError as e:
        hint = ''
        if isinstance(value, str) and value in _LEGACY_MODE_HINT:
            n = _LEGACY_MODE_HINT[value]
            hint = (f"\n  '{value}' is an old mode name: use {n} "
                    f"('{POINTMAP_MODES[n]}') instead.")
        elif isinstance(value, str) and value in ('baseline', 'o2_moge_affine', 'o4_ground_filter'):
            hint = (f"\n  '{value}' was removed (baseline / global affine / PseudoLabeler ground "
                    f"filter are gone; TerraSeg ground removal now runs for every mode).")
        raise ValueError(f'{e}{hint}') from None


def _warn_removed_config(cfg) -> None:
    """Old configs still parse; say loudly which of their sections are now ignored."""
    ignored = []
    for key in ('motion_compensation', 'obb_filter'):
        if getattr(cfg, key, None) is not None:
            ignored.append(key)
    _o = getattr(cfg, 'sam3d_objects', None)
    for key in ('hull_anchoring', 'pl_ground_inlier_thres'):
        if _o is not None and hasattr(_o, key):
            ignored.append(f'sam3d_objects.{key}')
    _f = getattr(cfg, 'lidar_filters', None)
    if _f is not None and hasattr(_f, 'max_range_m'):
        ignored.append('lidar_filters.max_range_m')
    if ignored:
        print('  [warn] config keys from removed features are IGNORED: ' + ', '.join(ignored))


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(
    cfg,
    frames: list,
    device: Optional[str] = None,
    checkpoint_dir: Optional[Path] = None,
    nusc=None,
    lidar_cache_dir: Optional[Path] = None,
    prepare_only: bool = False,
) -> tuple:
    """
    Run the full auto-labeling pipeline on a list of FrameRecords.

    Parameters
    ----------
    cfg             : config namespace (from load_config in run_pipeline.py)
    frames          : list of FrameRecord  (images NOT pre-loaded)
    device          : override device; defaults to 'cuda' if available
    checkpoint_dir  : directory for per-frame checkpoints; None = no checkpointing
    nusc            : NuScenes instance; required when lidar_aggregation.use_aggregation=true
    lidar_cache_dir : where the per-keyframe ground-free clouds are cached (shared across
                      cameras by run_multi_camera_pipeline). None = <checkpoint_dir>/_lidar when there
                      is a checkpoint_dir (so runs sharing a checkpoint folder also share the
                      TerraSeg results), else a temporary directory for this call only.
    prepare_only    : run only the mode-independent stages (SAM3, TerraSeg, PseudoLabeler, SAM3D Body
                      + B1/B2) into the checkpoints, then stop before SAM3D Objects and return
                      ({}, {}). Other runs that point at the same checkpoint folder then only pay for
                      SAM3D Objects.

    Returns
    -------
    body_results : {frame_idx: [pedestrian dicts with obb_* fields]}
    obj_results  : {frame_idx: [object dicts with obb_* fields]}
    """
    with contextlib.ExitStack() as stack:
        if lidar_cache_dir is None:
            if checkpoint_dir is not None:
                lidar_cache_dir = Path(checkpoint_dir) / '_lidar'
            else:
                lidar_cache_dir = Path(stack.enter_context(
                    tempfile.TemporaryDirectory(prefix='autolabel_lidar_')))
        return _run_pipeline(cfg, frames, device, checkpoint_dir, nusc, Path(lidar_cache_dir), prepare_only)


def _run_pipeline(cfg, frames, device, checkpoint_dir, nusc, lidar_cache_dir, prepare_only=False) -> tuple:
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    _warn_removed_config(cfg)

    dev_root     = Path(cfg.models.dev_root)
    sam3_ckpt    = cfg.models.sam3_ckpt or _hf_download_sam3()
    body_repo    = (Path(cfg.models.sam3d_body_repo) if cfg.models.sam3d_body_repo
                    else dev_root / 'models' / 'SAM3D' / 'sam-3d-body')
    obj_cfg_path = (Path(cfg.models.sam3d_obj_cfg) if cfg.models.sam3d_obj_cfg
                    else dev_root / 'models' / 'SAM3D' / 'sam-3d-objects' / 'checkpoints' / 'hf' / 'pipeline.yaml')
    obj_repo     = obj_cfg_path.parent.parent.parent

    # ── LiDAR: sweep aggregation + ego-body filter (the only pre-filter) ──────
    _agg_cfg  = getattr(cfg, 'lidar_aggregation', None)
    _use_agg  = bool(getattr(_agg_cfg, 'use_aggregation', False)) if _agg_cfg else False
    _n_before = int(getattr(_agg_cfg, 'n_before', 0)) if _agg_cfg else 0
    _n_after  = int(getattr(_agg_cfg, 'n_after',  0)) if _agg_cfg else 0
    if _use_agg and nusc is None:
        print('  [warn] lidar_aggregation.use_aggregation=true but nusc=None — '
              'falling back to single-sweep loading.')
        _use_agg = False
    if not _use_agg:
        _n_before = _n_after = 0

    _flt_cfg = getattr(cfg, 'lidar_filters', None)
    _ego_kw = dict(
        use_ego_body_filter=bool(getattr(_flt_cfg, 'use_ego_body_filter', True)) if _flt_cfg else True,
        ego_box_half_x=float(getattr(_flt_cfg, 'ego_box_half_x', 4.0)) if _flt_cfg else 4.0,
        ego_box_half_y=float(getattr(_flt_cfg, 'ego_box_half_y', 1.5)) if _flt_cfg else 1.5,
        ego_box_z_min=float(getattr(_flt_cfg, 'ego_box_z_min', 0.5)) if _flt_cfg else 0.5,
        ego_box_z_max=float(getattr(_flt_cfg, 'ego_box_z_max', 2.5)) if _flt_cfg else 2.5,
    )

    _gr_cfg         = getattr(cfg, 'ground_removal', None)
    _terraseg_ckpt  = getattr(_gr_cfg, 'terraseg_ckpt', None) if _gr_cfg else None

    # ── HDBSCAN params (shared by B1 and pointmap modes 2/3/4) ────────────────
    _obj_cfg     = getattr(cfg, 'sam3d_objects', None)
    _hdbscan_ns  = getattr(_obj_cfg, 'hdbscan', None)
    hdbscan_params = (
        {k: vars(v) for k, v in vars(_hdbscan_ns).items()}
        if _hdbscan_ns is not None else {}
    )

    # ── SAM3D Body: B1 / B2 are independent toggles ───────────────────────────
    _body_cfg = getattr(cfg, 'sam3d_body', None)
    _legacy_b = (getattr(_body_cfg, 'correction_mode', 'baseline') == 'b1_lidar_correction') \
                if _body_cfg else False          # old configs: one switch for both
    _b1_enabled = bool(getattr(_body_cfg, 'b1_depth_correction', _legacy_b)) if _body_cfg else False
    _b2_enabled = bool(getattr(_body_cfg, 'b2_ground_anchoring', _legacy_b)) if _body_cfg else False
    _body_label = 'B1 depth' if _b1_enabled else 'no B1'
    _body_label += ' + B2 ground anchoring' if _b2_enabled else ' + no B2'

    # ── SAM3D Objects: pointmap mode 1-11 ──────────────────────────────────────
    pointmap_mode = _resolve_mode(getattr(_obj_cfg, 'pointmap_mode', 'moge_affine_local'))
    _mode_no      = {v: k for k, v in POINTMAP_MODES.items()}[pointmap_mode]
    _proximity_min_pts = int(float(getattr(_obj_cfg, 'proximity_min_pts', 30)))
    _proximity_ratio   = float(getattr(_obj_cfg, 'proximity_ratio', 0.70))
    _cformer_drop_on_failure = bool(getattr(_obj_cfg, 'cformer_drop_on_failure', False))  # mode 4/10/11
    # Token-drop probe (analysis only): zero named SAM3D conditioning inputs after loading. Names are
    # 'pointmap'/'rgb_pointmap' (crop/full pointmap), 'image'/'rgb_image' (crop/full RGB), 'mask'/
    # 'rgb_image_mask' (crop/full mask) -- see SAM3DObjectsModel.set_force_drop. Part of _h_obj's hash
    # (vars(_obj_cfg) below) automatically, since it's just another sam3d_objects config key.
    _force_drop = getattr(_obj_cfg, 'force_drop_modalities', None)
    _force_drop = list(_force_drop) if _force_drop else None
    _ldcm_kw = dict(
        ldcm_repo=getattr(_obj_cfg, 'ldcm_repo', None),                     # mode 11
        ldcm_ckpt=getattr(_obj_cfg, 'ldcm_ckpt', None),                     # mode 11
        ldcm_moge_ckpt=getattr(_obj_cfg, 'ldcm_moge_ckpt', None),           # mode 11
        ldcm_utils3d_vendor=getattr(_obj_cfg, 'ldcm_utils3d_vendor', None),  # mode 11, see _load_ldcm
    )
    _mask_erode_px     = int(float(getattr(_obj_cfg, 'mask_erode_px',     0)))
    _mask_erode_min_px = int(float(getattr(_obj_cfg, 'mask_erode_min_px', 0)))
    _mesh_points = int(getattr(_obj_cfg, 'mesh_points', 0) or 0)   # 0 = keep full meshes + faces
    _bg_kw = dict(
        bg_pad_factor=float(getattr(_obj_cfg, 'bg_pad_factor', 1.5)),
        bg_min_pts=int(getattr(_obj_cfg, 'bg_min_pts', 20)),
        bg_exclude_px=int(getattr(_obj_cfg, 'bg_exclude_px', 5)),
        bg_feather_px=int(getattr(_obj_cfg, 'bg_feather_px', 2)),
        reg_slope_ratio=float(getattr(_obj_cfg, 'reg_slope_ratio', 2.0)),        # mode 8
        comp_grid=tuple(int(x) for x in getattr(_obj_cfg, 'comp_grid', (4, 3))),  # mode 7: (columns, rows)
        comp_min_pts=int(getattr(_obj_cfg, 'comp_min_pts', 30)),
        comp_margin=float(getattr(_obj_cfg, 'comp_margin', 0.25)),
    )

    # ── Stage directory names: each carries a hash of what it depends on ──────
    _th = getattr(cfg, 'thresholds', None)
    _h_sam3  = _hash(getattr(cfg, 'prompts', None), getattr(cfg, 'cross_class_dedup', None),
                     getattr(_th, 'sam3_score', None), getattr(_th, 'min_mask_px', None))
    _h_lidar = _hash(_n_before, _n_after, _ego_kw, _terraseg_ckpt)
    _h_body  = _hash(_h_lidar, _h_sam3, _body_cfg, hdbscan_params.get('pedestrian'),
                     getattr(_th, 'body_bbox', None), getattr(_th, 'sam3_iou_merge', None),
                     _b1_enabled, _b2_enabled)
    _h_obj   = _hash(_h_lidar, _h_sam3, {k: v for k, v in vars(_obj_cfg).items() if k != 'mesh_points'},
                     pointmap_mode)
    _st_sam3 = f'sam3__{_h_sam3}'
    _st_body = f'body__{_h_body}'
    _st_obj  = f'objects__{pointmap_mode}__{_h_obj}'
    ng_dir   = lidar_cache_dir / f'nonground__{_h_lidar}'

    print(f'\nLiDAR: aggregation {"±%d/%d sweeps" % (_n_before, _n_after) if _use_agg else "off (anchor sweep only)"}'
          f' | ego-body filter {"on" if _ego_kw["use_ego_body_filter"] else "off"}'
          f' | TerraSeg ground removal per sweep before aggregation')
    print(f'SAM3D Body: {_body_label}')
    print(f'SAM3D Objects: pointmap mode {_mode_no} ({pointmap_mode})'
          + (f'  [force-drop: {_force_drop}]' if _force_drop else ''))
    print(f'  checkpoint stage: {_st_obj}')

    def _ng_path(frame) -> Path:
        return ng_dir / f'{frame.sample_token}.npy'

    def _load_ng(frame) -> Optional[np.ndarray]:
        """Ground-free aggregated cloud for this keyframe (None = frame has no LiDAR)."""
        p = _ng_path(frame)
        return np.load(p).astype(np.float64) if p.exists() else None

    def _load_pts_full(frame) -> Optional[np.ndarray]:
        """FULL aggregated cloud (ground included) -- PseudoLabeler, B2 RANSAC, mode-6 background fit."""
        if frame.lidar_path is None:
            return None
        return load_lidar_pts_aggregated(nusc, frame, _n_before, _n_after, **_ego_kw)

    # sam3_mem is only populated when checkpointing is disabled (fits in RAM).
    sam3_mem = {}

    # ── SAM3 segmentation ─────────────────────────────────────────────────────
    print('\n[SAM3 segmentation]')
    pending = []
    for i, frame in enumerate(frames):
        p = _ckpt_path(checkpoint_dir, _st_sam3, i)
        if p is not None and p.exists():
            # Validate now — corrupt SAM3 checkpoints would otherwise only be
            # discovered mid-way through the body/objects stage when the SAM3
            # model is no longer loaded, making recovery impossible.
            if _load(p) is None:
                pending.append((i, frame))  # corrupt, will be re-processed
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
            p = _ckpt_path(checkpoint_dir, _st_sam3, i)
            if p:
                _save(p, _sam3_to_ckpt(result))
            else:
                sam3_mem[i] = result
        segmentor.unload()
    else:
        print(f'  All {len(frames)} frame(s) loaded from cache.')

    _print_gpu(device)

    # Which frames still need body / objects work (validated by actually loading).
    def _pending_set(stage: str) -> set:
        out = set()
        for i in range(len(frames)):
            p = _ckpt_path(checkpoint_dir, stage, i)
            if p is None or not p.exists() or _load(p) is None:
                out.add(i)
        return out

    _body_pending = _pending_set(_st_body)
    _obj_pending  = _pending_set(_st_obj)

    # ── TerraSeg ground removal — per sweep, before aggregation ───────────────
    _ng_needed = (_body_pending if _b1_enabled else set()) | _obj_pending
    _ng_todo   = [i for i in sorted(_ng_needed)
                  if frames[i].lidar_path is not None and not _ng_path(frames[i]).exists()]
    print('\n[TerraSeg ground removal — per sweep, before aggregation]')
    if _ng_todo:
        n_cached_ng = len([i for i in _ng_needed if frames[i].lidar_path is not None]) - len(_ng_todo)
        if n_cached_ng:
            print(f'  {n_cached_ng} keyframe cloud(s) already cached (shared across cameras).')
        ts_filter = TerraSegGroundFilter(dev_root, ckpt_path=_terraseg_ckpt)
        bar = tqdm(_ng_todo, desc='TerraSeg', unit='frame')
        for i in bar:
            frame = frames[i]
            bar.set_postfix_str(f'{frame.scene_name}  frame {frame.frame_idx}')
            pts = load_lidar_pts_nonground_aggregated(
                nusc, frame, _n_before, _n_after, ts_filter, **_ego_kw)
            if pts is not None:
                _save_npy_atomic(_ng_path(frame), pts.astype(np.float32))
        ts_filter.unload()
    else:
        print('  Nothing to do (cached, or no frame needs LiDAR).')

    _print_gpu(device)

    # ── PseudoLabeler ground-surface fit (B2 only) — on the FULL cloud ────────
    # Fits gθ: (x,y)→z on the aggregated cloud INCLUDING ground for each frame.
    # State_dicts are lightweight (~10 KB each) so all frames fit in RAM.
    pl_states: Dict[int, Optional[dict]] = {i: None for i in range(len(frames))}
    if _b2_enabled:
        print('\n[PseudoLabeler ground estimation]')

        # SAM3 loading leaves BF16 autocast globally enabled.  Training PseudoLabeler
        # in BF16 causes weight saturation (±2.7 vs expected ±0.1–0.5), producing
        # z_ground predictions of 100,000+ m instead of ≈–0.8 m and completely wrong
        # pedestrian positions.
        if torch.is_autocast_enabled():
            torch.autocast('cuda', enabled=False).__enter__()
            print('  Disabled inherited BF16 autocast.')

        # If a cache file exists for this run and covers all frames, skip refitting.
        # Delete <checkpoint_dir>/pl_states__<hash>.pt to force a refit.
        _pl_cache_path = (Path(checkpoint_dir) / f'pl_states__{_h_lidar}.pt'
                          if checkpoint_dir else None)
        _pl_loaded = False
        if _pl_cache_path is not None and _pl_cache_path.exists():
            try:
                _cached = torch.load(_pl_cache_path, map_location='cpu')
                if isinstance(_cached, dict) and len(_cached) == len(frames):
                    pl_states = _cached
                    _pl_loaded = True
                    print(f'  Loaded cached PseudoLabeler states for {len(frames)} frame(s) '
                          f'from {_pl_cache_path}.')
                else:
                    print(f'  Cache size mismatch ({len(_cached)} vs {len(frames)} frames) '
                          f'— refitting.')
            except Exception as _e:
                print(f'  Cache load failed ({_e}) — refitting.')

        if not _pl_loaded:
            n_skip = len(frames) - len(_body_pending)
            if n_skip:
                print(f'  Skipping {n_skip} fully-cached frame(s).')
            for i, frame in enumerate(frames):
                if i not in _body_pending or frame.lidar_path is None:
                    continue
                pts_full = _load_pts_full(frame)
                if pts_full is None or len(pts_full) == 0:
                    print(f'  [{frame.scene_name} frame {frame.frame_idx}] no LiDAR — skipping.')
                    continue
                pl_model = _fit_pseudolabeler(pts_full, device, dev_root)
                pl_states[i] = ({k: v.cpu() for k, v in pl_model.state_dict().items()}
                                if pl_model is not None else None)

            if _pl_cache_path is not None and pl_states:
                try:
                    torch.save(pl_states, _pl_cache_path)
                    print(f'  Saved PseudoLabeler cache → {_pl_cache_path}')
                except Exception as _e:
                    print(f'  Warning: could not save PseudoLabeler cache: {_e}')

        _print_gpu(device)

    # ── SAM3D Body ────────────────────────────────────────────────────────────
    print(f'\n[SAM3D Body — {_body_label}]')
    body_results = {}
    pending = []
    for i, frame in enumerate(frames):
        p = _ckpt_path(checkpoint_dir, _st_body, i)
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
            frame_sam3 = _get_sam3(i, sam3_mem, checkpoint_dir, _st_sam3)
            ped_dets   = frame_sam3.get('pedestrian', [])
            result     = body_model.run_frame(frame, ped_dets)

            if result and frame.lidar_path is not None:
                # B1: depth correction on the GROUND-FREE cloud
                if _b1_enabled:
                    pts_ng = _load_ng(frame)
                    if pts_ng is not None:
                        pts_ego_vis, u_vis, v_vis, Z_vis, H, W = project_lidar_to_camera(frame, pts_ng)
                        _apply_b1_depth_correction(
                            result, frame, ped_dets,
                            pts_ego_vis, u_vis, v_vis, Z_vis, H, W,
                            hdbscan_kwargs=hdbscan_params.get('pedestrian'),
                        )
                # B2: ground anchoring -- needs the FULL cloud (RANSAC fallback needs ground)
                if _b2_enabled:
                    pts_full = _load_pts_full(frame)
                    if pts_full is not None:
                        pl_model = _restore_pseudolabeler(pl_states.get(i), device, dev_root)
                        _apply_b2_ground_anchoring(result, frame, pts_full, pl_model, device)

            body_results[i] = result
            p = _ckpt_path(checkpoint_dir, _st_body, i)
            if p:
                _save(p, _body_to_ckpt(result))
        body_model.unload()
    else:
        print(f'  All {len(frames)} frame(s) loaded from cache.')

    _print_gpu(device)

    if prepare_only:
        print('\nprepare-only: SAM3, TerraSeg, PseudoLabeler and SAM3D Body are cached; stopping before SAM3D Objects.')
        return body_results, {}

    # ── SAM3D Objects ─────────────────────────────────────────────────────────
    print(f'\n[SAM3D Objects — pointmap mode {_mode_no}: {pointmap_mode}]')
    obj_results = {}
    pending = []
    for i, frame in enumerate(frames):
        p = _ckpt_path(checkpoint_dir, _st_obj, i)
        if p is not None and p.exists():
            data = _load(p)
            if data is not None:
                obj_results[i] = slim_object_results(_obj_from_ckpt(data), _mesh_points)
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
            hdbscan_params=hdbscan_params,
            cformer_ckpt=getattr(_obj_cfg, 'cformer_ckpt', None),
            lidar_lines=getattr(_obj_cfg, 'lidar_lines', 32),
            proximity_min_pts=_proximity_min_pts,
            proximity_ratio=_proximity_ratio,
            cformer_drop_on_failure=_cformer_drop_on_failure,
            **_ldcm_kw,
            mask_erode_px=_mask_erode_px,
            mask_erode_min_px=_mask_erode_min_px,
            **_bg_kw,
        )
        obj_model.load()
        if _force_drop:
            obj_model.set_force_drop(_force_drop)
            print(f'  [probe] zeroed conditioning input(s): {_force_drop}')
        bar = tqdm(pending, total=len(frames), initial=len(frames) - len(pending),
                   desc='SAM3D Objects', unit='frame')
        for i, frame in bar:
            bar.set_postfix_str(f'{frame.scene_name}  frame {frame.frame_idx}')
            frame_sam3 = _get_sam3(i, sam3_mem, checkpoint_dir, _st_sam3)
            pts_ng = _load_ng(frame) if frame.lidar_path is not None else None
            pts_full = (_load_pts_full(frame) if pointmap_mode in _FULL_CLOUD_MODES
                        and frame.lidar_path is not None else None)
            result = obj_model.run_frame(frame, frame_sam3, pts_ego=pts_ng, pts_ego_full=pts_full)
            attach_full_obbs(result, frame)              # OBB from the FULL mesh, kept in the checkpoint
            result = slim_object_results(result, _mesh_points)
            obj_results[i] = result
            p = _ckpt_path(checkpoint_dir, _st_obj, i)
            if p:
                _save(p, _obj_to_ckpt(result))
        obj_model.unload()
    else:
        print(f'  All {len(frames)} frame(s) loaded from cache.')

    _print_gpu(device)

    # ── Orientation + OBB + rider merge (CPU) — no OBB filters ────────────────
    print('\nOrientation estimation + OBB fitting...')
    for i, frame in enumerate(frames):
        _postprocess_frame(frame, body_results[i], obj_results[i])
        body_results[i], obj_results[i] = _merge_rider_obbs(
            frame, body_results[i], obj_results[i]
        )
    print('  Done.')

    return body_results, obj_results


def run_multi_camera_pipeline(
    cfg,
    frames_per_cam: dict,
    device: Optional[str] = None,
    checkpoint_dir: Optional[Path] = None,
    nusc=None,
) -> tuple:
    """
    Run the full pipeline for every camera in frames_per_cam, then apply
    cross-camera duplicate suppression.

    Parameters
    ----------
    cfg            : config namespace
    frames_per_cam : {camera_name: [FrameRecord]}  — same keyframe ordering
    device         : override device; defaults to 'cuda' if available
    checkpoint_dir : base dir for per-frame checkpoints; camera-specific
                     sub-directories are created automatically (<dir>/<cam>/)
    nusc           : NuScenes instance (required when lidar_aggregation enabled)

    The ground-free LiDAR cloud is a property of the keyframe, not the camera, so it is
    computed once and cached in <checkpoint_dir>/_lidar (or a temp dir when checkpointing
    is off) and reused by every camera.

    Returns
    -------
    body_results_all : {cam: {frame_list_idx: [body dicts]}}
    obj_results_all  : {cam: {frame_list_idx: [obj  dicts]}}
    """
    body_results_all: Dict[str, dict] = {}
    obj_results_all:  Dict[str, dict] = {}

    with contextlib.ExitStack() as stack:
        if checkpoint_dir:
            lidar_cache = Path(checkpoint_dir) / '_lidar'
        else:
            lidar_cache = Path(stack.enter_context(
                tempfile.TemporaryDirectory(prefix='autolabel_lidar_')))

        for cam, frames in frames_per_cam.items():
            print(f'\n{"=" * 60}')
            print(f'Camera: {cam}  ({len(frames)} frame(s))')
            print(f'{"=" * 60}')
            cam_ckpt = (Path(checkpoint_dir) / cam) if checkpoint_dir else None
            body_r, obj_r = run_pipeline(
                cfg, frames, device=device, checkpoint_dir=cam_ckpt, nusc=nusc,
                lidar_cache_dir=lidar_cache,
            )
            body_results_all[cam] = body_r
            obj_results_all[cam]  = obj_r

    # Cross-camera duplicate suppression
    _xc_cfg = getattr(cfg, 'cross_camera_merge', None)
    if _xc_cfg and getattr(_xc_cfg, 'enabled', False):
        print('\n[Cross-camera merge]')
        from .cross_camera_merge import cross_camera_merge
        body_results_all, obj_results_all = cross_camera_merge(
            body_results_all, obj_results_all, frames_per_cam, _xc_cfg,
        )
    else:
        print('\nCross-camera merge: disabled — skipped.')

    return body_results_all, obj_results_all


def _print_gpu(device: str) -> None:
    if device == 'cuda':
        free, total = torch.cuda.mem_get_info()
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f'  GPU free: {free / 1024**3:.1f} GB of {total / 1024**3:.1f} GB ({torch.cuda.get_device_name(0)})'
              f' | peak allocated by this stage: {peak:.1f} GB')
        torch.cuda.reset_peak_memory_stats()


def _hf_download_sam3() -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id='facebook/sam3', filename='sam3.pt')
