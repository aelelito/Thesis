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
1. SAM3 segmentation           (all frames, GPU)
2. SAM3D Body + B1 correction  (all frames, GPU + CPU)
   - Stage 1: LiDAR tz correction (HDBSCAN per pedestrian)
   - Stage 2: Ground anchoring (PseudoLabeler MLP fitted per frame)
3. SAM3D Objects + MoGe        (all frames, GPU)
4. Orientation + OBB           (CPU, per frame)
"""
import gzip
import zlib
import os
import pickle
import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from sklearn.linear_model import RANSACRegressor
from tqdm import tqdm

from .fitting.obb import compute_obb_gravity_aligned, compute_obb_pedestrian
from .models.sam3_segmentor import SAM3Segmentor
from .models.sam3d_body import SAM3DBodyModel
from .models.sam3d_objects import SAM3DObjectsModel
from .orientation.pedestrian import facing_direction
from .utils.lidar import (
    filter_inmask_lidar_hdbscan,
    filter_lidar_pts,
    load_lidar_pts,
    load_lidar_pts_aggregated,
    project_lidar_to_camera,
)


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
    _pl_path = str(dev_root / 'Models' / 'TerraSeg' / 'PseudoLabeler_scripts')
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
    _pl_path = str(dev_root / 'Models' / 'TerraSeg' / 'PseudoLabeler_scripts')
    if _pl_path not in sys.path:
        sys.path.insert(0, _pl_path)
    from pseudolabeler_model import PseudoLabeler
    _dev     = torch.device(device)
    pl_model = PseudoLabeler().to(_dev)
    pl_model.load_state_dict({k: v.to(_dev) for k, v in state_dict.items()})
    pl_model.eval()
    return pl_model


# ── B1 — depth correction helpers ────────────────────────────────────────────

def _best_sam3_mask(bbox, ped_dets,
                    iou_thresh: float = 0.3) -> Optional[np.ndarray]:
    """Return highest-IoU SAM3 mask for bbox, or None if below threshold."""
    x1, y1, x2, y2 = bbox
    best_mask, best_iou = None, iou_thresh
    for d in ped_dets:
        ys, xs = np.where(d['binary_mask'])
        if len(xs) == 0:
            continue
        mx1, my1 = float(xs.min()), float(ys.min())
        mx2, my2 = float(xs.max()), float(ys.max())
        inter = max(0., min(x2, mx2) - max(x1, mx1)) * max(0., min(y2, my2) - max(y1, my1))
        iou   = inter / ((x2-x1)*(y2-y1) + (mx2-mx1)*(my2-my1) - inter + 1e-8)
        if iou > best_iou:
            best_iou, best_mask = iou, d['binary_mask']
    return best_mask


def _apply_b1_depth_correction(body_results: list, frame, ped_dets: list,
                                pts_ego_vis, u_vis, v_vis, Z_vis,
                                H: int, W: int,
                                hdbscan_kwargs: dict = None) -> None:
    """
    Stage 1: override tz for each pedestrian with HDBSCAN-filtered median LiDAR depth.
    Recomputes tx, ty from mask centroid. Shifts vertices by delta; cam_t updated.
    joints_3d are body-relative and are NOT shifted.
    Modifies body_results in-place.
    """
    fx, fy, cx, cy = frame.K[0, 0], frame.K[1, 1], frame.K[0, 2], frame.K[1, 2]
    u_int = np.round(u_vis).astype(int).clip(0, W - 1)
    v_int = np.round(v_vis).astype(int).clip(0, H - 1)

    for i, r in enumerate(body_results):
        bbox = r.get('bbox')
        mask = _best_sam3_mask(bbox, ped_dets) if bbox is not None else None

        if mask is None:
            if bbox is not None:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                mask = np.zeros((H, W), dtype=bool)
                mask[max(0, y1):min(H, y2+1), max(0, x1):min(W, x2+1)] = True
                mask_src = 'bbox_rect'
            else:
                r['b1_mode'] = 'no_bbox'
                continue
        else:
            mask_src = 'sam3_mask'

        in_mask    = mask[v_int, u_int]
        pts_inmask = pts_ego_vis[in_mask]
        Z_inmask   = Z_vis[in_mask]

        if len(pts_inmask) == 0:
            r['b1_mode'] = 'no_lidar'
            continue

        keep    = filter_inmask_lidar_hdbscan(pts_inmask, **(hdbscan_kwargs or {}))
        tz_pred = float(r['cam_t'][2])

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


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(
    cfg,
    frames: list,
    device: Optional[str] = None,
    checkpoint_dir: Optional[Path] = None,
    nusc=None,
) -> tuple:
    """
    Run the full auto-labeling pipeline on a list of FrameRecords.

    Parameters
    ----------
    cfg            : config namespace (from load_config in run_pipeline.py)
    frames         : list of FrameRecord  (images NOT pre-loaded)
    device         : override device; defaults to 'cuda' if available
    checkpoint_dir : directory for per-frame checkpoints; None = no checkpointing
    nusc           : NuScenes instance; required when lidar_aggregation.use_aggregation=true

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
                    else dev_root / 'Models' / 'SAM3D' / 'sam-3d-body')
    obj_cfg_path = (Path(cfg.models.sam3d_obj_cfg) if cfg.models.sam3d_obj_cfg
                    else dev_root / 'Models' / 'SAM3D' / 'sam-3d-objects' / 'checkpoints' / 'hf' / 'pipeline.yaml')
    obj_repo     = obj_cfg_path.parent.parent.parent

    # ── LiDAR aggregation config ──────────────────────────────────────────────
    _agg_cfg      = getattr(cfg, 'lidar_aggregation', None)
    _use_agg      = bool(getattr(_agg_cfg, 'use_aggregation', False)) if _agg_cfg else False
    _n_before     = int(getattr(_agg_cfg, 'n_before', 0)) if _agg_cfg else 0
    _n_after      = int(getattr(_agg_cfg, 'n_after',  0)) if _agg_cfg else 0
    if _use_agg and nusc is None:
        print('  [warn] lidar_aggregation.use_aggregation=true but nusc=None — '
              'falling back to single-sweep loading.')
        _use_agg = False

    # ── LiDAR point cloud pre-filters ─────────────────────────────────────────
    _flt_cfg            = getattr(cfg, 'lidar_filters', None)
    _use_ego_filter     = bool(getattr(_flt_cfg, 'use_ego_body_filter', True))  if _flt_cfg else True
    _ego_box_half_x     = float(getattr(_flt_cfg, 'ego_box_half_x',     4.0))  if _flt_cfg else 4.0
    _ego_box_half_y     = float(getattr(_flt_cfg, 'ego_box_half_y',     1.5))  if _flt_cfg else 1.5
    _ego_box_z_min      = float(getattr(_flt_cfg, 'ego_box_z_min',      0.5))  if _flt_cfg else 0.5
    _ego_box_z_max      = float(getattr(_flt_cfg, 'ego_box_z_max',      2.5))  if _flt_cfg else 2.5
    _max_range_m        = float(getattr(_flt_cfg, 'max_range_m',        52.0)) if _flt_cfg else 52.0

    def _load_pts_ego(frame) -> 'Optional[np.ndarray]':
        """Load ego-frame point cloud for a frame (aggregated or single sweep),
        then apply ego-body exclusion and max-range pre-filters."""
        if _use_agg:
            pts = load_lidar_pts_aggregated(nusc, frame, _n_before, _n_after)
        else:
            pts = load_lidar_pts(frame)
        if pts is not None:
            pts = filter_lidar_pts(
                pts,
                use_ego_body_filter=_use_ego_filter,
                ego_box_half_x=_ego_box_half_x,
                ego_box_half_y=_ego_box_half_y,
                ego_box_z_min=_ego_box_z_min,
                ego_box_z_max=_ego_box_z_max,
                max_range_m=_max_range_m,
            )
        return pts

    # ── HDBSCAN params (shared by B1 body correction and O3 objects) ─────────
    _hdbscan_ns    = getattr(getattr(cfg, 'sam3d_objects', None), 'hdbscan', None)
    hdbscan_params = (
        {k: vars(v) for k, v in vars(_hdbscan_ns).items()}
        if _hdbscan_ns is not None else {}
    )

    # ── SAM3D Body config ─────────────────────────────────────────────────────
    _body_cfg        = getattr(cfg, 'sam3d_body', None)
    _b1_enabled      = (getattr(_body_cfg, 'correction_mode', 'baseline') == 'b1_lidar_correction') \
                       if _body_cfg else False
    _b2_enabled      = _b1_enabled   # Stage 2 always runs together with B1
    _body_mode_label = 'B1 — LiDAR depth + PseudoLabeler ground anchoring' \
                       if _b1_enabled else 'baseline'

    # ── SAM3D Objects / PseudoLabeler config ──────────────────────────────────
    _lidar_cfg_pre       = getattr(cfg, 'sam3d_objects', None)
    _pointmap_mode_pre   = getattr(_lidar_cfg_pre, 'pointmap_mode', 'baseline')
    _pl_ground_inlier    = float(getattr(_lidar_cfg_pre, 'pl_ground_inlier_thres', 0.10))
    _proximity_min_pts   = int(float(getattr(_lidar_cfg_pre, 'proximity_min_pts', 30)))
    _proximity_ratio     = float(getattr(_lidar_cfg_pre, 'proximity_ratio', 0.70))
    _hull_anchoring      = bool(getattr(_lidar_cfg_pre, 'hull_anchoring', False))
    # PseudoLabeler is needed when B2 ground anchoring is active OR when O4 uses
    # ground filtering before CompletionFormer sparse anchors.
    # O5 does NOT need PseudoLabeler — per-mask HDBSCAN handles ground clutter implicitly.
    _needs_pseudolabeler = _b2_enabled or (_pointmap_mode_pre == 'o4_ground_filter')

    # sam3_mem is only populated when checkpointing is disabled (fits in RAM).
    # When checkpointing is enabled, each stage loads results per-frame from disk.
    sam3_mem = {}

    # pl_states: frame index → PseudoLabeler state_dict (or None if fitting failed /
    # not needed).  Populated in the pre-fitting pass below, consumed by both the
    # Body (B2 foot anchoring) and Objects (O4 ground filtering) stages.
    pl_states: Dict[int, Optional[dict]] = {}

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

    # ── PseudoLabeler ground estimation (once per frame, shared by Body + Objects) ──
    # Fits a small MLP gθ: R²→R on the aggregated LiDAR cloud for each frame.
    # Runs before both Body and Objects stages so the result can be reused by:
    #   B2 — foot anchoring: query pl_model(ped_xy) → z_ground, shift mesh feet
    #   O4 — ground filtering: get_ground_bool() removes road-surface LiDAR before CFormer
    # State_dicts are lightweight (~10 KB each) so all frames fit in RAM.
    if _needs_pseudolabeler:
        print('\n[PseudoLabeler ground estimation]')

        # ── PseudoLabeler checkpoint ──────────────────────────────────────────
        # If a cache file exists for this run and covers all frames, skip refitting.
        # Delete <checkpoint_dir>/pl_states_cache.pt to force a refit.
        _pl_cache_path = (Path(checkpoint_dir) / 'pl_states_cache.pt'
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
            # Fit only for frames not yet fully cached in both downstream stages.
            _body_pending  = {i for i in range(len(frames))
                              if _ckpt_path(checkpoint_dir, 'body', i) is None
                              or not _ckpt_path(checkpoint_dir, 'body', i).exists()}
            _obj_pending   = {i for i in range(len(frames))
                              if _ckpt_path(checkpoint_dir, 'objects', i) is None
                              or not _ckpt_path(checkpoint_dir, 'objects', i).exists()}
            _pl_needed_for = ((_body_pending if _b2_enabled else set()) |
                              (_obj_pending  if _pointmap_mode_pre == 'o4_ground_filter' else set()))

            n_cached_pl = len(frames) - len(_pl_needed_for)
            if n_cached_pl:
                print(f'  Skipping {n_cached_pl} fully-cached frame(s).')

            for i, frame in enumerate(frames):
                if i not in _pl_needed_for or frame.lidar_path is None:
                    pl_states[i] = None
                    continue
                pts_ego = _load_pts_ego(frame)
                if pts_ego is None or len(pts_ego) == 0:
                    pl_states[i] = None
                    print(f'  [{frame.scene_name} frame {frame.frame_idx}] no LiDAR — skipping.')
                    continue
                pl_model = _fit_pseudolabeler(pts_ego, device, dev_root)
                pl_states[i] = ({k: v.cpu() for k, v in pl_model.state_dict().items()}
                                if pl_model is not None else None)

            # Save cache so subsequent runs can skip refitting.
            if _pl_cache_path is not None and pl_states:
                try:
                    torch.save(pl_states, _pl_cache_path)
                    print(f'  Saved PseudoLabeler cache → {_pl_cache_path}')
                except Exception as _e:
                    print(f'  Warning: could not save PseudoLabeler cache: {_e}')
    else:
        pl_states = {i: None for i in range(len(frames))}

    _print_gpu(device)

    # ── SAM3D Body ────────────────────────────────────────────────────────────
    print(f'\n[SAM3D Body — {_body_mode_label}]')
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
        # # ── [DEBUG] Ground anchoring stats ────────────────────────────────────
        # _gnd_stats: list = []
        # # ──────────────────────────────────────────────────────────────────────
        bar = tqdm(pending, total=len(frames), initial=len(frames) - len(pending),
                   desc='SAM3D Body', unit='frame')
        for i, frame in bar:
            bar.set_postfix_str(f'{frame.scene_name}  frame {frame.frame_idx}')
            frame_sam3 = _get_sam3(i, sam3_mem, checkpoint_dir)
            ped_dets   = frame_sam3.get('pedestrian', [])
            result     = body_model.run_frame(frame, ped_dets)

            # ── B1: depth correction + ground anchoring ───────────────────────
            if _b1_enabled and result and frame.lidar_path is not None:
                pts_ego = _load_pts_ego(frame)
                if pts_ego is not None:
                    pts_ego_vis, u_vis, v_vis, Z_vis, H, W = project_lidar_to_camera(
                        frame, pts_ego
                    )
                    _apply_b1_depth_correction(
                        result, frame, ped_dets,
                        pts_ego_vis, u_vis, v_vis, Z_vis, H, W,
                        hdbscan_kwargs=hdbscan_params.get('pedestrian'),
                    )
                    if _b2_enabled:
                        # Reconstruct PseudoLabeler from pre-fitted state_dict
                        pl_model = _restore_pseudolabeler(pl_states.get(i), device, dev_root)
                        _apply_b2_ground_anchoring(
                            result, frame, pts_ego, pl_model, device,
                            # _stats=_gnd_stats,   # [DEBUG]
                        )
            # ─────────────────────────────────────────────────────────────────

            body_results[i] = result
            p = _ckpt_path(checkpoint_dir, 'body', i)
            if p:
                _save(p, _body_to_ckpt(result))
        body_model.unload()
        # # ── [DEBUG] Write ground anchoring stats ───────────────────────────────
        # if _gnd_stats and getattr(cfg, 'output_dir', None) is not None:
        #     import json
        #     _stats_path = (checkpoint_dir.parent if checkpoint_dir is not None
        #                    else Path(cfg.output_dir)) / 'ground_anchoring_stats.json'
        #     _stats_path.parent.mkdir(parents=True, exist_ok=True)
        #     with open(_stats_path, 'w') as _f:
        #         json.dump(_gnd_stats, _f, indent=2)
        #     print(f'  [DEBUG] Ground anchoring stats → {_stats_path}  ({len(_gnd_stats)} entries)')
        # # ──────────────────────────────────────────────────────────────────────
    else:
        print(f'  All {len(frames)} frame(s) loaded from cache.')

    _print_gpu(device)

    # ── SAM3D Objects ─────────────────────────────────────────────────────────
    _lidar_cfg    = _lidar_cfg_pre   # already read above
    pointmap_mode = _pointmap_mode_pre
    _mode_labels = {
        'baseline':        'MoGe baseline (non-metric)',
        'o1_lidar':        'O1 — sparse LiDAR pointmap',
        'o2_moge_affine':  'O2 — MoGe + global affine calibration',
        'o3_local_affine': 'O3 — MoGe + per-object local affine calibration',
        'o4_ground_filter':  'O4 — CompletionFormer + global ground filter',
        'o5_mask_hdbscan':   'O5 — CompletionFormer + per-mask HDBSCAN',
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
        cformer_ckpt  = getattr(_lidar_cfg, 'cformer_ckpt', None)
        lidar_lines   = getattr(_lidar_cfg, 'lidar_lines', 32)
        _ss_correction = bool(getattr(_lidar_cfg, 'ss_correction', False))  # THESIS: SS Correction flag
        obj_model = SAM3DObjectsModel(
            repo_path=str(obj_repo),
            config_path=str(obj_cfg_path),
            prompts=vars(cfg.prompts),
            device=device,
            pointmap_mode=pointmap_mode,
            hdbscan_params=hdbscan_params,
            cformer_ckpt=cformer_ckpt,
            lidar_lines=lidar_lines,
            pl_ground_inlier_thres=_pl_ground_inlier,
            ss_correction=_ss_correction,  # THESIS: SS Correction flag
            proximity_min_pts=_proximity_min_pts,
            proximity_ratio=_proximity_ratio,
            hull_anchoring=_hull_anchoring,
        )
        obj_model.load()
        bar = tqdm(pending, total=len(frames), initial=len(frames) - len(pending),
                   desc='SAM3D Objects', unit='frame')
        for i, frame in bar:
            bar.set_postfix_str(f'{frame.scene_name}  frame {frame.frame_idx}')
            frame_sam3 = _get_sam3(i, sam3_mem, checkpoint_dir)
            pts_ego  = _load_pts_ego(frame) if frame.lidar_path is not None else None
            pl_model = (_restore_pseudolabeler(pl_states.get(i), device, dev_root)
                        if pointmap_mode == 'o4_ground_filter' else None)
            result = obj_model.run_frame(frame, frame_sam3, pts_ego=pts_ego, pl_model=pl_model)
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
