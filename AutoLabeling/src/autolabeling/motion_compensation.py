"""
Per-object SE(2) ICP motion compensation.

Two-phase pipeline per SAM3 mask:
  Phase 1 — HDBSCAN tracking (no alignment):
      Walk sweeps outward from anchor in both directions.  For each sweep,
      crop a ROI around the propagated search centre, Z-filter with TerraSeg
      non-ground labels, run HDBSCAN, pick the cluster nearest the search centre.
      Output: per-sweep cluster points + centroids.

  Phase 2 — SE(2) ICP with growing target, alternating fwd/bwd order:
      Process order t+1, t-1, t+2, t-2 (zip_longest) so temporally close
      sweeps align first.  Each sweep is initialised with a centroid-to-centroid
      translation (from Phase 1) and refined by open3d ICP on a FLAT (z=0)
      projection — SE(2) constraint.  The ICP target grows as each aligned
      sweep is appended: later sweeps see a richer shape → better correspondences.

Per-mask outputs:
  pts_comp_all  : (N, 3) float32  final ego-frame point cloud for the object
  is_dynamic    : bool
  velocity_mps  : float   (0.0 when static)
  heading_rad   : float   (motion heading when dynamic, PCA long-axis when static,
                           nan when undefined)
  is_turning    : bool

Ground removal:
  TerraSeg-S (PointTransformerV3) assigns binary ground/non-ground labels to
  every LiDAR point.  When TerraSeg is not available the module falls back to a
  simple Z-threshold filter (z > z_ground_min).
"""
from __future__ import annotations

import sys
from itertools import zip_longest
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import hdbscan as _hdbscan_mod
except ImportError:
    _hdbscan_mod = None

try:
    import open3d as _o3d
except ImportError:
    _o3d = None

try:
    import cv2 as _cv2
except ImportError:
    _cv2 = None

from .utils.lidar import filter_inmask_lidar_hdbscan


# ── Class-specific parameters ─────────────────────────────────────────────────

CLASS_MAX_HEIGHT_M: Dict[str, float] = {
    'car': 2.2, 'van': 2.7, 'truck': 4.5, 'bus': 5.0,
    'motorcycle': 2.2, 'bicycle': 2.5, 'pedestrian': 2.5,
    'construction vehicle': 5.0, 'construction_vehicle': 5.0,
    'trailer': 4.0,
}
_DEFAULT_MAX_HEIGHT_M = 3.5

CLASS_ICP_METRIC: Dict[str, str] = {
    # point-to-plane for flat-surface classes (side walls, roof planes → good normals)
    'car': 'p2l', 'van': 'p2l', 'truck': 'p2l', 'bus': 'p2l',
    'construction vehicle': 'p2l', 'construction_vehicle': 'p2l',
    'trailer': 'p2l',
    # point-to-point for 3D-structured classes (no dominant flat surface)
    'motorcycle': 'p2p', 'bicycle': 'p2p', 'pedestrian': 'p2p',
}
_DEFAULT_ICP_METRIC = 'p2p'

CLASS_SEARCH_RADIUS_M: Dict[str, float] = {
    'car': 2.5, 'truck': 3.5, 'bus': 4.0, 'van': 3.0,
    'motorcycle': 1.5, 'bicycle': 1.2, 'pedestrian': 1.0,
    'construction vehicle': 4.0, 'construction_vehicle': 4.0,
    'trailer': 4.0,
}
_DEFAULT_RADIUS_M = 2.5

CLASS_MAX_SPEED_MPS: Dict[str, float] = {
    'car': 20.0, 'truck': 15.0, 'bus': 15.0, 'van': 18.0,
    'motorcycle': 25.0, 'bicycle': 8.0, 'pedestrian': 3.0,
    'construction vehicle': 10.0, 'construction_vehicle': 10.0,
    'trailer': 15.0,
}
_DEFAULT_MAX_SPEED = 15.0

CLASS_DYNAMIC_THRESH_MPS: Dict[str, float] = {
    'car': 2.5, 'truck': 2.5, 'bus': 2.5, 'van': 2.5,
    'motorcycle': 1.5, 'bicycle': 1.0, 'pedestrian': 0.8,
    'construction vehicle': 2.5, 'construction_vehicle': 2.5,
    'trailer': 2.5,
}
_DEFAULT_DYNAMIC_THRESH = 2.5

# ── Numeric constants ─────────────────────────────────────────────────────────
_MIN_PTS_HDBSCAN   = 2      # min points to attempt HDBSCAN
_MIN_PTS_ICP       = 8      # min points to attempt full ICP (else centroid-only)
_Z_ANCHOR_FLOOR_SLACK = 0.20  # extra slack below anchor z_min for TerraSeg artefacts


# ─────────────────────────────────────────────────────────────────────────────
# TerraSeg ground filter
# ─────────────────────────────────────────────────────────────────────────────

class TerraSegGroundFilter:
    """
    Wraps TerraSegPredictor for batch per-sweep ground removal.

    Parameters
    ----------
    dev_root    : Path to <dev_root> (parent of Models/)
    variant     : 'S' (default)
    ckpt_path   : explicit checkpoint path; None = auto-resolve from HF cache
    """

    def __init__(self, dev_root, variant: str = 'S', ckpt_path: Optional[str] = None):
        import torch as _torch
        _ts_lib  = str(dev_root / 'Models' / 'TerraSeg' / 'terraseg_lib' / 'src')
        _ptv3_lib = str(dev_root / 'Models' / 'TerraSeg' / 'ptv3' / 'src')
        for _p in [_ts_lib, _ptv3_lib]:
            if _p not in sys.path:
                sys.path.insert(0, _p)

        if ckpt_path is None:
            try:
                from huggingface_hub import hf_hub_download as _hf
                ckpt_path = _hf(
                    repo_id='TedLentsch/TerraSeg',
                    filename=f'terraseg_{variant.lower()}.pth',
                )
            except Exception as _e:
                raise RuntimeError(
                    f'Cannot locate TerraSeg checkpoint and HF download failed: {_e}'
                )

        from terraseg.predictor import TerraSegPredictor
        self._predictor = TerraSegPredictor(variant=variant, checkpoint_path=ckpt_path)
        self._torch = _torch
        print(f'TerraSeg-{variant} loaded.')

    def segment(self, pts: np.ndarray) -> np.ndarray:
        """
        Return a boolean mask: True = non-ground for each point in pts (N, 3).
        """
        import torch as _torch
        with _torch.no_grad():
            lbl = self._predictor.predict(
                _torch.tensor(pts.copy(), dtype=_torch.float32).contiguous()
            ).cpu().numpy()
        return lbl == 1   # 1 = non-ground

    def unload(self):
        del self._predictor
        self._predictor = None
        try:
            import torch as _torch
            import gc
            gc.collect()
            _torch.cuda.empty_cache()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def _erode_mask(binary_mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0 or _cv2 is None:
        return binary_mask
    kernel = _cv2.getStructuringElement(_cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return _cv2.erode(binary_mask.astype(np.uint8), kernel).astype(bool)


def _crop_roi(pts: np.ndarray, center_xy: np.ndarray, radius: float) -> np.ndarray:
    m = (
        (pts[:, 0] >= center_xy[0] - radius) &
        (pts[:, 0] <= center_xy[0] + radius) &
        (pts[:, 1] >= center_xy[1] - radius) &
        (pts[:, 1] <= center_xy[1] + radius)
    )
    return pts[m]


def _dominant_cluster(pts: np.ndarray, hdb_params: dict
                      ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Largest HDBSCAN cluster, no location bias."""
    if len(pts) < _MIN_PTS_HDBSCAN:
        return None, None
    keep = filter_inmask_lidar_hdbscan(
        pts,
        min_cluster_size=hdb_params['min_cluster_size'],
        min_samples=hdb_params['min_samples'],
        cluster_eps=hdb_params['cluster_eps'],
    )
    if keep is None:
        return None, None
    cl = pts[keep]
    return (cl, cl[:, :2].mean(0)) if len(cl) >= _MIN_PTS_HDBSCAN else (None, None)


def _dominant_cluster_near(pts: np.ndarray, search_xy: np.ndarray,
                           hdb_params: dict
                           ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """HDBSCAN cluster nearest to search_xy (among the 3 largest)."""
    if _hdbscan_mod is None or len(pts) < _MIN_PTS_HDBSCAN:
        return None, None
    labels = _hdbscan_mod.HDBSCAN(
        min_cluster_size=hdb_params['min_cluster_size'],
        min_samples=hdb_params['min_samples'],
        metric='euclidean',
        cluster_selection_epsilon=hdb_params['cluster_eps'],
    ).fit_predict(pts)
    unique = [l for l in np.unique(labels) if l >= 0]
    if not unique:
        return None, None
    clusters = sorted(
        [(pts[labels == lbl], pts[labels == lbl][:, :2].mean(0)) for lbl in unique],
        key=lambda x: -len(x[0]),
    )
    cl, cent = min(clusters[:3], key=lambda x: np.linalg.norm(x[1] - search_xy))
    return (cl, cent) if len(cl) >= _MIN_PTS_HDBSCAN else (None, None)


def _centroid_T(src: np.ndarray, tgt: np.ndarray) -> Tuple[np.ndarray, float, float]:
    d = tgt[:, :2].mean(0) - src[:, :2].mean(0)
    T = np.eye(4, dtype=np.float64)
    T[0, 3] = float(d[0])
    T[1, 3] = float(d[1])
    return T, float(d[0]), float(d[1])


def _apply_se2(pts: np.ndarray, T: np.ndarray) -> np.ndarray:
    h = np.hstack([pts, np.ones((len(pts), 1), dtype=pts.dtype)])
    return (T @ h.T).T[:, :3].astype(pts.dtype)


# ─────────────────────────────────────────────────────────────────────────────
# SE(2) ICP
# ─────────────────────────────────────────────────────────────────────────────

def _se2_icp(
    src: np.ndarray,
    tgt: np.ndarray,
    init_T: np.ndarray,
    allow_yaw: bool = False,
    metric: str = 'p2p',
    max_corresp: float = 0.4,
    max_iter: int = 60,
) -> Tuple[np.ndarray, float, float, float, float, float]:
    """
    SE(2)-constrained ICP between two point clouds.

    Points are flattened to z=0 before ICP so only in-plane translation
    and (optionally) rotation are estimated.  The recovered yaw is only
    applied when allow_yaw=True; otherwise only translation is kept.

    Parameters
    ----------
    src        : (N, 3) source cloud (sweep to be aligned)
    tgt        : (M, 3) target cloud (growing anchor)
    init_T     : (4, 4) initial SE(3) transform (centroid shift)
    allow_yaw  : apply yaw component from ICP result (for turning objects)
    metric     : 'p2p' point-to-point | 'p2l' point-to-plane
    max_corresp: max correspondence distance [m]
    max_iter   : ICP iteration budget

    Returns
    -------
    T2        : (4, 4) final SE(2) transform applied to src
    fitness   : ICP fitness score
    inlier_rmse
    tx, ty    : final translation [m]
    yaw_deg   : rotation extracted from ICP (0 when allow_yaw=False)
    """
    if _o3d is None:
        raise ImportError('open3d is required for ICP. Install with: pip install open3d')

    def _flat_pcd(p, with_normals=False):
        f = p.copy().astype(np.float64)
        f[:, 2] = 0.0
        pc = _o3d.geometry.PointCloud()
        pc.points = _o3d.utility.Vector3dVector(f)
        if with_normals:
            pc.estimate_normals(
                search_param=_o3d.geometry.KDTreeSearchParamHybrid(
                    radius=max_corresp * 3, max_nn=30))
            pc.orient_normals_towards_camera_location([0.0, 0.0, 100.0])
        return pc

    if metric == 'p2l':
        pcd_src = _flat_pcd(src, with_normals=False)
        pcd_tgt = _flat_pcd(tgt, with_normals=True)
        estimation = _o3d.pipelines.registration.TransformationEstimationPointToPlane()
    else:
        pcd_src = _flat_pcd(src)
        pcd_tgt = _flat_pcd(tgt)
        estimation = _o3d.pipelines.registration.TransformationEstimationPointToPoint()

    res = _o3d.pipelines.registration.registration_icp(
        pcd_src, pcd_tgt,
        max_correspondence_distance=max_corresp,
        init=init_T.astype(np.float64),
        estimation_method=estimation,
        criteria=_o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=max_iter),
    )

    T    = np.asarray(res.transformation)
    yaw  = float(np.arctan2(T[1, 0], T[0, 0]))
    tx   = float(T[0, 3])
    ty   = float(T[1, 3])

    T2 = np.eye(4, dtype=np.float64)
    if allow_yaw:
        T2[0, 0] =  np.cos(yaw);  T2[0, 1] = -np.sin(yaw)
        T2[1, 0] =  np.sin(yaw);  T2[1, 1] =  np.cos(yaw)
    T2[0, 3] = tx
    T2[1, 3] = ty

    return T2, float(res.fitness), float(res.inlier_rmse), tx, ty, float(np.degrees(yaw))


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory analysis
# ─────────────────────────────────────────────────────────────────────────────

def _trajectory_yaw_rate(sw_cents: list, sw_data: list) -> float:
    """
    Estimate yaw rate (deg/s) from per-sweep centroid trail.

    Splits trail into two halves, fits a velocity direction to each half with
    lstsq, computes the heading change between halves.  Using regression within
    each half averages out centroid noise instead of letting it accumulate.

    Returns 0.0 when the trail is too short or displacement is too small.
    """
    txy = [(s['dt_ms'] / 1000.0, c[0], c[1])
           for s, c in zip(sw_data, sw_cents) if c is not None]
    if len(txy) < 4:
        return 0.0
    txy = np.array(txy)
    t, x, y = txy[:, 0], txy[:, 1], txy[:, 2]
    if np.hypot(x[-1] - x[0], y[-1] - y[0]) < 1.0:
        return 0.0

    def _fit_heading(ts, xs, ys):
        A  = np.stack([ts, np.ones(len(ts))], axis=1)
        vx = np.linalg.lstsq(A, xs, rcond=None)[0][0]
        vy = np.linalg.lstsq(A, ys, rcond=None)[0][0]
        return float(np.degrees(np.arctan2(vy, vx))) if np.hypot(vx, vy) >= 0.1 else None

    mid = len(txy) // 2
    h1  = _fit_heading(t[:mid], x[:mid], y[:mid])
    h2  = _fit_heading(t[mid:], x[mid:], y[mid:])
    if h1 is None or h2 is None:
        return 0.0

    dh         = (h2 - h1 + 180) % 360 - 180
    total_time = abs(float(t[-1]) - float(t[0]))
    return abs(dh) / total_time if total_time > 0 else 0.0


def _compute_motion(sw_cents: list, sw_data: list,
                    anchor_sw_i: int) -> Tuple[float, float]:
    """
    Estimate object speed (m/s) and heading (rad) from centroid trail via lstsq.
    Returns (speed, heading_rad). heading_rad is nan when < 2 valid centroids.
    """
    txy = []
    for i, s in enumerate(sw_data):
        if i == anchor_sw_i or sw_cents[i] is None:
            continue
        txy.append((s['dt_ms'] / 1000.0, sw_cents[i][0], sw_cents[i][1]))
    if len(txy) < 2:
        return 0.0, float('nan')
    txy = np.array(txy)
    A  = np.stack([txy[:, 0], np.ones(len(txy))], axis=1)
    vx = np.linalg.lstsq(A, txy[:, 1], rcond=None)[0][0]
    vy = np.linalg.lstsq(A, txy[:, 2], rcond=None)[0][0]
    return float(np.sqrt(vx**2 + vy**2)), float(np.arctan2(vy, vx))


def _static_orientation_rad(pts: np.ndarray) -> float:
    """PCA long-axis orientation in XY plane (ambiguous ±π). Returns nan if < 2 pts."""
    if len(pts) < 2:
        return float('nan')
    xy   = pts[:, :2]
    xy_c = xy - xy.mean(0)
    _, _, Vt = np.linalg.svd(xy_c, full_matrices=False)
    return float(np.arctan2(Vt[0, 1], Vt[0, 0])) % np.pi


# ─────────────────────────────────────────────────────────────────────────────
# UV projection helper
# ─────────────────────────────────────────────────────────────────────────────

def _project_sweep_to_cam(
    pts_ego: np.ndarray,
    K: np.ndarray,
    R_c2e: np.ndarray,
    t_c2e: np.ndarray,
    H: int,
    W: int,
) -> np.ndarray:
    """
    Project ego-frame points into a camera.

    Returns (N, 2) float32 UV array where U=-1 means "not visible" in that camera.
    Points behind camera or outside image bounds are marked U=-1.
    """
    pts_cam = (R_c2e.T @ (pts_ego.astype(np.float64) - t_c2e).T).T
    Z       = pts_cam[:, 2]
    uv      = np.full((len(pts_ego), 2), -1.0, dtype=np.float32)
    front   = Z > 0
    if not front.any():
        return uv

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    u = pts_cam[front, 0] / Z[front] * fx + cx
    v = pts_cam[front, 1] / Z[front] * fy + cy
    in_img  = (u >= 0) & (u < W) & (v >= 0) & (v < H)

    idx         = np.where(front)[0][in_img]
    uv[idx, 0]  = u[in_img].astype(np.float32)
    uv[idx, 1]  = v[in_img].astype(np.float32)
    return uv


def _get_mask_pts(
    mask_2d: np.ndarray,
    sweep_pts: np.ndarray,
    sweep_uv: np.ndarray,
) -> np.ndarray:
    """
    Return ego-frame points from sweep_pts whose projection falls inside mask_2d.

    Parameters
    ----------
    mask_2d   : (H, W) bool
    sweep_pts : (N, 3) float  ego-frame points for this sweep
    sweep_uv  : (N, 2) float  UV coordinates; U=-1 means not visible
    """
    visible = sweep_uv[:, 0] >= 0
    if not visible.any():
        return np.empty((0, 3), dtype=np.float32)
    ui = sweep_uv[visible, 0].astype(np.int32)
    vi = sweep_uv[visible, 1].astype(np.int32)
    in_mask = mask_2d[vi, ui]
    return sweep_pts[visible][in_mask].astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Main compensation function
# ─────────────────────────────────────────────────────────────────────────────

def compensate_object_motion(
    sweep_data: list,
    sweep_nonground_pts: list,
    masks: list,
    K: np.ndarray,
    R_c2e: np.ndarray,
    t_c2e: np.ndarray,
    H: int,
    W: int,
    hdbscan_params: Optional[dict] = None,
    use_icp: bool = True,
    mask_erode_px: int = 4,
    turning_yaw_rate_deg_s: float = 5.0,
    icp_max_corresp: float = 0.4,
    icp_max_iter: int = 60,
    verbose: bool = True,
) -> list:
    """
    Run two-phase ICP motion compensation for all SAM3 masks in one camera.

    Parameters
    ----------
    sweep_data          : list of dicts {rel_idx, dt_ms, pts_ego_anc}
                          in chronological order (from load_sweep_data)
    sweep_nonground_pts : list of (N_i, 3) float32 — ground-removed points per sweep
                          (same length and order as sweep_data)
    masks               : list of mask dicts {binary_mask, prompt, score, ...}
                          from SAM3 for this camera (all prompts merged)
    K, R_c2e, t_c2e, H, W : camera intrinsics/extrinsics and image size
    hdbscan_params      : {prompt: {min_cluster_size, min_samples, cluster_eps}}
                          falls back to per-class defaults when None or prompt missing
    use_icp             : False = skip ICP, return HDBSCAN-only clusters
    mask_erode_px       : mask erosion radius before in-mask point selection
    turning_yaw_rate_deg_s : yaw rate threshold to enable ICP yaw component
    icp_max_corresp     : ICP max correspondence distance [m]
    icp_max_iter        : ICP iteration budget
    verbose             : print per-mask summary

    Returns
    -------
    list of result dicts (one per input mask), in the same order as masks:
        mask_idx           : int
        prompt             : str
        score              : float
        is_dynamic         : bool
        is_turning         : bool
        traj_yaw_rate_degs : float
        velocity_mps       : float
        heading_rad        : float  (motion if dynamic, PCA if static, nan if no data)
        pts_comp_all       : (N, 3) float32  final object point cloud
        n_found_sweeps     : int    number of non-anchor sweeps with a valid cluster
    """
    if _hdbscan_mod is None:
        raise ImportError('hdbscan is required. Install with: pip install hdbscan')
    if use_icp and _o3d is None:
        raise ImportError('open3d is required for ICP. Install with: pip install open3d')

    anchor_sw_i = next(i for i, s in enumerate(sweep_data) if s['rel_idx'] == 0)
    hdb_defaults = {
        'pedestrian':           {'min_cluster_size': 2, 'min_samples': 1, 'cluster_eps': 0.20},
        'bicycle':              {'min_cluster_size': 2, 'min_samples': 1, 'cluster_eps': 0.20},
        'motorcycle':           {'min_cluster_size': 2, 'min_samples': 1, 'cluster_eps': 0.30},
        'car':                  {'min_cluster_size': 2, 'min_samples': 1, 'cluster_eps': 0.50},
        'truck':                {'min_cluster_size': 2, 'min_samples': 1, 'cluster_eps': 0.60},
        'construction vehicle': {'min_cluster_size': 2, 'min_samples': 1, 'cluster_eps': 0.60},
        'construction_vehicle': {'min_cluster_size': 2, 'min_samples': 1, 'cluster_eps': 0.60},
        'bus':                  {'min_cluster_size': 2, 'min_samples': 1, 'cluster_eps': 0.80},
        'trailer':              {'min_cluster_size': 2, 'min_samples': 1, 'cluster_eps': 0.70},
    }
    _DEFAULT_HDBSCAN = {'min_cluster_size': 2, 'min_samples': 1, 'cluster_eps': 0.50}

    def _hdb(prompt):
        if hdbscan_params and prompt in hdbscan_params:
            return hdbscan_params[prompt]
        return hdb_defaults.get(prompt, _DEFAULT_HDBSCAN)

    # ── Pre-erode masks + project each sweep into camera ─────────────────────
    eroded_masks = [_erode_mask(m['binary_mask'], mask_erode_px) for m in masks]

    # sweep_uv[i]: (N_i, 2) UV coords for sweep i (U=-1 = not visible)
    sweep_uv = [
        _project_sweep_to_cam(
            s['pts_ego_anc'], K, R_c2e, t_c2e, H, W)
        for s in sweep_data
    ]

    results = []

    for mi, minfo in enumerate(masks):
        prompt      = minfo['prompt']
        hdb_p       = _hdb(prompt)
        radius      = CLASS_SEARCH_RADIUS_M.get(prompt, _DEFAULT_RADIUS_M)
        max_spd     = CLASS_MAX_SPEED_MPS.get(prompt, _DEFAULT_MAX_SPEED)
        dyn_thr     = CLASS_DYNAMIC_THRESH_MPS.get(prompt, _DEFAULT_DYNAMIC_THRESH)
        icp_metric  = CLASS_ICP_METRIC.get(prompt, _DEFAULT_ICP_METRIC)
        max_h       = CLASS_MAX_HEIGHT_M.get(prompt, _DEFAULT_MAX_HEIGHT_M)

        # ── Anchor cluster C0 ─────────────────────────────────────────────────
        anc_raw  = _get_mask_pts(eroded_masks[mi],
                                 sweep_data[anchor_sw_i]['pts_ego_anc'],
                                 sweep_uv[anchor_sw_i])
        anc_pts, anc_c = _dominant_cluster(anc_raw, hdb_p)
        if anc_pts is None:
            anc_pts = anc_raw
            anc_c   = anc_pts[:, :2].mean(0) if len(anc_pts) >= _MIN_PTS_HDBSCAN else None
        anc_z_min = float(anc_pts[:, 2].min()) if len(anc_pts) >= _MIN_PTS_HDBSCAN else -99.0

        sw_pts   = [None] * len(sweep_data)
        sw_cents = [None] * len(sweep_data)
        comp     = [None] * len(sweep_data)

        sw_pts[anchor_sw_i]   = anc_pts
        sw_cents[anchor_sw_i] = anc_c
        comp[anchor_sw_i]     = anc_pts.copy()

        # ── Phase 1: ROI propagation — centroids only, no ICP ────────────────
        for _pass in [list(range(anchor_sw_i + 1, len(sweep_data))),
                      list(range(anchor_sw_i - 1, -1, -1))]:
            search_c = anc_c.copy() if anc_c is not None else None

            for i in _pass:
                s  = sweep_data[i]
                dt = abs(s['dt_ms']) / 1000.0

                if search_c is None or len(anc_pts) < _MIN_PTS_HDBSCAN:
                    sw_pts[i]   = np.empty((0, 3), np.float32)
                    sw_cents[i] = None
                    continue

                # Z ceiling from anchor bottom (robust: bottom always visible)
                z_ceil   = anc_z_min + max_h
                crop_raw = _crop_roi(sweep_nonground_pts[i], search_c, radius)
                z_mask   = (
                    (crop_raw[:, 2] >= anc_z_min - _Z_ANCHOR_FLOOR_SLACK) &
                    (crop_raw[:, 2] <= z_ceil)
                )
                crop = crop_raw[z_mask]

                cl_pts, cl_cent = _dominant_cluster_near(crop, search_c, hdb_p)
                if cl_pts is None or len(cl_pts) < _MIN_PTS_HDBSCAN:
                    sw_pts[i]   = crop if cl_pts is None else cl_pts
                    sw_cents[i] = None
                    continue

                cl_cent = cl_pts[:, :2].mean(0)
                if np.linalg.norm(cl_cent - search_c) < max_spd * dt:
                    search_c = cl_cent

                sw_pts[i]   = cl_pts
                sw_cents[i] = cl_cent

        # ── Turning detection from centroid trail ─────────────────────────────
        yaw_rate   = _trajectory_yaw_rate(sw_cents, sweep_data)
        is_turning = yaw_rate > turning_yaw_rate_deg_s

        # ── Phase 2: ICP — alternating fwd/bwd, growing target ───────────────
        fwd_order    = list(range(anchor_sw_i + 1, len(sweep_data)))
        bwd_order    = list(range(anchor_sw_i - 1, -1, -1))
        phase2_order = [x for pair in zip_longest(fwd_order, bwd_order)
                        for x in pair if x is not None]

        agg_pts = anc_pts.copy()   # growing target cloud

        for i in phase2_order:
            cl_pts  = sw_pts[i]
            cl_cent = sw_cents[i]

            # Gate: require a valid Phase 1 cluster to use as ICP seed
            if cl_pts is None or len(cl_pts) < _MIN_PTS_HDBSCAN or cl_cent is None:
                comp[i] = np.empty((0, 3), np.float32)
                continue

            if not use_icp:
                comp[i] = cl_pts.copy()
                continue

            init_T, tx0, ty0 = _centroid_T(cl_pts, agg_pts)

            if len(cl_pts) < _MIN_PTS_ICP:
                # Too few points for ICP — use centroid shift only
                comp[i] = _apply_se2(cl_pts, init_T)
            else:
                T, _fit, _rmse, _tx, _ty, _yaw = _se2_icp(
                    cl_pts, agg_pts, init_T,
                    allow_yaw=is_turning,
                    metric=icp_metric,
                    max_corresp=icp_max_corresp,
                    max_iter=icp_max_iter,
                )
                comp[i] = _apply_se2(cl_pts, T)

            # Grow target with newly compensated points
            if len(comp[i]) > 0:
                agg_pts = np.concatenate([agg_pts, comp[i]], axis=0)

        # ── Dynamic/static classification ─────────────────────────────────────
        cent_vels = []
        for i, s in enumerate(sweep_data):
            if i == anchor_sw_i or sw_cents[i] is None or anc_c is None:
                continue
            dt = abs(s['dt_ms']) / 1000.0
            if dt < 1e-3:
                continue
            cent_vels.append(np.linalg.norm(sw_cents[i] - anc_c) / dt)

        vel_class = float(np.median(cent_vels)) if cent_vels else 0.0
        is_dynamic = vel_class > dyn_thr
        n_found    = len(cent_vels)

        # ── Final cloud ───────────────────────────────────────────────────────
        if is_dynamic:
            valid_comp = [c for c in comp if c is not None and len(c) > 0]
            if valid_comp:
                comp_cat    = np.concatenate(valid_comp, axis=0)
                final_cl, _ = _dominant_cluster(comp_cat, hdb_p)
                pts_comp_all = (final_cl
                                if final_cl is not None and len(final_cl) >= _MIN_PTS_HDBSCAN
                                else comp_cat)
            else:
                pts_comp_all = np.empty((0, 3), np.float32)
        else:
            # Static: aggregate raw in-mask points from all sweeps (no ICP needed),
            # then take dominant HDBSCAN cluster to remove background noise.
            all_parts = [
                _get_mask_pts(eroded_masks[mi], sweep_data[i]['pts_ego_anc'], sweep_uv[i])
                for i in range(len(sweep_data))
            ]
            valid_parts = [p for p in all_parts if len(p) > 0]
            if valid_parts:
                all_cat    = np.concatenate(valid_parts, axis=0)
                static_cl, _ = _dominant_cluster(all_cat, hdb_p)
                pts_comp_all = (static_cl
                                if static_cl is not None and len(static_cl) >= _MIN_PTS_HDBSCAN
                                else all_cat)
            else:
                pts_comp_all = np.empty((0, 3), np.float32)

        # ── Velocity + heading ────────────────────────────────────────────────
        if is_dynamic:
            velocity_mps, heading_rad = _compute_motion(sw_cents, sweep_data, anchor_sw_i)
        else:
            velocity_mps = 0.0
            heading_rad  = _static_orientation_rad(pts_comp_all)

        if verbose:
            tag      = 'DYNAMIC' if is_dynamic else 'static '
            turn_str = f'  turn={yaw_rate:.1f}deg/s' if is_turning else ''
            hdg_deg  = np.degrees(heading_rad) if not np.isnan(heading_rad) else float('nan')
            hdg_str  = (f'  hdg={hdg_deg:.0f}°({"motion" if is_dynamic else "PCA"})'
                        if not np.isnan(hdg_deg) else '')
            print(f'  mask #{mi:2d}  {prompt:<22}  {tag}  '
                  f'v_cls={vel_class:.2f}m/s(thr={dyn_thr})  v_net={velocity_mps:.2f}m/s  '
                  f'anc={len(anc_pts):3d}pts  out={len(pts_comp_all):4d}pts  '
                  f'sweeps={n_found}/{len(sweep_data)-1}{turn_str}{hdg_str}')

        results.append(dict(
            mask_idx           = mi,
            prompt             = prompt,
            score              = minfo.get('score', 1.0),
            is_dynamic         = is_dynamic,
            is_turning         = is_turning,
            traj_yaw_rate_degs = yaw_rate,
            velocity_mps       = velocity_mps,
            heading_rad        = heading_rad,
            pts_comp_all       = pts_comp_all,
            n_found_sweeps     = n_found,
        ))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Flat mask list builder
# ─────────────────────────────────────────────────────────────────────────────

def flatten_sam3_masks(frame_sam3: dict) -> list:
    """
    Convert SAM3 output dict {prompt: [dets]} into a flat list of mask dicts,
    adding 'prompt' field to each entry and a sequential mask_idx.
    """
    flat = []
    for prompt, dets in frame_sam3.items():
        for d in dets:
            flat.append({**d, 'prompt': prompt})
    return flat
