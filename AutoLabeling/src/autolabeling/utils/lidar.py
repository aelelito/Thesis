"""
Shared LiDAR utilities.

Provides:
  - HDBSCAN-based in-mask point filtering (O3, B1 depth correction)
  - LiDAR loading and camera projection (ground estimation, B1)
  - Multi-sweep LiDAR aggregation with ego-motion compensation
"""
from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np

try:
    import hdbscan as _hdbscan
except ImportError:
    _hdbscan = None


def filter_lidar_pts(
    pts_ego: np.ndarray,
    use_ego_body_filter: bool = True,
    ego_box_half_x: float = 4.0,
    ego_box_half_y: float = 1.5,
    ego_box_z_min: float = 0.5,
    ego_box_z_max: float = 2.5,
    max_range_m: float = 52.0,
) -> np.ndarray:
    """
    Pre-filter a full ego-frame LiDAR point cloud before camera projection
    and HDBSCAN in-mask clustering.

    Two filters are applied in sequence:

    1. Ego-body exclusion zone  (if use_ego_body_filter=True)
       Removes returns that hit the ego vehicle's own roof, windshield, and
       LiDAR mount hardware.  These appear at a fixed position in ego space
       and form a dense horizontal strip after sweep aggregation, which can
       dominate HDBSCAN when an object happens to overlap that region.
       Only points inside the box *and* within the Z band are removed:
           |x| < ego_box_half_x  AND  |y| < ego_box_half_y
           AND  ego_box_z_min < z < ego_box_z_max
       (The Z band preserves close ground-level returns and tall objects above.)

    2. Maximum BEV range filter
       Removes points beyond max_range_m (sqrt(x² + y²) from ego origin).
       Cuts distant background returns that cannot form valid object clusters
       but can still fall inside far-away projected masks.

    Parameters
    ----------
    pts_ego           : (N, 3) float  ego-frame XYZ
    use_ego_body_filter : bool         enable ego-body box filter
    ego_box_half_x    : float         half-length fore/aft [m]
    ego_box_half_y    : float         half-width left/right [m]
    ego_box_z_min     : float         lower Z cutoff of ego box [m]
    ego_box_z_max     : float         upper Z cutoff of ego box [m]
    max_range_m       : float         BEV range cutoff [m]

    Returns
    -------
    pts_ego_filtered : (M, 3) float  filtered point cloud (M ≤ N)
    """
    if use_ego_body_filter:
        in_ego_box = (
            (np.abs(pts_ego[:, 0]) < ego_box_half_x) &
            (np.abs(pts_ego[:, 1]) < ego_box_half_y) &
            (pts_ego[:, 2] > ego_box_z_min) &
            (pts_ego[:, 2] < ego_box_z_max)
        )
        pts_ego = pts_ego[~in_ego_box]

    bev_range = np.sqrt(pts_ego[:, 0] ** 2 + pts_ego[:, 1] ** 2)
    pts_ego   = pts_ego[bev_range <= max_range_m]

    return pts_ego


def filter_inmask_lidar_hdbscan(
    pts_ego_inmask,
    min_cluster_size=3,
    min_samples=1,
    cluster_eps=0.4,
    proximity_min_pts=30,
    proximity_ratio=0.70,
):
    """
    Isolate the dominant LiDAR surface cluster from in-mask points using HDBSCAN.

    Clustering is performed in 3D ego-frame coordinates (metres) so the distance
    metric is physically meaningful and range-invariant.

    Cluster selection:
    - Normally the largest cluster is returned (object surface estimate).
    - Proximity selection: when the two largest clusters both have ≥ proximity_min_pts
      points and the second is ≥ proximity_ratio × largest, pick the CLOSER one by
      ego-frame centroid distance.  This prevents a large background cluster from
      dominating when an occluder and an object each contribute roughly equal points.

    Smaller clusters and noise (label -1) are always discarded.

    Parameters
    ----------
    pts_ego_inmask   : (N, 3) float32  ego-frame XYZ of LiDAR points inside the mask
    min_cluster_size : int             HDBSCAN min_cluster_size
    min_samples      : int             HDBSCAN min_samples
    cluster_eps      : float           HDBSCAN cluster_selection_epsilon [m]
                                       merges sub-clusters closer than this distance
    proximity_min_pts : int            min size of largest cluster to trigger
                                       proximity selection (default 30)
    proximity_ratio   : float          second/largest size ratio threshold for
                                       proximity selection (default 0.70)

    Returns
    -------
    keep : (N,) bool  True for points belonging to the dominant cluster,
                      or None if no valid cluster was found (all noise)
    """
    if _hdbscan is None:
        raise ImportError(
            'hdbscan is required for LiDAR in-mask filtering. '
            'Install with: pip install hdbscan'
        )

    if len(pts_ego_inmask) < min_cluster_size:
        return None

    clusterer = _hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
        cluster_selection_epsilon=cluster_eps,
    )
    labels = clusterer.fit_predict(pts_ego_inmask)
    unique, counts = np.unique(labels[labels >= 0], return_counts=True)

    if len(unique) == 0:
        return None

    order    = np.argsort(counts)[::-1]
    biggest  = int(counts[order[0]])
    if (biggest >= proximity_min_pts
            and len(order) >= 2
            and counts[order[1]] >= proximity_ratio * biggest):
        c0, c1 = unique[order[0]], unique[order[1]]
        d0 = float(np.linalg.norm(pts_ego_inmask[labels == c0].mean(axis=0)))
        d1 = float(np.linalg.norm(pts_ego_inmask[labels == c1].mean(axis=0)))
        dominant = c0 if d0 <= d1 else c1
    else:
        dominant = unique[order[0]]

    return labels == dominant


def _walk_lidar_tokens(nusc, anchor_token: str, n_before: int, n_after: int) -> List[Tuple[int, str]]:
    """
    Walk the nuScenes sample_data linked list around an anchor LiDAR sweep.

    Skips sweeps that share the same file path as already-seen sweeps (handles
    ECP where adjacent entries sometimes point to the same .bin file).

    Parameters
    ----------
    nusc         : NuScenes instance
    anchor_token : sample_data token of the anchor (keyframe-aligned) LiDAR sweep
    n_before     : maximum number of sweeps to collect before the anchor
    n_after      : maximum number of sweeps to collect after the anchor

    Returns
    -------
    tokens : list of (rel_idx, token) tuples in chronological order.
             rel_idx = 0 for anchor, negative for before, positive for after.
    """
    anchor_path = nusc.get_sample_data_path(anchor_token)
    seen_paths  = {anchor_path}

    # Collect sweeps before anchor (walk backward).
    backward = []
    tok = nusc.get('sample_data', anchor_token)['prev']
    while tok and len(backward) < n_before:
        path = nusc.get_sample_data_path(tok)
        if path not in seen_paths:
            backward.append(tok)
            seen_paths.add(path)
        tok = nusc.get('sample_data', tok)['prev']

    tokens: List[Tuple[int, str]] = []
    for i, t in enumerate(reversed(backward)):
        tokens.append((-(len(backward) - i), t))
    tokens.append((0, anchor_token))

    # Collect sweeps after anchor (walk forward).
    count = 0
    tok   = nusc.get('sample_data', anchor_token)['next']
    while tok and count < n_after:
        path = nusc.get_sample_data_path(tok)
        if path not in seen_paths:
            count += 1
            tokens.append((count, tok))
            seen_paths.add(path)
        tok = nusc.get('sample_data', tok)['next']

    return tokens


def _load_sweep_ego(
    nusc, tok: str, R_e2g_anchor: np.ndarray, t_e2g_anchor: np.ndarray,
    use_ego_body_filter: bool = True,
    ego_box_half_x: float = 4.0,
    ego_box_half_y: float = 1.5,
    ego_box_z_min: float = 0.5,
    ego_box_z_max: float = 2.5,
) -> np.ndarray:
    """
    Load one LiDAR sweep and transform it into the anchor ego frame.

    Chain: LiDAR sensor → sweep ego → (ego-body filter) → global → anchor ego.

    The ego-body filter is applied in the sweep's own ego frame (vehicle always
    at the origin) before the global transform.  This ensures ego-body returns
    are removed regardless of the temporal offset between this sweep and the
    anchor, avoiding the displacement artefact that occurs when filtering in
    anchor ego frame after sweeps from different timestamps are already merged.

    Parameters
    ----------
    nusc               : NuScenes instance
    tok                : sample_data token for this sweep
    R_e2g_anchor       : (3,3) rotation  ego → global at anchor timestamp
    t_e2g_anchor       : (3,)  translation ego → global at anchor timestamp
    use_ego_body_filter: bool  apply ego-body exclusion box in sweep ego frame
    ego_box_half_x     : float half-length fore/aft  [m]
    ego_box_half_y     : float half-width left/right [m]
    ego_box_z_min      : float lower Z cutoff of ego box [m]
    ego_box_z_max      : float upper Z cutoff of ego box [m]

    Returns
    -------
    pts_ego_anc : (N, 3) float64 in anchor ego frame
    """
    from pyquaternion import Quaternion as _Quaternion

    sd       = nusc.get('sample_data', tok)
    lid_cal  = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
    ego_pose = nusc.get('ego_pose', sd['ego_pose_token'])
    lid_path = nusc.get_sample_data_path(tok)

    R_l2e    = _Quaternion(lid_cal['rotation']).rotation_matrix.astype(np.float64)
    t_l2e    = np.array(lid_cal['translation'], dtype=np.float64)
    R_e2g_sw = _Quaternion(ego_pose['rotation']).rotation_matrix.astype(np.float64)
    t_e2g_sw = np.array(ego_pose['translation'], dtype=np.float64)

    pts_raw    = np.fromfile(lid_path, dtype=np.float32).reshape(-1, 5)[:, :3].astype(np.float64)
    pts_ego_sw = (R_l2e @ pts_raw.T).T + t_l2e

    if use_ego_body_filter:
        _in_box    = (
            (np.abs(pts_ego_sw[:, 0]) < ego_box_half_x) &
            (np.abs(pts_ego_sw[:, 1]) < ego_box_half_y) &
            (pts_ego_sw[:, 2] > ego_box_z_min) &
            (pts_ego_sw[:, 2] < ego_box_z_max)
        )
        pts_ego_sw = pts_ego_sw[~_in_box]

    pts_global  = (R_e2g_sw @ pts_ego_sw.T).T + t_e2g_sw
    pts_ego_anc = (R_e2g_anchor.T @ (pts_global - t_e2g_anchor).T).T
    return pts_ego_anc


def load_lidar_pts_aggregated(
    nusc, frame, n_before: int, n_after: int,
    use_ego_body_filter: bool = True,
    ego_box_half_x: float = 4.0,
    ego_box_half_y: float = 1.5,
    ego_box_z_min: float = 0.5,
    ego_box_z_max: float = 2.5,
) -> Optional[np.ndarray]:
    """
    Load and aggregate multiple LiDAR sweeps around the anchor frame.

    All sweeps are ego-motion compensated to the anchor ego frame before
    concatenation.  The ego-body filter is applied per-sweep in each sweep's
    own ego frame so that returns from the ego vehicle are always removed at
    the origin regardless of sweep timing.

    Falls back to a single-sweep load when:
      - frame has no lidar_sd_token (dataset without LIDAR_TOP)
      - n_before=0 and n_after=0 (caller just wants the anchor sweep)

    Parameters
    ----------
    nusc               : NuScenes instance
    frame              : FrameRecord with lidar_sd_token, R_e2g, t_e2g populated
    n_before           : number of sweeps before anchor to include
    n_after            : number of sweeps after anchor to include
    use_ego_body_filter: bool  passed through to _load_sweep_ego
    ego_box_half_x/y   : float ego-body box half-extents [m]
    ego_box_z_min/max  : float ego-body box Z band [m]

    Returns
    -------
    pts_ego : (N, 3) float64 aggregated point cloud in anchor ego frame,
              or None if no LiDAR available for this frame.
    """
    if frame.lidar_sd_token is None:
        return load_lidar_pts(frame)   # fallback: single sweep via calibration fields

    if n_before == 0 and n_after == 0:
        return load_lidar_pts(frame)   # no aggregation requested

    _filter_kw = dict(
        use_ego_body_filter=use_ego_body_filter,
        ego_box_half_x=ego_box_half_x,
        ego_box_half_y=ego_box_half_y,
        ego_box_z_min=ego_box_z_min,
        ego_box_z_max=ego_box_z_max,
    )
    sweep_tokens = _walk_lidar_tokens(nusc, frame.lidar_sd_token, n_before, n_after)
    all_pts = [
        _load_sweep_ego(nusc, tok, frame.R_e2g, frame.t_e2g, **_filter_kw)
        for _, tok in sweep_tokens
    ]
    return np.concatenate(all_pts, axis=0)


def load_sweep_data(
    nusc, frame, n_before: int, n_after: int,
    use_ego_body_filter: bool = True,
    ego_box_half_x: float = 4.0,
    ego_box_half_y: float = 1.5,
    ego_box_z_min: float = 0.5,
    ego_box_z_max: float = 2.5,
) -> Optional[List[dict]]:
    """
    Load per-sweep data for the ICP motion compensation pass.

    Returns the same N_BEFORE/N_AFTER window used by load_lidar_pts_aggregated,
    but keeps each sweep separate instead of concatenating.

    Parameters
    ----------
    nusc               : NuScenes instance
    frame              : FrameRecord with lidar_sd_token, R_e2g, t_e2g populated
    n_before / n_after : sweep window (same values as lidar_aggregation config)
    use_ego_body_filter: apply ego-body box per sweep (in sweep's own ego frame)
    ego_box_*          : ego-body box parameters

    Returns
    -------
    list of dicts in chronological order, each containing:
        rel_idx     : int    0 = anchor, negative = before, positive = after
        dt_ms       : float  timestamp offset from anchor [ms]
        pts_ego_anc : (N, 3) float32  ground-unfiltered points in anchor ego frame
    None when frame has no LiDAR.
    """
    if frame.lidar_sd_token is None:
        return None

    _filter_kw = dict(
        use_ego_body_filter=use_ego_body_filter,
        ego_box_half_x=ego_box_half_x,
        ego_box_half_y=ego_box_half_y,
        ego_box_z_min=ego_box_z_min,
        ego_box_z_max=ego_box_z_max,
    )
    sweep_tokens = _walk_lidar_tokens(nusc, frame.lidar_sd_token, n_before, n_after)

    # Anchor timestamp for dt computation
    anchor_sd  = nusc.get('sample_data', frame.lidar_sd_token)
    anchor_ts  = anchor_sd['timestamp']

    result = []
    for rel_idx, tok in sweep_tokens:
        sd     = nusc.get('sample_data', tok)
        dt_ms  = (sd['timestamp'] - anchor_ts) / 1_000.0
        pts    = _load_sweep_ego(nusc, tok, frame.R_e2g, frame.t_e2g, **_filter_kw)
        result.append(dict(
            rel_idx     = rel_idx,
            dt_ms       = dt_ms,
            pts_ego_anc = pts.astype(np.float32),
        ))
    return result


def load_lidar_pts(frame) -> Optional[np.ndarray]:
    """
    Load the full LiDAR sweep for a frame into ego frame coordinates.

    Parameters
    ----------
    frame : FrameRecord  (needs lidar_path, R_l2e, t_l2e)

    Returns
    -------
    pts_ego : (N, 3) float64  ego-frame XYZ, or None if no LiDAR available
    """
    if frame.lidar_path is None or frame.R_l2e is None:
        return None
    pts_raw = np.fromfile(frame.lidar_path, dtype=np.float32).reshape(-1, 5)[:, :3]
    return (frame.R_l2e @ pts_raw.astype(np.float64).T).T + frame.t_l2e


def project_lidar_to_camera(
    frame, pts_ego: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """
    Project ego-frame LiDAR points into the camera, keeping only visible points.

    Parameters
    ----------
    frame    : FrameRecord  (needs K, R_c2e, t_c2e, img_path)
    pts_ego  : (N, 3) float64  full LiDAR sweep in ego frame

    Returns
    -------
    pts_ego_vis : (M, 3) float32  ego-frame coords of visible points
    u_vis       : (M,)   float32  pixel column
    v_vis       : (M,)   float32  pixel row
    Z_vis       : (M,)   float32  camera-space depth [m]
    H, W        : int    image height and width
    """
    from PIL import Image as _PIL
    with _PIL.open(frame.img_path) as _img:
        W, H = _img.size

    pts_cam = (frame.R_c2e.T @ (pts_ego - frame.t_c2e).T).T
    Z_cam   = pts_cam[:, 2]
    front   = Z_cam > 0

    fx, fy, cx, cy = frame.K[0, 0], frame.K[1, 1], frame.K[0, 2], frame.K[1, 2]
    u_proj = pts_cam[front, 0] / Z_cam[front] * fx + cx
    v_proj = pts_cam[front, 1] / Z_cam[front] * fy + cy
    in_img = (u_proj >= 0) & (u_proj < W) & (v_proj >= 0) & (v_proj < H)

    pts_ego_vis = pts_ego[front][in_img].astype(np.float32)
    u_vis       = u_proj[in_img].astype(np.float32)
    v_vis       = v_proj[in_img].astype(np.float32)
    Z_vis       = pts_cam[front][in_img][:, 2].astype(np.float32)

    return pts_ego_vis, u_vis, v_vis, Z_vis, H, W
