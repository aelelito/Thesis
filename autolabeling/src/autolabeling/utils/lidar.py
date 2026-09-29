"""
Shared LiDAR utilities.

Provides:
  - Ego-body point filter (the only LiDAR pre-filter -- no range/volume filters)
  - HDBSCAN-based in-mask point filtering (O3-family, B1 depth correction)
  - LiDAR loading and camera projection
  - Two multi-sweep aggregations, both ego-motion compensated to the anchor frame:
      * load_lidar_pts_aggregated            -- full cloud, ground INCLUDED
                                                (PseudoLabeler / B2 RANSAC fallback)
      * load_lidar_pts_nonground_aggregated  -- TerraSeg per sweep BEFORE aggregating,
                                                ground-free by construction
                                                (B1 + every pointmap mode)
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
) -> np.ndarray:
    """
    Ego-body exclusion filter -- the only LiDAR pre-filter in the pipeline.

    Removes returns that hit the ego vehicle's own roof, windshield, and LiDAR
    mount hardware.  These appear at a fixed position in ego space and form a
    dense horizontal strip after sweep aggregation, which can dominate HDBSCAN
    when an object happens to overlap that region.  Only points inside the box
    *and* within the Z band are removed:
        |x| < ego_box_half_x  AND  |y| < ego_box_half_y
        AND  ego_box_z_min < z < ego_box_z_max
    (The Z band preserves close ground-level returns and tall objects above.)

    Must be applied in the ego frame of the sweep the points came from (the
    vehicle is at the origin there) -- both aggregation loaders below do this
    per sweep, before any transform into the anchor frame.

    Parameters
    ----------
    pts_ego           : (N, 3) float  ego-frame XYZ
    use_ego_body_filter : bool         enable ego-body box filter
    ego_box_half_x    : float         half-length fore/aft [m]
    ego_box_half_y    : float         half-width left/right [m]
    ego_box_z_min     : float         lower Z cutoff of ego box [m]
    ego_box_z_max     : float         upper Z cutoff of ego box [m]

    Returns
    -------
    pts_ego_filtered : (M, 3) float  filtered point cloud (M <= N)
    """
    if not use_ego_body_filter:
        return pts_ego
    in_ego_box = (
        (np.abs(pts_ego[:, 0]) < ego_box_half_x) &
        (np.abs(pts_ego[:, 1]) < ego_box_half_y) &
        (pts_ego[:, 2] > ego_box_z_min) &
        (pts_ego[:, 2] < ego_box_z_max)
    )
    return pts_ego[~in_ego_box]


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


def _load_sweep_own_frame(nusc, tok: str):
    """
    Load one LiDAR sweep in ITS OWN ego frame (vehicle at the origin at that
    sweep's own timestamp) -- i.e. before any transform toward the anchor frame.

    This is the frame TerraSeg must see (its height/range features are relative to
    the scan's own sensor origin), and the frame the ego-body filter must run in.

    Returns
    -------
    pts_ego_sw : (N, 3) float64  points in the sweep's own ego frame
    R_e2g_sw   : (3, 3) float64  rotation  sweep ego -> global
    t_e2g_sw   : (3,)   float64  translation sweep ego -> global
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
    return pts_ego_sw, R_e2g_sw, t_e2g_sw


def _sweep_ego_to_anchor(
    pts_ego_sw: np.ndarray, R_e2g_sw: np.ndarray, t_e2g_sw: np.ndarray,
    R_e2g_anchor: np.ndarray, t_e2g_anchor: np.ndarray,
) -> np.ndarray:
    """Sweep ego -> global -> anchor ego (ego-motion compensation)."""
    pts_global = (R_e2g_sw @ pts_ego_sw.T).T + t_e2g_sw
    return (R_e2g_anchor.T @ (pts_global - t_e2g_anchor).T).T


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

    Chain: LiDAR sensor -> sweep ego -> (ego-body filter) -> global -> anchor ego.

    The ego-body filter is applied in the sweep's own ego frame (vehicle always
    at the origin) before the global transform, so ego-body returns are removed
    regardless of the temporal offset between this sweep and the anchor.

    Returns
    -------
    pts_ego_anc : (N, 3) float64 in anchor ego frame
    """
    pts_ego_sw, R_e2g_sw, t_e2g_sw = _load_sweep_own_frame(nusc, tok)
    pts_ego_sw = filter_lidar_pts(
        pts_ego_sw, use_ego_body_filter=use_ego_body_filter,
        ego_box_half_x=ego_box_half_x, ego_box_half_y=ego_box_half_y,
        ego_box_z_min=ego_box_z_min, ego_box_z_max=ego_box_z_max,
    )
    return _sweep_ego_to_anchor(pts_ego_sw, R_e2g_sw, t_e2g_sw, R_e2g_anchor, t_e2g_anchor)


def load_lidar_pts_aggregated(
    nusc, frame, n_before: int, n_after: int,
    use_ego_body_filter: bool = True,
    ego_box_half_x: float = 4.0,
    ego_box_half_y: float = 1.5,
    ego_box_z_min: float = 0.5,
    ego_box_z_max: float = 2.5,
) -> Optional[np.ndarray]:
    """
    FULL multi-sweep aggregation (ground INCLUDED) around the anchor frame.

    Used by PseudoLabeler (and B2's RANSAC fallback), which need real ground
    points to find where the ground is, and which benefit from the extra density
    (offline per-frame optimization, not a model with a fixed training input
    distribution). Everything that consumes points for detection/depth uses
    `load_lidar_pts_nonground_aggregated` instead.

    All sweeps are ego-motion compensated to the anchor ego frame before
    concatenation.  The ego-body filter is applied per-sweep in each sweep's own
    ego frame (and in the single-sweep fallbacks, so every path is filtered).

    Falls back to a single-sweep load when the frame has no lidar_sd_token or
    n_before=n_after=0.

    Returns
    -------
    pts_ego : (N, 3) float64 aggregated point cloud in anchor ego frame,
              or None if no LiDAR available for this frame.
    """
    _filter_kw = dict(
        use_ego_body_filter=use_ego_body_filter,
        ego_box_half_x=ego_box_half_x,
        ego_box_half_y=ego_box_half_y,
        ego_box_z_min=ego_box_z_min,
        ego_box_z_max=ego_box_z_max,
    )

    if frame.lidar_sd_token is None or (n_before == 0 and n_after == 0):
        pts = load_lidar_pts(frame)    # single sweep: frame ego == anchor ego
        return None if pts is None else filter_lidar_pts(pts, **_filter_kw)

    sweep_tokens = _walk_lidar_tokens(nusc, frame.lidar_sd_token, n_before, n_after)
    all_pts = [
        _load_sweep_ego(nusc, tok, frame.R_e2g, frame.t_e2g, **_filter_kw)
        for _, tok in sweep_tokens
    ]
    return np.concatenate(all_pts, axis=0)


def load_lidar_pts_nonground_aggregated(
    nusc, frame, n_before: int, n_after: int, ground_filter,
    use_ego_body_filter: bool = True,
    ego_box_half_x: float = 4.0,
    ego_box_half_y: float = 1.5,
    ego_box_z_min: float = 0.5,
    ego_box_z_max: float = 2.5,
) -> Optional[np.ndarray]:
    """
    Ground-free multi-sweep aggregation: TerraSeg runs PER SWEEP, before aggregating.

    For each sweep: load it in its own native ego frame -> ego-body filter (same
    frame) -> TerraSeg classify (same frame) -> keep NON-ground only -> transform
    into the anchor frame -> concatenate across sweeps.

    Why per sweep and not once on the aggregated cloud: TerraSeg (PTv3, trained on
    OmniLiDAR) was trained/evaluated strictly on single independent scans, with
    height/range features defined relative to each scan's own sensor origin.
    Running it on an aggregated cloud would be out-of-distribution, would compute
    range/height for non-anchor points relative to the WRONG origin, and would
    expose it to motion-smeared trails of dynamic objects. Per-sweep labels simply
    propagate through the ego-motion transform, so the aggregated cloud is
    ground-free at no extra cost and no ground mask needs threading downstream.
    (See notes/clean_pipeline_overview.md for the full argument.)

    n_before = n_after = 0 (or no lidar_sd_token) degenerates to single-sweep
    ground removal -- aggregation is simply a no-op.

    Parameters
    ----------
    ground_filter : object with .segment((N,3) ndarray) -> (N,) bool, True = non-ground
                    (utils.terraseg.TerraSegGroundFilter)

    Returns
    -------
    pts_ego : (N, 3) float64 ground-free point cloud in anchor ego frame,
              or None if no LiDAR available for this frame.
    """
    _filter_kw = dict(
        use_ego_body_filter=use_ego_body_filter,
        ego_box_half_x=ego_box_half_x,
        ego_box_half_y=ego_box_half_y,
        ego_box_z_min=ego_box_z_min,
        ego_box_z_max=ego_box_z_max,
    )

    if frame.lidar_sd_token is None or (n_before == 0 and n_after == 0):
        pts = load_lidar_pts(frame)    # single sweep: frame ego == anchor ego
        if pts is None:
            return None
        pts = filter_lidar_pts(pts, **_filter_kw)
        return pts[ground_filter.segment(pts)]

    sweep_tokens = _walk_lidar_tokens(nusc, frame.lidar_sd_token, n_before, n_after)
    parts = []
    for _, tok in sweep_tokens:
        pts_ego_sw, R_e2g_sw, t_e2g_sw = _load_sweep_own_frame(nusc, tok)
        pts_ego_sw = filter_lidar_pts(pts_ego_sw, **_filter_kw)
        pts_ng     = pts_ego_sw[ground_filter.segment(pts_ego_sw)]
        parts.append(_sweep_ego_to_anchor(
            pts_ng, R_e2g_sw, t_e2g_sw, frame.R_e2g, frame.t_e2g))
    return np.concatenate(parts, axis=0)


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
