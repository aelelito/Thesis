import numpy as np

from ..utils.geometry import cam_to_ego


def compute_obb_pedestrian(vertices: np.ndarray, fwd: np.ndarray):
    """
    Fit an OBB aligned to the pedestrian's facing direction, in camera space.

    Parameters
    ----------
    vertices : (V, 3) float32  mesh vertices in camera space (R3/OpenCV)
    fwd      : (3,)   float32  unit facing vector in XZ plane (Y=0)

    Returns
    -------
    corners : (8, 3) float32  OBB corners in camera space
    center  : (3,)   float32
    dims    : (3,)   float32  [width, height, depth]  (right, up, fwd axes)
    """
    right = np.array([fwd[2], 0.0, -fwd[0]], dtype=np.float32)  # 90° CW in XZ
    up    = np.array([0.0, 1.0,  0.0], dtype=np.float32)

    r_proj = vertices @ right
    y_proj = vertices[:, 1]
    f_proj = vertices @ fwd

    r_min, r_max = float(r_proj.min()), float(r_proj.max())
    y_min, y_max = float(y_proj.min()), float(y_proj.max())
    f_min, f_max = float(f_proj.min()), float(f_proj.max())

    corners = np.array([
        r_min * right + y_min * up + f_min * fwd,
        r_max * right + y_min * up + f_min * fwd,
        r_max * right + y_max * up + f_min * fwd,
        r_min * right + y_max * up + f_min * fwd,
        r_min * right + y_min * up + f_max * fwd,
        r_max * right + y_min * up + f_max * fwd,
        r_max * right + y_max * up + f_max * fwd,
        r_min * right + y_max * up + f_max * fwd,
    ], dtype=np.float32)

    center = corners.mean(axis=0)
    dims   = np.array([r_max - r_min, y_max - y_min, f_max - f_min], dtype=np.float32)
    return corners, center, dims


def compute_obb_gravity_aligned(
    verts_r3: np.ndarray,
    R_c2e: np.ndarray,
    t_c2e: np.ndarray,
    ground_z: float = None,
    height_pct: tuple = (1, 99),
    height_clamp: tuple = (0.9, 2.8),
):
    """
    Fit a gravity-aligned OBB to mesh vertices, in ego frame.

    Footprint (yaw / length / width) from PCA of the mesh XY footprint in ego
    space.  Yaw has 180° ambiguity at stage 1 (no LiDAR to resolve it).

    Parameters
    ----------
    verts_r3     : (V, 3) float32  mesh vertices in R3 camera space
    R_c2e        : (3, 3) float64  rotation camera → ego
    t_c2e        : (3,)   float64  translation camera → ego
    ground_z     : float | None    ego-frame ground height (None = image-only)
    height_pct   : (low, high)     percentiles for robust height estimate
    height_clamp : (min_h, max_h)  plausible height range [m]

    Returns
    -------
    corners_ego : (8, 3) float64  OBB corners in ego frame
    center_ego  : (3,)   float64
    dims        : (3,)   float64  [length, width, height]
    yaw_rad     : float           yaw of long axis (±π ambiguous)
    """
    verts_ego = cam_to_ego(verts_r3.astype(np.float64), R_c2e, t_c2e)

    xy    = verts_ego[:, :2]
    xy_c  = xy - xy.mean(axis=0)
    _, evecs = np.linalg.eigh(np.cov(xy_c.T))
    evecs = evecs[:, ::-1]  # largest eigenvalue first

    ax_l = np.array([evecs[0, 0], evecs[1, 0], 0.], dtype=np.float64)
    ax_w = np.array([evecs[0, 1], evecs[1, 1], 0.], dtype=np.float64)
    ax_h = np.array([0., 0., 1.],                   dtype=np.float64)

    if ax_l[0] * ax_w[1] - ax_l[1] * ax_w[0] < 0:
        ax_w = -ax_w
    axes  = np.stack([ax_l, ax_w, ax_h], axis=1)  # (3, 3)

    proj  = verts_ego @ axes
    min_p = proj.min(axis=0).copy()
    max_p = proj.max(axis=0).copy()

    if ground_z is not None:
        vz       = verts_ego[:, 2]
        raw_h    = float(np.percentile(vz, height_pct[1]) - np.percentile(vz, height_pct[0]))
        min_p[2] = ground_z
        max_p[2] = ground_z + float(np.clip(raw_h, height_clamp[0], height_clamp[1]))

    mid_p      = (min_p + max_p) / 2.0
    center_ego = axes @ mid_p

    offsets = np.array([
        [min_p[0] - mid_p[0], min_p[1] - mid_p[1], min_p[2] - mid_p[2]],
        [max_p[0] - mid_p[0], min_p[1] - mid_p[1], min_p[2] - mid_p[2]],
        [max_p[0] - mid_p[0], max_p[1] - mid_p[1], min_p[2] - mid_p[2]],
        [min_p[0] - mid_p[0], max_p[1] - mid_p[1], min_p[2] - mid_p[2]],
        [min_p[0] - mid_p[0], min_p[1] - mid_p[1], max_p[2] - mid_p[2]],
        [max_p[0] - mid_p[0], min_p[1] - mid_p[1], max_p[2] - mid_p[2]],
        [max_p[0] - mid_p[0], max_p[1] - mid_p[1], max_p[2] - mid_p[2]],
        [min_p[0] - mid_p[0], max_p[1] - mid_p[1], max_p[2] - mid_p[2]],
    ])
    corners_ego = (axes @ offsets.T).T + center_ego
    dims        = max_p - min_p  # [length, width, height]
    yaw_rad     = float(np.arctan2(evecs[1, 0], evecs[0, 0]))

    return corners_ego, center_ego, dims, yaw_rad
