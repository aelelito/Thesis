"""
Controls for the free-space measurement (what the red points mean).

align_mesh_to_box     : the mesh put at the GT position / yaw (/ size) = the free-space share a correctly placed mesh of this shape has
ray_shift_curve       : slide the mesh along the LiDAR ray and re-measure: is there a ray-direction offset, and how much would it recover
classify_violations   : for every violating surface point, would a depth constraint on the MASK pixels have seen it?
                        (the question "does free space add anything beyond LiDAR depth inside the mask?")

Everything in the ego frame of the keyframe, mesh surface samples (M, 3), boxes as centre / yaw / dims [length, width, height].
"""
from typing import Dict, Iterable, Optional, Sequence

import numpy as np

from . import freespace as fs


def align_mesh_to_box(verts: np.ndarray, src_center, src_yaw: float, src_dims: Sequence[float], dst_center, dst_yaw: float,
                      dst_dims: Optional[Sequence[float]] = None) -> np.ndarray:
    """
    Move a mesh so that its OBB (src) coincides with a target box (dst): translate the centre, rotate about the vertical axis
    by the smallest yaw difference (the OBB heading is only known up to 180 deg), and, if dst_dims is given, stretch each axis of
    the box frame from src_dims to dst_dims. The mesh SHAPE is kept, only where it is placed changes.
    """
    v = np.asarray(verts, np.float64) - np.asarray(src_center, np.float64)
    d = (dst_yaw - src_yaw + np.pi / 2) % np.pi - np.pi / 2
    c, s = np.cos(d), np.sin(d)
    v = np.stack([c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1], v[:, 2]], axis=1)
    if dst_dims is not None:
        cy, sy = np.cos(dst_yaw), np.sin(dst_yaw)
        lx, ly = cy * v[:, 0] + sy * v[:, 1], -sy * v[:, 0] + cy * v[:, 1]
        lx, ly, lz = lx * dst_dims[0] / src_dims[0], ly * dst_dims[1] / src_dims[1], v[:, 2] * dst_dims[2] / src_dims[2]
        v = np.stack([cy * lx - sy * ly, sy * lx + cy * ly, lz], axis=1)
    return v + np.asarray(dst_center, np.float64)


def ray_shift_curve(grid: fs.FreeSpaceGrid, surf: np.ndarray, center, lidar_pts: np.ndarray,
                    shifts: Iterable[float] = tuple(np.round(np.arange(-0.5, 0.51, 0.1), 2)), claim: float = 0.2) -> Dict[str, np.ndarray]:
    """
    Translate the surface samples `surf` along the horizontal LiDAR ray through `center` (positive = away from the sensor) and
    measure at every shift: share of the surface in free space (>= claim in front of a return), share on occupied voxels, and the
    median distance from `lidar_pts` (LiDAR returns near the object) to the nearest surface sample.
    """
    from scipy.spatial import cKDTree
    d = np.asarray(center, np.float64)[:2] - grid.sensor[:2]
    u = np.array([d[0], d[1], 0.0]) / max(np.linalg.norm(d), 1e-9)
    shifts = np.asarray(list(shifts), np.float64)
    free = np.zeros(len(shifts)); occ = np.zeros(len(shifts)); lid = np.full(len(shifts), np.nan)
    for i, sh in enumerate(shifts):
        P = surf + sh * u
        st = grid.state(P)
        f = st == fs.FREE
        red = np.zeros(len(P), bool)
        if f.any():
            red[f] = grid.dist_to_occupied(P[f]) >= claim
        free[i], occ[i] = red.mean(), (st == fs.OCC).mean()
        if len(lidar_pts):
            lid[i] = float(np.median(cKDTree(P).query(lidar_pts)[0]))
    return dict(shifts=shifts, free=free, occ=occ, lidar_dist=lid)


def violating_points(grid: fs.FreeSpaceGrid, surf: np.ndarray, claim: float = 0.2):
    """Surface samples in free space >= claim in front of the nearest return, and the beam end behind each (occupied voxel)."""
    st = grid.state(surf)
    idx = np.where(st == fs.FREE)[0]
    if len(idx) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3))
    dist = grid.dist_to_occupied(surf[idx])
    keep = np.isfinite(dist) & (dist >= claim)
    P = surf[idx[keep]]
    u = P - grid.sensor
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    return P, P + u * dist[keep][:, None]


CLASSES = ('implied', 'in_mask_no_return_near', 'in_mask_returns_not_behind', 'outside_mask', 'off_image')


def classify_violations(P: np.ndarray, ends: np.ndarray, frame, mask: np.ndarray, ret_uvz: np.ndarray, vol_pts: Optional[np.ndarray] = None,
                        vs: float = 0.1, radius_px: float = 30.0, margin: float = 0.2) -> Dict[str, float]:
    """
    Would a depth constraint on the mask pixels have flagged each violating point P (M, 3)?
    Each point is projected into the image (u, v, depth z). Classes (fractions of the violating points):
      implied                      inside the mask AND a LiDAR return inside the mask within `radius_px` lies at least `margin`
                                   BEHIND the point: measured depth at that pixel says the surface is further back. A method that makes
                                   the mesh surface match the in-mask depth already excludes this point.
      in_mask_no_return_near       inside the mask but no in-mask return within `radius_px`: sparse LiDAR gives no depth there
      in_mask_returns_not_behind   inside the mask, returns nearby, but none behind the point: depth matching would not object
      outside_mask                 projects outside the object's own mask: the silhouette is not respected there
      off_image                    behind the camera or outside the image
    `ret_uvz` (R, 3): pixel u, v and camera depth z of the in-mask LiDAR returns. Also returns the share of the points whose
    beam ENDS inside the mesh volume (`end_in_mesh`: the beam went through the surface into the object = glass / gaps, not overshoot).
    """
    from scipy.spatial import cKDTree
    out = {f'red_{c}': 0.0 for c in CLASSES}
    out['red_n'] = int(len(P)); out['red_end_in_mesh'] = float('nan')
    if len(P) == 0:
        return out
    H, W = mask.shape
    pc = (frame.R_c2e.T @ (P - frame.t_c2e).T).T
    z = pc[:, 2]
    zs = np.where(z > 0.05, z, 1.0)
    u = pc[:, 0] / zs * frame.K[0, 0] + frame.K[0, 2]
    v = pc[:, 1] / zs * frame.K[1, 1] + frame.K[1, 2]
    on = (z > 0.05) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    cls = np.full(len(P), 'off_image', dtype=object)
    ui, vi = np.round(u[on]).astype(int).clip(0, W - 1), np.round(v[on]).astype(int).clip(0, H - 1)
    inm = mask[vi, ui]
    sub = np.full(on.sum(), 'outside_mask', dtype=object)
    if len(ret_uvz):
        tree = cKDTree(ret_uvz[:, :2])
        k = min(8, len(ret_uvz))
        dist, nb = tree.query(np.stack([u[on], v[on]], 1), k=k, distance_upper_bound=radius_px)
        dist, nb = dist.reshape(-1, k), nb.reshape(-1, k)
        valid = np.isfinite(dist)
        zr = np.where(valid, ret_uvz[np.minimum(nb, len(ret_uvz) - 1), 2], -np.inf)
        behind = (zr >= z[on][:, None] + margin).any(axis=1)
        any_near = valid.any(axis=1)
    else:
        behind = np.zeros(on.sum(), bool); any_near = np.zeros(on.sum(), bool)
    sub[inm & behind] = 'implied'
    sub[inm & ~behind & any_near] = 'in_mask_returns_not_behind'
    sub[inm & ~any_near] = 'in_mask_no_return_near'
    cls[on] = sub
    for c in CLASSES:
        out[f'red_{c}'] = float((cls == c).mean())
    if vol_pts is not None and len(vol_pts):
        out['red_end_in_mesh'] = float((cKDTree(vol_pts).query(ends)[0] <= vs).mean())
    return out
