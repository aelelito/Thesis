"""
Per-object diagnostics against nuScenes GT, for comparing SAM3D Objects pointmap modes.

The GT box of a detection is chosen by how well its projected 3D box overlaps the SAM3 mask, which does
not depend on the pointmap mode -- so every mode is scored against the SAME GT box for the same object.
"""
import cv2
import numpy as np
from pyquaternion import Quaternion

from .geometry import project


def gt_boxes_ego(nusc, sample_token: str, R_e2g: np.ndarray, t_e2g: np.ndarray) -> list:
    """All GT annotations of a keyframe, in that keyframe's ego frame."""
    out = []
    for tok in nusc.get('sample', sample_token)['anns']:
        a = nusc.get('sample_annotation', tok)
        w, l, h = a['size']                                    # nuScenes size = [width, length, height]
        center = R_e2g.T @ (np.array(a['translation'], np.float64) - t_e2g)
        R = R_e2g.T @ Quaternion(a['rotation']).rotation_matrix
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
        lx = np.array([1, 1, 1, 1, -1, -1, -1, -1]) * l / 2
        ly = np.array([1, -1, -1, 1, 1, -1, -1, 1]) * w / 2
        lz = np.array([1, 1, -1, -1, 1, 1, -1, -1]) * h / 2
        corners_3d = center + (R @ np.stack([lx, ly, lz])).T
        cy, sy, hl, hw = np.cos(yaw), np.sin(yaw), l / 2, w / 2
        corners_2d = center[:2] + np.array([
            [+hl * cy - hw * sy, +hl * sy + hw * cy], [+hl * cy + hw * sy, +hl * sy - hw * cy],
            [-hl * cy + hw * sy, -hl * sy - hw * cy], [-hl * cy - hw * sy, -hl * sy + hw * cy]])
        out.append(dict(center=center, size=(w, l, h), yaw=yaw, corners_3d=corners_3d, corners_2d=corners_2d,
                        category=a['category_name'], num_lidar_pts=a['num_lidar_pts']))
    return out


def associate_gt(gt_boxes: list, binary_mask: np.ndarray, frame, category_prefix: str = 'vehicle',
                 min_iou: float = 0.2):
    """
    Index of the GT box (of the given top-level category) whose projected 3D box overlaps `binary_mask`
    best, and that IoU; (None, best_iou) if nothing overlaps at least `min_iou`. Boxes reaching behind the
    camera are skipped.
    """
    H, W = binary_mask.shape
    m = binary_mask.astype(bool)
    best, best_iou = None, min_iou
    for i, g in enumerate(gt_boxes):
        if not g['category'].startswith(category_prefix):
            continue
        pc = (frame.R_c2e.T @ (g['corners_3d'] - frame.t_c2e).T).T
        if (pc[:, 2] < 0.5).any():
            continue
        hull = cv2.convexHull(project(pc, frame.K).astype(np.int32))
        poly = np.zeros((H, W), np.uint8)
        cv2.fillConvexPoly(poly, hull, 1)
        poly = poly.astype(bool)
        union = (poly | m).sum()
        iou = float((poly & m).sum() / union) if union else 0.0
        if iou > best_iou:
            best, best_iou = i, iou
    return best, best_iou


def object_errors(obb_center, obb_dims, obb_yaw, gt: dict) -> dict:
    """
    Errors of one predicted OBB (ego frame; dims = [length, width, height]) against one GT box.

    center_err  BEV distance [m]                range_err  signed error along the ray from the ego origin
                                                            to the GT center [m] (+ = predicted too far)
    lat_err     BEV error across that ray [m]   long/short_ratio  predicted / GT footprint long / short side
    h_ratio     predicted / GT height           yaw_err   heading error in degrees, 0-90 (front/back flips
                                                            are ignored: the sign of the heading is ambiguous)
    """
    d = np.asarray(obb_center[:2], np.float64) - gt['center'][:2]
    u = gt['center'][:2] / max(np.linalg.norm(gt['center'][:2]), 1e-6)
    radial = float(d @ u)
    w, l, h = gt['size']
    pl, ps = max(obb_dims[0], obb_dims[1]), min(obb_dims[0], obb_dims[1])
    dy = (obb_yaw - gt['yaw']) % np.pi
    return dict(center_err=float(np.linalg.norm(d)), range_err=radial,
                lat_err=float(abs(d[0] * u[1] - d[1] * u[0])),
                long_ratio=float(pl / max(l, w)), short_ratio=float(ps / min(l, w)),
                h_ratio=float(obb_dims[2] / h), yaw_err=float(np.degrees(min(dy, np.pi - dy))),
                gt_range=float(np.hypot(*gt['center'][:2])))


def associate_gt_lidar(gt_boxes: list, binary_mask: np.ndarray, frame, pts_ego: np.ndarray,
                       category_prefix: str = 'vehicle', margin: float = 0.4, min_pts: int = 3,
                       ambiguity_ratio: float = 0.5, min_fraction: float = 0.3):
    """
    GT box of a detection from LiDAR: the returns that fall on the mask's pixels lie on the visible surface of
    the object, so the GT box (3D) that contains most of them is that object. Two objects that overlap in the
    image sit at different depths and therefore in different 3D boxes -- unlike 2D mask overlap this is not
    fooled by occlusion.

    pts_ego : (N, 3) LiDAR in the ego frame; a SINGLE sweep is best (multi-sweep clouds smear moving objects).
    margin  : boxes are enlarged by this much on every side [m].
    Returns (index or None, fraction of the in-mask points inside that box, ambiguous), where ambiguous means
    the runner-up box also holds >= ambiguity_ratio of the best box's points. None if fewer than min_pts points
    fall in the mask or in the best box, or if less than `min_fraction` of the in-mask points are inside it (a small
    mask whose few points mostly belong to some other, nearer object).
    """
    H, W = binary_mask.shape
    pc = (frame.R_c2e.T @ (pts_ego - frame.t_c2e).T).T
    front = pc[:, 2] > 0.5
    pc, pe = pc[front], pts_ego[front]
    u = np.round(pc[:, 0] / pc[:, 2] * frame.K[0, 0] + frame.K[0, 2]).astype(int)
    v = np.round(pc[:, 1] / pc[:, 2] * frame.K[1, 1] + frame.K[1, 2]).astype(int)
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    pe, u, v = pe[ok], u[ok], v[ok]
    pe = pe[binary_mask[v, u].astype(bool)]
    if len(pe) < min_pts:
        return None, 0.0, False
    counts = np.zeros(len(gt_boxes))
    for i, g in enumerate(gt_boxes):
        if not g['category'].startswith(category_prefix):
            continue
        c, s = np.cos(-g['yaw']), np.sin(-g['yaw'])
        d = pe - g['center']
        lx, ly, lz = c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1], d[:, 2]
        w, l, h = g['size']
        counts[i] = ((np.abs(lx) <= l / 2 + margin) & (np.abs(ly) <= w / 2 + margin) & (np.abs(lz) <= h / 2 + margin)).sum()
    best = int(np.argmax(counts))
    if counts[best] < min_pts or counts[best] / len(pe) < min_fraction:
        return None, 0.0, False
    second = np.partition(counts, -2)[-2] if len(counts) > 1 else 0.0
    return best, float(counts[best] / len(pe)), bool(second >= ambiguity_ratio * counts[best])


def resolve_duplicate_gt(claims: dict) -> set:
    """
    One GT box belongs to one detection. claims: {detection key: (gt index, mask area in px)}. When several detections
    claim the same GT box the one with the LARGEST mask keeps it (a small far mask whose few LiDAR points bleed from a
    nearer object is the usual wrong claimant). Returns the set of detection keys that lose their claim.
    """
    by_gt = {}
    for key, (gi, area) in claims.items():
        by_gt.setdefault(gi, []).append((area, key))
    lose = set()
    for members in by_gt.values():
        if len(members) > 1:
            members.sort(key=lambda t: -t[0])
            lose.update(k for _, k in members[1:])
    return lose
