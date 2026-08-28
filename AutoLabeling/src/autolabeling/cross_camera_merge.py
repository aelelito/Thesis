"""
Cross-camera duplicate suppression.

Two adjacent cameras will often detect the same physical object near their shared
border.  This module identifies those duplicates and keeps only the detection with
the larger SAM3 mask (more image evidence), suppressing the other.

Matching strategy (two passes, per adjacent camera pair, per keyframe):
  Main pass   — BEV OBB footprint overlap >= bev_overlap_thresh, greedy by
                descending overlap.  Body OBB corners are in camera space and are
                transformed to ego space for the comparison.
  Fallback    — ResNet18 appearance embedding cosine similarity for any border
                candidate still unmatched after the main pass (greedy, no
                minimum threshold — every remaining border detection gets a
                partner if one exists on the adjacent side).

All hyperparameters come from the config (cross_camera_merge section).
"""
import functools
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torchvision.models as _tv_models
import torchvision.transforms as _tv_T
from shapely.geometry import Polygon as _ShapelyPolygon


# ── Appearance embedding ──────────────────────────────────────────────────────

@functools.lru_cache(maxsize=1)
def _embed_model(device: str = 'cuda'):
    """ResNet18 truncated at the global-pool layer — returns 512-d features."""
    m = _tv_models.resnet18(weights=_tv_models.ResNet18_Weights.DEFAULT)
    m = torch.nn.Sequential(*list(m.children())[:-1])
    m.eval()
    return m.to(device)


_embed_tf = _tv_T.Compose([
    _tv_T.ToPILImage(),
    _tv_T.Resize((112, 112)),
    _tv_T.ToTensor(),
    _tv_T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _get_embed(r: dict, img_rgb: np.ndarray, cache: dict, device: str) -> Optional[np.ndarray]:
    """
    Compute or retrieve the cached 512-d normalised embedding for a detection.

    Parameters
    ----------
    r       : detection dict with 'binary_mask'
    img_rgb : (H, W, 3) uint8 full camera image for this frame
    cache   : per-frame cache dict; keyed by id(r)
    device  : torch device string

    Returns
    -------
    (512,) float32 unit-norm vector, or None if the mask is empty
    """
    key = id(r)
    if key in cache:
        return cache[key]
    mask = r.get('binary_mask')
    if mask is None:
        cache[key] = None
        return None
    ys, xs = np.where(mask)
    if len(ys) == 0:
        cache[key] = None
        return None
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    crop = img_rgb[y0:y1 + 1, x0:x1 + 1].copy()
    crop[~mask[y0:y1 + 1, x0:x1 + 1]] = 0
    with torch.no_grad(), torch.autocast(device, enabled=False):
        feat = (_embed_model(device)(_embed_tf(crop).unsqueeze(0).float().to(device))
                .squeeze().cpu().numpy())
    feat = feat / (np.linalg.norm(feat) + 1e-8)
    cache[key] = feat
    return feat


# ── BEV OBB overlap ───────────────────────────────────────────────────────────

def _bev_overlap(corners_a: np.ndarray, corners_b: np.ndarray) -> float:
    """
    Compute BEV footprint overlap between two oriented bounding boxes.

    Parameters
    ----------
    corners_a, corners_b : (8, 3) ego-frame OBB corners

    Returns
    -------
    overlap / area(smaller OBB)  in [0, 1]
    """
    pts_a = corners_a[:, :2]  # XY projection
    pts_b = corners_b[:, :2]
    poly_a = _ShapelyPolygon(pts_a).convex_hull
    poly_b = _ShapelyPolygon(pts_b).convex_hull
    smaller = min(poly_a.area, poly_b.area)
    if smaller < 1e-6:
        return 0.0
    return float(poly_a.intersection(poly_b).area / smaller)


def _corners_ego(r: dict, R_c2e: np.ndarray, t_c2e: np.ndarray) -> np.ndarray:
    """
    Return OBB corners in ego frame.

    Body OBBs are stored in camera space and need to be rotated/translated.
    Object OBBs are already in ego frame.
    """
    corners = np.array(r['obb_corners'], dtype=np.float64)
    if r.get('_xc_kind') == 'body':
        corners = (R_c2e @ corners.T).T + t_c2e
    return corners


# ── Class compatibility ───────────────────────────────────────────────────────

def _classes_compatible(label_a: str, label_b: str,
                        compatible_groups: List) -> bool:
    """True if the two class labels are the same or belong to a compatible group."""
    if label_a == label_b:
        return True
    for grp in compatible_groups:
        if label_a in grp and label_b in grp:
            return True
    return False


def _get_label(r: dict) -> str:
    return r.get('prompt', 'pedestrian')


# ── Main entry point ──────────────────────────────────────────────────────────

def cross_camera_merge(
    body_results_all: Dict[str, Dict[int, list]],
    obj_results_all:  Dict[str, Dict[int, list]],
    frames_per_cam:   Dict[str, list],
    cfg,
) -> Tuple[Dict, Dict]:
    """
    Suppress cross-camera duplicate detections in-place.

    Parameters
    ----------
    body_results_all : {cam: {frame_idx_in_list: [body result dicts]}}
    obj_results_all  : {cam: {frame_idx_in_list: [obj  result dicts]}}
    frames_per_cam   : {cam: [FrameRecord]}  — same length for every camera
    cfg              : cross_camera_merge config namespace with fields:
                         border_threshold      float   (0–1 fraction of image width)
                         bev_overlap_thresh    float   (min overlap / area_smaller)
                         compatible_class_groups  list of lists of str
                         adjacent_cam_pairs    list of [left_cam, right_cam]

    Returns
    -------
    (body_results_all, obj_results_all)  — same dicts, duplicates removed
    """
    border_threshold   = float(getattr(cfg, 'border_threshold',   0.20))
    bev_overlap_thresh = float(getattr(cfg, 'bev_overlap_thresh', 0.10))
    compatible_groups  = [set(g) for g in getattr(cfg, 'compatible_class_groups', [])]
    adjacent_pairs     = [tuple(p) for p in getattr(cfg, 'adjacent_cam_pairs',    [])]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cameras  = list(frames_per_cam.keys())
    n_frames = len(frames_per_cam[cameras[0]])

    # Warm up the embed model once (avoids first-frame latency in the fallback).
    _embed_model(device)

    total_suppressed = 0

    for i in range(n_frames):
        _embed_cache: dict = {}   # fresh per keyframe

        # Tag all detections so _corners_ego knows the coordinate space
        for cam in cameras:
            for r in body_results_all.get(cam, {}).get(i, []):
                r['_xc_kind'] = 'body'
            for r in obj_results_all.get(cam, {}).get(i, []):
                r['_xc_kind'] = 'object'

        suppress: set = set()   # (cam, kind, local_idx)

        for left_cam, right_cam in adjacent_pairs:
            if left_cam not in frames_per_cam or right_cam not in frames_per_cam:
                continue

            frame_l = frames_per_cam[left_cam][i]
            frame_r = frames_per_cam[right_cam][i]
            W_l = frame_l.img_width or int(frame_l.K[0, 2] * 2)  # fallback: 2*cx
            W_r = frame_r.img_width or int(frame_r.K[0, 2] * 2)

            # ── Collect border candidates ────────────────────────────────────
            left_cands  = []   # (kind, local_idx, result_dict)
            right_cands = []

            for _kind, _cam_results in [
                ('body',   body_results_all.get(left_cam, {}).get(i, [])),
                ('object', obj_results_all.get(left_cam,  {}).get(i, [])),
            ]:
                for _idx, _r in enumerate(_cam_results):
                    if (left_cam, _kind, _idx) in suppress:
                        continue
                    _mask = _r.get('binary_mask')
                    if _mask is None:
                        continue
                    _xs = np.where(_mask)[1]
                    if len(_xs) and _xs.max() / W_l >= (1.0 - border_threshold):
                        left_cands.append((_kind, _idx, _r))

            for _kind, _cam_results in [
                ('body',   body_results_all.get(right_cam, {}).get(i, [])),
                ('object', obj_results_all.get(right_cam,  {}).get(i, [])),
            ]:
                for _idx, _r in enumerate(_cam_results):
                    if (right_cam, _kind, _idx) in suppress:
                        continue
                    _mask = _r.get('binary_mask')
                    if _mask is None:
                        continue
                    _xs = np.where(_mask)[1]
                    if len(_xs) and _xs.min() / W_r <= border_threshold:
                        right_cands.append((_kind, _idx, _r))

            if not left_cands or not right_cands:
                continue

            print(f'  [cross-cam {left_cam}|{right_cam}]  frame {i}: '
                  f'left={len(left_cands)}  right={len(right_cands)}')

            # ── Pre-compute all class-compatible BEV overlap scores ──────────
            _all_pairs = []   # (overlap, li, ri)
            for _li, (_lk, _li_idx, _lr) in enumerate(left_cands):
                for _ri, (_rk, _ri_idx, _rr) in enumerate(right_cands):
                    if not _classes_compatible(_get_label(_lr), _get_label(_rr),
                                               compatible_groups):
                        continue
                    _ov = _bev_overlap(
                        _corners_ego(_lr, frame_l.R_c2e, frame_l.t_c2e),
                        _corners_ego(_rr, frame_r.R_c2e, frame_r.t_c2e),
                    )
                    _all_pairs.append((_ov, _li, _ri))
            _all_pairs.sort(key=lambda x: -x[0])

            # matched[li] = (ri, score, score_label)
            matched:   Dict[int, tuple] = {}
            matched_r: set              = set()

            # ── Main pass: BEV overlap ≥ threshold ──────────────────────────
            for _ov, _li, _ri in _all_pairs:
                if _ov < bev_overlap_thresh:
                    break
                if _li in matched or _ri in matched_r:
                    continue
                matched[_li]   = (_ri, _ov, f'BEV={_ov:.1%}')
                matched_r.add(_ri)
                print(f'    [main]     li={_li} ri={_ri}  BEV={_ov:.1%}')

            # ── Fallback: embed similarity for remaining unmatched ───────────
            _unmatched_l = [j for j in range(len(left_cands))  if j not in matched]
            _unmatched_r = [j for j in range(len(right_cands)) if j not in matched_r]

            if _unmatched_l and _unmatched_r:
                print(f'    [fallback] {len(_unmatched_l)} left, {len(_unmatched_r)} right'
                      f' — using embed similarity')
                img_rgb_l, _ = frame_l.load_images()
                img_rgb_r, _ = frame_r.load_images()

                _fb_pairs = []
                for _li in _unmatched_l:
                    _lk, _li_idx, _lr = left_cands[_li]
                    for _ri in _unmatched_r:
                        _rk, _ri_idx, _rr = right_cands[_ri]
                        if not _classes_compatible(_get_label(_lr), _get_label(_rr),
                                                   compatible_groups):
                            continue
                        _ea = _get_embed(_lr, img_rgb_l, _embed_cache, device)
                        _eb = _get_embed(_rr, img_rgb_r, _embed_cache, device)
                        if _ea is None or _eb is None:
                            _sim = 0.0
                        else:
                            _sim = float(np.dot(_ea, _eb))
                        _fb_pairs.append((_sim, _li, _ri))

                _fb_pairs.sort(key=lambda x: -x[0])
                _fb_used_r: set = set()
                for _sim, _li, _ri in _fb_pairs:
                    if _li in matched or _ri in _fb_used_r:
                        continue
                    matched[_li]   = (_ri, _sim, f'sim={_sim:.2f} [fallback]')
                    _fb_used_r.add(_ri)
                    print(f'    [fallback] li={_li} ri={_ri}  sim={_sim:.2f}')

            # ── Apply matches → mark smaller-mask detection for suppression ──
            for _li, (_ri, _score, _score_label) in matched.items():
                _lk, _li_idx, _lr = left_cands[_li]
                _rk, _ri_idx, _rr = right_cands[_ri]
                mask_px_l = int(_lr['binary_mask'].sum())
                mask_px_r = int(_rr['binary_mask'].sum())
                if mask_px_l >= mask_px_r:
                    suppress.add((right_cam, _rk, _ri_idx))
                    kept_cam, drop_cam = left_cam, right_cam
                else:
                    suppress.add((left_cam, _lk, _li_idx))
                    kept_cam, drop_cam = right_cam, left_cam
                print(f'    [keep {kept_cam} / drop {drop_cam}]  '
                      f'{_get_label(_lr)} <-> {_get_label(_rr)}  '
                      f'{_score_label}  px {mask_px_l} vs {mask_px_r}')

        # ── Apply suppression for this keyframe ──────────────────────────────
        n_frame_suppressed = 0
        for cam in cameras:
            _b0 = len(body_results_all.get(cam, {}).get(i, []))
            _o0 = len(obj_results_all.get(cam,  {}).get(i, []))
            if cam in body_results_all and i in body_results_all[cam]:
                body_results_all[cam][i] = [
                    r for j, r in enumerate(body_results_all[cam][i])
                    if (cam, 'body', j) not in suppress
                ]
            if cam in obj_results_all and i in obj_results_all[cam]:
                obj_results_all[cam][i] = [
                    r for j, r in enumerate(obj_results_all[cam][i])
                    if (cam, 'object', j) not in suppress
                ]
            _db = _b0 - len(body_results_all.get(cam, {}).get(i, []))
            _do = _o0 - len(obj_results_all.get(cam,  {}).get(i, []))
            n_frame_suppressed += _db + _do
            if _db + _do:
                print(f'  [{cam}] frame {i}: suppressed {_db} body, {_do} object(s)')

        total_suppressed += n_frame_suppressed

    print(f'\nCross-camera merge: {total_suppressed} duplicate(s) suppressed '
          f'across {n_frames} keyframe(s).')
    return body_results_all, obj_results_all
