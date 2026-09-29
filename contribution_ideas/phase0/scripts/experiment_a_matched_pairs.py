"""
Regime A/B object-matched absolute free-voxel comparison.

Complements experiment_a_batch.py's distributional excess_free% metric (which
normalises each box by its own volume, and compares against a category/range-bin
GT median — no object matching). This script instead matches individual GT
boxes to individual predictions and compares their ABSOLUTE free-voxel counts,
which is not diluted by a wildly oversized box's own inflated denominator.

Regime C is excluded: too little LiDAR signal (mostly unknown space) and GT/pred
boxes are too unreliably close in range to match with any confidence there.

Matching, per (frame, category):
  1. Only GT boxes whose OWN regime is A or B are eligible (GT position is the
     more trustworthy signal for scoping than a possibly-distorted prediction).
  2. Global greedy assignment by ascending 2-D spatial distance between box
     centres: every (GT, pred) pair of the same category is considered in order
     of increasing distance, and a pair is accepted only if both its GT and its
     prediction are still unclaimed. This does not privilege either side's size,
     and does not filter on any size-plausibility gate — the hyperinflated cases
     are exactly the point of this experiment, not something to exclude.
  3. MAX_MATCH_DISTANCE_M (5 m) cuts off implausible pairs: without it, once the
     close candidates are used up, whatever's left gets force-paired regardless
     of how far apart the objects actually are (observed: a pedestrian at 7 m
     matched to one at 39 m before this was added).
  4. Leftover predictions/GT (count mismatch, or beyond the cutoff) are left
     unmatched and counted, not force-matched — that's a separate false-
     positive/false-negative question, not this experiment's job.

Per matched pair:
  absolute_diff   = n_free_pred - n_free_gt
  relative_excess = absolute_diff / n_free_gt   (only when n_free_gt >= 1;
                                                  undefined below that)

Usage:
    python experiment_a_matched_pairs.py --dataset ecp
    python experiment_a_matched_pairs.py --dataset nuscenes_mini
"""

import argparse, json, os, sys, time
from collections import defaultdict
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import experiment_a_batch as batch
from pyquaternion import Quaternion

MIN_N_FREE_GT_FOR_RELATIVE = 1
MAX_MATCH_DISTANCE_M = 5.0   # reject a GT<->pred pair as implausible beyond this

NUSCENES_GT_TO_PRED = {
    'vehicle.car':                          'car',
    'vehicle.truck':                        'truck',
    'vehicle.bus.bendy':                    'bus',
    'vehicle.bus.rigid':                    'bus',
    'vehicle.trailer':                      'trailer',
    'vehicle.motorcycle':                   'motorcycle',
    'vehicle.bicycle':                      'bicycle',
    'vehicle.construction':                 'construction_vehicle',
    'human.pedestrian.adult':               'pedestrian',
    'human.pedestrian.child':               'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'human.pedestrian.personal_mobility':   'pedestrian',
    'human.pedestrian.police_officer':      'pedestrian',
}


def gt_cat_to_pred(c):
    return NUSCENES_GT_TO_PRED.get(c, c)   # ECP GT already uses prediction names


def regime_of(dist):
    return 'A' if dist < 15.0 else ('B' if dist < 30.0 else 'C')


def bev_area(size):
    return float(size[0]) * float(size[1])   # width x length


def process_dataset(dataset: str):
    from nuscenes.nuscenes import NuScenes

    if dataset == 'ecp':
        data_root = batch.ECP_DATA_ROOT
        version   = 'v1.0-trainval'
        pred_file = os.path.join(batch.PRED_DIR_ECP, 'autolabel_ecp_annotated_8class.json')
    else:
        data_root = batch.NUSCENES_DATA_ROOT
        version   = 'v1.0-mini'
        pred_file = os.path.join(batch.PRED_DIR_NUSCENES, 'autolabel_nuscenes_mini_train_8class.json')

    print(f'\nLoading nuScenes ({version}) from {data_root}')
    nusc = NuScenes(version=version, dataroot=data_root, verbose=False)

    with open(pred_file) as f:
        pred_data = json.load(f)
    sample_tokens = list(pred_data['results'].keys())
    print(f'Processing {len(sample_tokens)} frames')

    if batch.HAS_NUMBA:
        _log = np.zeros((10, 10, 10), dtype=np.float32)
        _pts = np.array([[1, 0, 0]], dtype=np.float32)
        _so  = np.array([0, 0, 0], dtype=np.float32)
        batch.raytrace(_log, _pts, _so, np.zeros(3, dtype=np.float32),
                        batch.VOXEL_SIZE, batch.LO_FREE, batch.LO_OCC, batch.LO_MIN, batch.LO_MAX)
        print('Numba JIT warmed up.')

    log_odds = np.zeros((batch.NX, batch.NY, batch.NZ), dtype=np.float32)
    rows = []
    n_unmatched_pred = 0
    n_unmatched_gt = 0
    t0 = time.time()

    for frame_i, sample_token in enumerate(sample_tokens):
        sample = nusc.get('sample', sample_token)

        lidar_token = sample['data']['LIDAR_TOP']
        lidar_sd = nusc.get('sample_data', lidar_token)
        ep = nusc.get('ego_pose', lidar_sd['ego_pose_token'])
        cs = nusc.get('calibrated_sensor', lidar_sd['calibrated_sensor_token'])
        t_e2g = np.array(ep['translation']); R_e2g = Quaternion(ep['rotation']).rotation_matrix
        t_l2e = np.array(cs['translation']); R_l2e = Quaternion(cs['rotation']).rotation_matrix

        lidar_path = os.path.join(data_root, lidar_sd['filename'])
        raw = np.fromfile(lidar_path, dtype=np.float32)
        stride = 5   # both datasets are (x,y,z,intensity,ring) — see experiment_a_batch.py note
        pts_lidar = raw.reshape(-1, stride)[:, :3]
        pts_ego = (R_l2e @ pts_lidar.T).T + t_l2e
        sensor_origin = t_l2e.astype(np.float32)
        ranges = np.linalg.norm(pts_ego - sensor_origin, axis=1)
        mask = (ranges >= batch.RAY_MIN_RANGE) & (ranges <= batch.RAY_MAX_RANGE)
        pts_cast = pts_ego[mask].astype(np.float32)

        log_odds[:] = 0.0
        batch.raytrace(log_odds, pts_cast, sensor_origin, batch.GRID_ORIGIN, batch.VOXEL_SIZE,
                        batch.LO_FREE, batch.LO_OCC, batch.LO_MIN, batch.LO_MAX)

        # ── GT boxes, regime A/B only ────────────────────────────────────────
        gt_by_cat = defaultdict(list)
        for ann_token in sample['anns']:
            ann = nusc.get('sample_annotation', ann_token)
            cat = gt_cat_to_pred(ann['category_name'])
            if cat is None:
                continue
            center_e = R_e2g.T @ (np.array(ann['translation']) - t_e2g)
            R_box_e  = R_e2g.T @ Quaternion(ann['rotation']).rotation_matrix
            yaw  = float(np.arctan2(R_box_e[1, 0], R_box_e[0, 0]))
            size = np.array(ann['size'])
            dist = float(np.linalg.norm(center_e[:2]))
            regime = regime_of(dist)
            if regime not in ('A', 'B'):
                continue
            gt_by_cat[cat].append(dict(center_e=center_e, size=size, yaw=yaw,
                                        dist=dist, regime=regime))

        # ── Predictions, all of them (matching decides which regime applies) ──
        pred_by_cat = defaultdict(list)
        for b in pred_data['results'].get(sample_token, []):
            cat = b['detection_name']
            center_e = R_e2g.T @ (np.array(b['translation']) - t_e2g)
            R_box_e  = R_e2g.T @ Quaternion(b['rotation']).rotation_matrix
            yaw  = float(np.arctan2(R_box_e[1, 0], R_box_e[0, 0]))
            size = np.array(b['size'])
            dist = float(np.linalg.norm(center_e[:2]))
            pred_by_cat[cat].append(dict(center_e=center_e, size=size, yaw=yaw, dist=dist))

        for cat in set(gt_by_cat) | set(pred_by_cat):
            gts = gt_by_cat.get(cat, [])
            preds = pred_by_cat.get(cat, [])
            if not gts or not preds:
                n_unmatched_pred += len(preds)
                n_unmatched_gt += len(gts)
                continue

            # Global greedy matching by ascending spatial distance (2-D ego-frame
            # distance between box centres): consider every (GT, pred) pair for this
            # category, process closest-first, and take a pair only if both its GT
            # and its prediction are still unclaimed. This is the standard greedy
            # bipartite-matching heuristic — simpler and more robust than sorting by
            # one side's size first, and it does not privilege any particular object.
            #
            # MAX_MATCH_DISTANCE_M is essential here: without a cutoff, once the closer
            # candidates are used up, whatever's left gets force-paired regardless of
            # how far apart they actually are (e.g. a pedestrian at 7m matched to one
            # at 39m) purely because the algorithm must assign *something*. Beyond the
            # cutoff a pair is not a plausible same-object match, so leave both unmatched.
            all_pairs = sorted(
                (
                    (float(np.linalg.norm(g['center_e'][:2] - p['center_e'][:2])), gi, pi)
                    for gi, g in enumerate(gts) for pi, p in enumerate(preds)
                ),
                key=lambda t: t[0],
            )
            used_gt, used_pred = set(), set()
            for dist, gi, pi in all_pairs:
                if dist > MAX_MATCH_DISTANCE_M:
                    break   # sorted ascending -- nothing further can be valid either
                if gi in used_gt or pi in used_pred:
                    continue
                used_gt.add(gi); used_pred.add(pi)

                g, p = gts[gi], preds[pi]
                stat_g = batch.query_box_freespace_obb(log_odds, g['center_e'], g['size'], g['yaw'])
                stat_p = batch.query_box_freespace_obb(log_odds, p['center_e'], p['size'], p['yaw'])
                if stat_g is None or stat_p is None:
                    continue

                n_free_gt, n_free_pred = stat_g['n_free'], stat_p['n_free']
                abs_diff = n_free_pred - n_free_gt
                rel_excess = (abs_diff / n_free_gt) if n_free_gt >= MIN_N_FREE_GT_FOR_RELATIVE else None

                rows.append({
                    'dataset': dataset, 'sample_token': sample_token, 'category': cat,
                    'gt_regime': g['regime'],
                    'gt_range_m': round(g['dist'], 2), 'pred_range_m': round(p['dist'], 2),
                    'range_delta_m': round(abs(g['dist'] - p['dist']), 2),
                    'spatial_delta_m': round(dist, 2),
                    'gt_bev_area': round(bev_area(g['size']), 3),
                    'pred_bev_area': round(bev_area(p['size']), 3),
                    'bev_area_ratio': round(bev_area(p['size']) / max(bev_area(g['size']), 1e-6), 2),
                    'n_total_gt': stat_g['n_total'], 'n_free_gt': n_free_gt,
                    'n_total_pred': stat_p['n_total'], 'n_free_pred': n_free_pred,
                    'absolute_diff': abs_diff,
                    'relative_excess': rel_excess,
                })

            n_unmatched_pred += len(preds) - len(used_pred)
            n_unmatched_gt += len(gts) - len(used_gt)

        if (frame_i + 1) % 10 == 0 or frame_i == 0:
            elapsed = time.time() - t0
            rate = (frame_i + 1) / elapsed
            eta = (len(sample_tokens) - frame_i - 1) / rate
            print(f'  frame {frame_i+1}/{len(sample_tokens)}  ({rate:.1f} fps)  ETA {eta:.0f}s')

    elapsed = time.time() - t0
    os.makedirs(batch.OUT_DIR, exist_ok=True)
    _vtag = '' if abs(batch.VOXEL_SIZE - 0.05) < 1e-9 else f'_vox{batch.VOXEL_SIZE:g}'
    df = pd.DataFrame(rows)
    out_path = os.path.join(batch.OUT_DIR, f'experiment_a_matched_pairs_{dataset}{_vtag}.csv')
    df.to_csv(out_path, index=False)
    print(f'\n{dataset}: {len(df)} matched pairs -> {out_path}')
    print(f'  unmatched predictions: {n_unmatched_pred}   unmatched GT: {n_unmatched_gt}')
    print(f'Total time: {elapsed:.1f}s')
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['ecp', 'nuscenes_mini'], required=True)
    parser.add_argument('--voxel-size', type=float, default=batch.VOXEL_SIZE,
                        help='Isotropic voxel size in metres (default: matches '
                             'experiment_a_batch.py). Output tagged _vox<N> unless default.')
    args = parser.parse_args()

    if args.voxel_size != batch.VOXEL_SIZE:
        batch.VOXEL_SIZE = args.voxel_size
        batch.NX = int((batch.GRID_X_MAX - batch.GRID_X_MIN) / batch.VOXEL_SIZE)
        batch.NY = int((batch.GRID_Y_MAX - batch.GRID_Y_MIN) / batch.VOXEL_SIZE)
        batch.NZ = int((batch.GRID_Z_MAX - batch.GRID_Z_MIN) / batch.VOXEL_SIZE)
        print(f'Voxel size override: {batch.VOXEL_SIZE} m  ->  grid {batch.NX} x {batch.NY} x '
              f'{batch.NZ} = {batch.NX*batch.NY*batch.NZ/1e6:.1f} M voxels')

    process_dataset(args.dataset)
