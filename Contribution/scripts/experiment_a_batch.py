"""
Phase 0.4 + 0.5 — Batch Experiment A across all annotated frames.

For each sample token in the prediction JSON:
  1. Load the LiDAR sweep and build a per-frame free-space voxel grid
  2. Transform predicted boxes to ego frame
  3. Run OBB-based free-space query on every predicted box          → experiment_a_<dataset>.csv
  4. Run the same OBB query on every GT annotation (Phase 0.5)     → experiment_a_gt_baseline_<dataset>.csv

The GT baseline characterises the sensor-geometry floor: how much certified-free
space a correctly-fitted box accumulates at a given range purely from beam density
and voxel resolution.  No matching between predictions and GT is needed.

Usage:
    python experiment_a_batch.py --dataset ecp
    python experiment_a_batch.py --dataset nuscenes_mini
    python experiment_a_batch.py --dataset both
"""

import argparse, json, os, sys, time, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
ECP_DATA_ROOT      = '/media/leander/ECP_Nuscenes_01/output2/ecp2nuscenes'
NUSCENES_DATA_ROOT = '/media/leander/ECP_Nuscenes_01/nuScenes_mini'
PRED_DIR_ECP       = ('/media/leander/ECP_Nuscenes_01/autolabeling/output'
                      '/ecp/lidar_integration_objects_o3_body_b1_all_cameras')
PRED_DIR_NUSCENES  = ('/media/leander/ECP_Nuscenes_01/autolabeling/output'
                      '/nuscenes_mini/lidar_integration_objects_o3_body_b1_all_cameras')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

# ── Voxel grid parameters (must match notebook) ────────────────────────────
VOXEL_SIZE   = 0.20
GRID_X_MIN, GRID_X_MAX = -50.0, 50.0
GRID_Y_MIN, GRID_Y_MAX = -50.0, 50.0
GRID_Z_MIN, GRID_Z_MAX =  -2.0,  6.0
LO_FREE, LO_OCC        = -0.4,  0.85
LO_MIN,  LO_MAX        = -5.0, 10.0
THRESHOLD_FREE         = -0.5
THRESHOLD_OCC          =  0.5
RAY_MIN_RANGE          =  0.5
RAY_MAX_RANGE          = 80.0
FREE_THRESHOLD         =  0.10   # flag if OBB free% > 10%

NX = int((GRID_X_MAX - GRID_X_MIN) / VOXEL_SIZE)
NY = int((GRID_Y_MAX - GRID_Y_MIN) / VOXEL_SIZE)
NZ = int((GRID_Z_MAX - GRID_Z_MIN) / VOXEL_SIZE)
GRID_ORIGIN = np.array([GRID_X_MIN, GRID_Y_MIN, GRID_Z_MIN], dtype=np.float32)

# ── Numba ray caster (copy from notebook §4) ───────────────────────────────
try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

if HAS_NUMBA:
    @numba.njit(cache=True)
    def _cast_ray(log_odds, ox, oy, oz, dx, dy, dz, t_max,
                  gox, goy, goz, vs, NX, NY, NZ, lo_free, lo_occ, lo_min, lo_max):
        INF = 1e30
        ix = int((ox - gox) / vs)
        iy = int((oy - goy) / vs)
        iz = int((oz - goz) / vs)
        if ix < 0 or ix >= NX or iy < 0 or iy >= NY or iz < 0 or iz >= NZ:
            return
        sx = 1 if dx > 0.0 else (-1 if dx < 0.0 else 0)
        sy = 1 if dy > 0.0 else (-1 if dy < 0.0 else 0)
        sz = 1 if dz > 0.0 else (-1 if dz < 0.0 else 0)
        tdx = (vs / abs(dx)) if abs(dx) > 1e-12 else INF
        tdy = (vs / abs(dy)) if abs(dy) > 1e-12 else INF
        tdz = (vs / abs(dz)) if abs(dz) > 1e-12 else INF
        if sx > 0: tmx = ((gox + (ix+1)*vs) - ox) / dx
        elif sx < 0: tmx = ((gox + ix*vs) - ox) / dx
        else: tmx = INF
        if sy > 0: tmy = ((goy + (iy+1)*vs) - oy) / dy
        elif sy < 0: tmy = ((goy + iy*vs) - oy) / dy
        else: tmy = INF
        if sz > 0: tmz = ((goz + (iz+1)*vs) - oz) / dz
        elif sz < 0: tmz = ((goz + iz*vs) - oz) / dz
        else: tmz = INF
        while True:
            tc = tmx if tmx < tmy else tmy
            if tmz < tc: tc = tmz
            if tc >= t_max:
                v = log_odds[ix, iy, iz] + lo_occ
                log_odds[ix, iy, iz] = lo_max if v > lo_max else (lo_min if v < lo_min else v)
                return
            v = log_odds[ix, iy, iz] + lo_free
            log_odds[ix, iy, iz] = lo_max if v > lo_max else (lo_min if v < lo_min else v)
            if tc == tmx:
                ix += sx; tmx += tdx
            elif tc == tmy:
                iy += sy; tmy += tdy
            else:
                iz += sz; tmz += tdz
            if ix < 0 or ix >= NX or iy < 0 or iy >= NY or iz < 0 or iz >= NZ:
                return

    @numba.njit(cache=True, parallel=False)
    def _raytrace_all(log_odds, pts, so, gox, goy, goz, vs, NX, NY, NZ,
                      lo_free, lo_occ, lo_min, lo_max):
        for i in range(pts.shape[0]):
            ox, oy, oz = so[0], so[1], so[2]
            px, py, pz = pts[i, 0], pts[i, 1], pts[i, 2]
            dx, dy, dz = px - ox, py - oy, pz - oz
            t_max = (dx*dx + dy*dy + dz*dz) ** 0.5
            if t_max < 1e-6:
                continue
            _cast_ray(log_odds, ox, oy, oz, dx/t_max, dy/t_max, dz/t_max,
                      t_max, gox, goy, goz, vs, NX, NY, NZ,
                      lo_free, lo_occ, lo_min, lo_max)

    def raytrace(log_odds, pts, so, grid_origin, vs, lo_free, lo_occ, lo_min, lo_max):
        _raytrace_all(log_odds, pts.astype(np.float32), so.astype(np.float32),
                      grid_origin[0], grid_origin[1], grid_origin[2],
                      vs, log_odds.shape[0], log_odds.shape[1], log_odds.shape[2],
                      lo_free, lo_occ, lo_min, lo_max)
else:
    def raytrace(log_odds, pts, so, grid_origin, vs, lo_free, lo_occ, lo_min, lo_max):
        gox, goy, goz = grid_origin
        NX_, NY_, NZ_ = log_odds.shape
        for pt in pts:
            ox, oy, oz = so
            dx, dy, dz = pt - so
            t_max = np.sqrt(dx**2 + dy**2 + dz**2)
            if t_max < 1e-6: continue
            dx /= t_max; dy /= t_max; dz /= t_max
            n = int(t_max / vs) + 2
            for k in range(n):
                t = k * vs
                x = ox + dx*t; y = oy + dy*t; z = oz + dz*t
                ix = int((x-gox)/vs); iy = int((y-goy)/vs); iz = int((z-goz)/vs)
                if not (0<=ix<NX_ and 0<=iy<NY_ and 0<=iz<NZ_): break
                if t >= t_max:
                    log_odds[ix,iy,iz] = np.clip(log_odds[ix,iy,iz]+lo_occ, lo_min, lo_max)
                    break
                log_odds[ix,iy,iz] = np.clip(log_odds[ix,iy,iz]+lo_free, lo_min, lo_max)


def query_box_freespace_obb(log_odds, center_e, size, yaw):
    half_w, half_l, half_h = size[0]/2, size[1]/2, size[2]/2
    gox, goy, goz = GRID_ORIGIN
    vs = VOXEL_SIZE
    NX_g, NY_g, NZ_g = log_odds.shape
    cx, cy, cz = center_e
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    diag = float(np.sqrt(half_l**2 + half_w**2))
    ix_lo = max(0,     int((cx - diag   - gox) / vs))
    ix_hi = min(NX_g,  int((cx + diag   - gox) / vs) + 2)
    iy_lo = max(0,     int((cy - diag   - goy) / vs))
    iy_hi = min(NY_g,  int((cy + diag   - goy) / vs) + 2)
    iz_lo = max(0,     int((cz - half_h - goz) / vs))
    iz_hi = min(NZ_g,  int((cz + half_h - goz) / vs) + 2)
    if ix_lo >= ix_hi or iy_lo >= iy_hi or iz_lo >= iz_hi:
        return None
    gx = gox + (np.arange(ix_lo, ix_hi) + 0.5) * vs
    gy = goy + (np.arange(iy_lo, iy_hi) + 0.5) * vs
    gz = goz + (np.arange(iz_lo, iz_hi) + 0.5) * vs
    gx3, gy3, gz3 = np.meshgrid(gx, gy, gz, indexing='ij')
    dx = gx3 - cx; dy = gy3 - cy; dz = gz3 - cz
    lx =  cos_y * dx + sin_y * dy
    ly = -sin_y * dx + cos_y * dy
    inside = (np.abs(lx) <= half_l) & (np.abs(ly) <= half_w) & (np.abs(dz) <= half_h)
    if not inside.any():
        return None
    lo_vals = log_odds[ix_lo:ix_hi, iy_lo:iy_hi, iz_lo:iz_hi][inside]
    n_total = int(inside.sum())
    n_free  = int((lo_vals < THRESHOLD_FREE).sum())
    n_occ   = int((lo_vals > THRESHOLD_OCC).sum())
    return {
        'n_total': n_total, 'n_free': n_free, 'n_occupied': n_occ,
        'frac_free': n_free / n_total, 'frac_occ': n_occ / n_total,
    }


def process_dataset(dataset: str):
    from nuscenes.nuscenes import NuScenes
    from pyquaternion import Quaternion

    if dataset == 'ecp':
        data_root = ECP_DATA_ROOT
        version   = 'v1.0-trainval'
        pred_file = os.path.join(PRED_DIR_ECP, 'autolabel_ecp_annotated_8class.json')
    else:
        data_root = NUSCENES_DATA_ROOT
        version   = 'v1.0-mini'
        pred_file = os.path.join(PRED_DIR_NUSCENES, 'autolabel_nuscenes_mini_train_8class.json')

    print(f'\nLoading nuScenes ({version}) from {data_root}')
    nusc = NuScenes(version=version, dataroot=data_root, verbose=False)

    with open(pred_file) as f:
        pred_data = json.load(f)
    sample_tokens = list(pred_data['results'].keys())
    print(f'Processing {len(sample_tokens)} frames, '
          f'{sum(len(v) for v in pred_data["results"].values())} predictions total')

    # Warm up numba JIT
    if HAS_NUMBA:
        _log = np.zeros((10,10,10), dtype=np.float32)
        _pts = np.array([[1,0,0]], dtype=np.float32)
        _so  = np.array([0,0,0],  dtype=np.float32)
        raytrace(_log, _pts, _so, np.zeros(3,dtype=np.float32),
                 0.2, LO_FREE, LO_OCC, LO_MIN, LO_MAX)
        print('Numba JIT warmed up.')

    log_odds = np.zeros((NX, NY, NZ), dtype=np.float32)
    rows    = []   # predicted box results
    gt_rows = []   # GT box baseline results (Phase 0.5)
    t0 = time.time()

    for frame_i, sample_token in enumerate(sample_tokens):
        sample = nusc.get('sample', sample_token)

        # ── LiDAR data record ────────────────────────────────────────────────
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_sd    = nusc.get('sample_data', lidar_token)
        ep          = nusc.get('ego_pose', lidar_sd['ego_pose_token'])
        cs          = nusc.get('calibrated_sensor', lidar_sd['calibrated_sensor_token'])

        t_e2g = np.array(ep['translation'])
        R_e2g = Quaternion(ep['rotation']).rotation_matrix
        t_l2e = np.array(cs['translation'])
        R_l2e = Quaternion(cs['rotation']).rotation_matrix

        # ── Load LiDAR ───────────────────────────────────────────────────────
        lidar_path = os.path.join(data_root, lidar_sd['filename'])
        raw = np.fromfile(lidar_path, dtype=np.float32)
        stride = 5 if dataset == 'ecp' else 4   # ECP: x,y,z,intensity,ring  nuScenes: x,y,z,intensity
        pts_lidar = raw.reshape(-1, stride)[:, :3]

        # LiDAR → ego
        pts_ego = (R_l2e @ pts_lidar.T).T + t_l2e
        sensor_origin = t_l2e.astype(np.float32)

        # Range filter
        ranges = np.linalg.norm(pts_ego - sensor_origin, axis=1)
        mask   = (ranges >= RAY_MIN_RANGE) & (ranges <= RAY_MAX_RANGE)
        pts_cast = pts_ego[mask].astype(np.float32)

        # ── Build free-space grid ─────────────────────────────────────────────
        log_odds[:] = 0.0
        raytrace(log_odds, pts_cast, sensor_origin, GRID_ORIGIN, VOXEL_SIZE,
                 LO_FREE, LO_OCC, LO_MIN, LO_MAX)

        # ── Predicted boxes: global → ego ─────────────────────────────────────
        pred_boxes_raw = pred_data['results'][sample_token]
        for b in pred_boxes_raw:
            center_g = np.array(b['translation'])
            size     = np.array(b['size'])
            rot_g    = Quaternion(b['rotation'])
            center_e = R_e2g.T @ (center_g - t_e2g)
            R_box_e  = R_e2g.T @ rot_g.rotation_matrix
            yaw      = float(np.arctan2(R_box_e[1, 0], R_box_e[0, 0]))
            dist     = float(np.linalg.norm(center_e[:2]))

            if dist < 15.0:
                regime = 'A'
            elif dist < 30.0:
                regime = 'B'
            else:
                regime = 'C'

            st = query_box_freespace_obb(log_odds, center_e, size, yaw)
            if st is None:
                continue

            rows.append({
                'dataset'     : dataset,
                'sample_token': sample_token,
                'category'    : b['detection_name'],
                'score'       : float(b['detection_score']),
                'range_m'     : round(dist, 2),
                'regime'      : regime,
                'frac_free'   : round(st['frac_free'], 4),
                'frac_occ'    : round(st['frac_occ'],  4),
                'n_free'      : st['n_free'],
                'n_occupied'  : st['n_occupied'],
                'n_total'     : st['n_total'],
                'flagged'     : st['frac_free'] > FREE_THRESHOLD,
            })

        # ── Phase 0.5: GT baseline — same grid, iterate annotations ─────────
        for ann_token in sample['anns']:
            ann = nusc.get('sample_annotation', ann_token)
            center_g = np.array(ann['translation'])
            size     = np.array(ann['size'])          # [w, l, h] nuScenes convention
            rot_g    = Quaternion(ann['rotation'])
            center_e = R_e2g.T @ (center_g - t_e2g)
            R_box_e  = R_e2g.T @ rot_g.rotation_matrix
            yaw      = float(np.arctan2(R_box_e[1, 0], R_box_e[0, 0]))
            dist     = float(np.linalg.norm(center_e[:2]))

            if dist < 15.0:
                regime = 'A'
            elif dist < 30.0:
                regime = 'B'
            else:
                regime = 'C'

            st = query_box_freespace_obb(log_odds, center_e, size, yaw)
            if st is None:
                continue

            gt_rows.append({
                'dataset'     : dataset,
                'sample_token': sample_token,
                'category'    : ann['category_name'],
                'range_m'     : round(dist, 2),
                'regime'      : regime,
                'frac_free'   : round(st['frac_free'], 4),
                'frac_occ'    : round(st['frac_occ'],  4),
                'n_free'      : st['n_free'],
                'n_occupied'  : st['n_occupied'],
                'n_total'     : st['n_total'],
            })

        if (frame_i + 1) % 10 == 0 or frame_i == 0:
            elapsed = time.time() - t0
            rate    = (frame_i + 1) / elapsed
            eta     = (len(sample_tokens) - frame_i - 1) / rate
            print(f'  frame {frame_i+1}/{len(sample_tokens)}  '
                  f'({rate:.1f} fps)  ETA {eta:.0f}s')

    elapsed = time.time() - t0
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.DataFrame(rows)
    out_path = os.path.join(OUT_DIR, f'experiment_a_{dataset}.csv')
    df.to_csv(out_path, index=False)
    print(f'Predictions: {len(df)} records → {out_path}')

    df_gt = pd.DataFrame(gt_rows)
    gt_path = os.path.join(OUT_DIR, f'experiment_a_gt_baseline_{dataset}.csv')
    df_gt.to_csv(gt_path, index=False)
    print(f'GT baseline: {len(df_gt)} records → {gt_path}')
    print(f'Total time: {elapsed:.1f}s')
    return df, df_gt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['ecp', 'nuscenes_mini', 'both'],
                        default='both')
    args = parser.parse_args()

    datasets = ['ecp', 'nuscenes_mini'] if args.dataset == 'both' else [args.dataset]
    for ds in datasets:
        process_dataset(ds)
        print()
