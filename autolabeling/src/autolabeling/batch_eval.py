"""
Full-dataset mesh / free-space / ground evaluation (contribution_ideas/phase0_mesh_freespace/PLAN.md).

Reads the FULL-mesh checkpoints of an already-finished `run_pipeline.py` run (mesh_points: 0, all cameras) and
computes the five metrics agreed in PLAN.md, one row per object. Does not run any model -- pure post-hoc measurement.

Checkpoint index -> real frame: checkpoints are indexed by POSITION in the frame list the pipeline run used, not by
sample token, so this module reconstructs the identical frame list (`collect_frames_multi_cam` /
`collect_annotated_frames_multi_cam` with the same arguments run_pipeline.py used: whole scene, frame_start=0,
frame_end=None for nuScenes; all annotated keyframes for ECP) and zips it against the checkpoint files by index.

One free-space grid is built once per KEYFRAME (all cameras of a keyframe share the same LiDAR sweep), not once per
camera -- reused across every camera's objects for that keyframe. PseudoLabeler states are one file PER CAMERA
holding every frame's state dict (`pl_states__<hash>.pt`, keyed by frame index) -- loaded once per camera, not once
per frame.

Usage (one dataset, optionally one scene, at a time -- matches the existing per-scene array-job convention):
    bash container/run_in_container.sh python -m autolabeling.batch_eval \\
        --dataset nuscenes_mini --run-root /workspace/autolabeling/output/nuscenes_mini/all_cam_shared \\
        --scene scene-0061 --mode-name moge_affine_local \\
        --out /workspace/contribution_ideas/phase0_mesh_freespace/results/nuscenes_mini_scene-0061.csv
    bash container/run_in_container.sh python -m autolabeling.batch_eval \\
        --dataset ecp --run-root /workspace/autolabeling/output/ecp/all_cam_shared --mode-name moge_affine_local \\
        --out /workspace/contribution_ideas/phase0_mesh_freespace/results/ecp.csv
"""
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .fitting.obb import compute_obb_gravity_aligned, compute_obb_pedestrian
from .orientation.pedestrian import facing_direction
from .pilot import box_of, gt_box_of, gt_with_speed, lidar_in_camera, occluder_mask, single_sweep
from .pipeline import _body_from_ckpt, _load, _obj_from_ckpt, _restore_pseudolabeler
from .utils import controls as ct
from .utils import freespace as fs
from .utils import mesh_mask as mm
from .utils.diagnostics import associate_gt_lidar, gt_boxes_ego, resolve_duplicate_gt
from .utils.geometry import cam_to_ego

DEV_ROOT = Path('/workspace')
RIDER_CLASSES = ('bicycle', 'motorcycle')
RIDER_DIST_THRESH = 1.0          # matches pipeline.py's _merge_rider_obbs default exactly
CLASSIFY_MARGIN = 0.2            # internal only (not a reported metric): standoff used to decide whether an
                                  # in-mask return "implies" a violating point, so grazing noise doesn't dominate #5
GRID_VS = 0.1

DATASET_INFO = {
    'nuscenes_mini': dict(gt_categories=('vehicle',),
                          cameras=['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
                                   'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']),
    'ecp':           dict(gt_categories=('car', 'trailer', 'motorcycle', 'bicycle'),
                          cameras=['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT']),
}


# ── Frame reconstruction (must match run_pipeline.py's collection exactly) ─────────────────

def frames_per_cam_nuscenes(nusc, scene_name: str, cameras: List[str]) -> Dict[str, list]:
    from .data.nuscenes_loader import collect_frames_multi_cam
    return collect_frames_multi_cam(nusc, [scene_name], cameras, 0, None)


def frames_per_cam_ecp(nusc, cameras: List[str]) -> Dict[str, list]:
    from .data.ecp_loader import collect_annotated_frames_multi_cam
    return collect_annotated_frames_multi_cam(nusc, cameras)


def find_ckpt_dir(cam_dir: Path, prefix: str) -> Path:
    hits = sorted(cam_dir.glob(f'{prefix}__*'))
    if not hits:
        raise FileNotFoundError(f'no {prefix}__* checkpoint under {cam_dir}')
    if len(hits) > 1:
        raise FileNotFoundError(f'ambiguous {prefix}__* checkpoints under {cam_dir}: {[h.name for h in hits]}')
    return hits[0]


def load_pl_states(cam_dir: Path) -> Dict[int, Optional[dict]]:
    """{frame_index: state_dict or None}, one file per camera holding every frame -- loaded once, not per frame."""
    import torch
    hits = sorted(cam_dir.glob('pl_states__*.pt'))
    if not hits:
        return {}
    return torch.load(hits[0], map_location='cpu')


def load_obj_ckpt(objects_dir: Path, i: int) -> list:
    p = objects_dir / f'{i:06d}.pkl.gz'
    return _obj_from_ckpt(_load(p)) if p.exists() else []


def load_body_ckpt(cam_dir: Path, i: int) -> list:
    hits = sorted(cam_dir.glob('body__*'))
    if not hits:
        return []
    p = hits[0] / f'{i:06d}.pkl.gz'
    data = _load(p) if p.exists() else None
    return _body_from_ckpt(data) if data is not None else []


# ── Rider merge (reproduces pipeline._merge_rider_obbs for one object, read-only) ──────────

def rider_merged_obb(mesh_verts_cam: np.ndarray, obb_center_ego: np.ndarray, bodies: List[dict], frame):
    """(center, dims, yaw) [ego] of the object merged with its closest body within RIDER_DIST_THRESH, or None."""
    best, best_d = None, RIDER_DIST_THRESH
    for br in bodies:
        # body checkpoints don't persist obb_center (only set later by _postprocess_frame) -- recompute it the
        # same way: PCA-free, facing-direction-aligned box from the joints.
        _, body_center_cam, _ = compute_obb_pedestrian(br['vertices'], facing_direction(br['joints_3d']))
        bc_ego = frame.R_c2e @ np.asarray(body_center_cam, np.float64) + frame.t_c2e
        d = float(np.linalg.norm(obb_center_ego - bc_ego))
        if d <= best_d:
            best_d, best = d, br
    if best is None:
        return None
    combined = np.concatenate([np.asarray(best['vertices'], np.float32), mesh_verts_cam.astype(np.float32)], axis=0)
    _, center, dims, yaw = compute_obb_gravity_aligned(combined, frame.R_c2e, frame.t_c2e, ground_z=None)
    return center, dims, yaw


# ── Per-object evaluation: the five PLAN.md metrics ────────────────────────────────────────

def pl_ground_z(pl_model, xy: np.ndarray, device: str) -> float:
    if pl_model is None:
        return float('nan')
    import torch
    with torch.no_grad():
        xy_t = torch.tensor(xy, dtype=torch.float32, device=device).unsqueeze(0)
        return float(pl_model(xy_t).item())


def evaluate_object(obj: dict, frame, grid: fs.FreeSpaceGrid, pts_sweep: np.ndarray, H: int, W: int,
                    occluders: np.ndarray, gt: Optional[dict], pl_model, device: str,
                    n_surface: int = 20000) -> dict:
    verts, faces, mask = obj['vertices'], obj['faces'], obj['binary_mask']
    row: Dict[str, float] = dict(cls=obj['prompt'], has_gt=gt is not None)

    # 1. mask fit
    sil, _ = mm.render_mesh(verts, faces, frame.K, H, W, device=device)
    ma = mm.mask_agreement(sil, mask, occluders)
    row['recall'] = ma['recall']
    clean = (not ma['touches_border']) and not bool((sil & occluders).any())
    row['clean'] = clean
    row['iou'] = ma['iou'] if clean else float('nan')

    verts_ego = cam_to_ego(np.asarray(verts, np.float64), frame.R_c2e, frame.t_c2e)
    # obb_raw: fitted on the full mesh right after inference (attach_full_obbs), yaw ambiguous by 180 deg -- fine
    # here, every metric below (free space, below-ground, size) is invariant to a 180 deg heading flip.
    raw = obj['obb_raw']
    oc, ol, ow, oh, oy = box_of(raw['center'], raw['dims'], raw['yaw'])

    # 2. free-space touch (mesh surface, mesh volume, OBB volume, GT-box volume) -- NO distance margin:
    #    plain free/occupied/unknown classification, checked against real data to barely depend on a margin.
    surf = fs.sample_mesh_surface(verts_ego, faces, n_surface)
    s_surf = fs.freespace_stats(grid, surf, margins=(), claim=0.0)
    row['surf_free'], row['surf_unknown'] = s_surf['frac_free'], s_surf['frac_unknown']

    vol = fs.mesh_volume_points(verts_ego, faces, grid.vs)
    s_vol = fs.freespace_stats(grid, vol, margins=(), volume=True, claim=0.0)
    row['mesh_free_frac'], row['mesh_free_m3'] = s_vol['frac_free'], s_vol['frac_free'] * s_vol['vol_m3']

    s_obb = fs.freespace_stats(grid, fs.box_lattice(oc, ol, ow, oh, oy, grid.vs), margins=(), volume=True, claim=0.0)
    row['obb_free_frac'], row['obb_free_m3'] = s_obb['frac_free'], s_obb['frac_free'] * s_obb['vol_m3']
    row['pipeline_obb_free_frac'], row['pipeline_obb_free_m3'] = row['obb_free_frac'], row['obb_free_m3']
    row['pipeline_obb_below_ground'] = None   # filled in below (ground) or by the rider-merge branch in the caller

    if gt is not None:
        gc, gl, gw, gh, gy = gt_box_of(gt)
        s_gt = fs.freespace_stats(grid, fs.box_lattice(gc, gl, gw, gh, gy, grid.vs), margins=(), volume=True, claim=0.0)
        row['gt_free_frac'], row['gt_free_m3'] = s_gt['frac_free'], s_gt['frac_free'] * s_gt['vol_m3']

        # 4. floor control: mesh re-centred to GT position + yaw, OWN shape/size kept
        v_floor = ct.align_mesh_to_box(verts_ego, oc, oy, (ol, ow, oh), gc, gy, dst_dims=None)
        s_floor = fs.freespace_stats(grid, fs.sample_mesh_surface(v_floor, faces, n_surface, seed=1), margins=(), claim=0.0)
        row['floor_free'] = s_floor['frac_free']

    # 3. below-ground reach (PseudoLabeler ground surface), mesh / OBB / GT
    gz = pl_ground_z(pl_model, oc[:2], device)
    row['ground_z'] = gz
    row['mesh_below_ground'] = gz - float(np.percentile(verts_ego[:, 2], 0.5))
    row['obb_below_ground'] = gz - (oc[2] - oh / 2)
    row['pipeline_obb_below_ground'] = row['obb_below_ground']
    if gt is not None:
        row['gt_below_ground'] = gz - (gt['center'][2] - gt['size'][2] / 2)

    # 5. "not implied by in-mask depth" -- the direct answer to "method 1 already covers this"
    P, ends = ct.violating_points(grid, surf, claim=CLASSIFY_MARGIN)
    row['n_violating'] = len(P)
    if len(P):
        u, v, z = lidar_in_camera(frame, pts_sweep, H, W)
        ui, vi = np.clip(np.round(u).astype(int), 0, W - 1), np.clip(np.round(v).astype(int), 0, H - 1)
        inm = mask[vi, ui]
        ret_uvz = np.stack([u[inm], v[inm], z[inm]], axis=1)
        cls = ct.classify_violations(P, ends, frame, mask, ret_uvz, vol_pts=None, margin=CLASSIFY_MARGIN)
        row['not_implied'] = 1.0 - cls['red_implied']
    else:
        row['not_implied'] = float('nan')

    if gt is not None:
        row['gt_speed'] = gt.get('speed', float('nan'))
        row['gt_range'] = float(np.hypot(*gt['center'][:2]))
    return row, verts, oc


# ── One keyframe, all cameras ───────────────────────────────────────────────────────────────

def evaluate_keyframe(dataset: str, nusc, frame_by_cam: Dict[str, object], obj_by_cam: Dict[str, list],
                      body_by_cam: Dict[str, list], pl_model_by_cam: Dict[str, object], device: str) -> List[dict]:
    """All arguments are {camera: value} for ONE keyframe, already loaded by the caller."""
    any_frame = next(iter(frame_by_cam.values()))
    pts_sweep, sensor = single_sweep(nusc, any_frame)
    grid = fs.build_grid(pts_sweep, sensor, vs=GRID_VS)
    gt_categories = DATASET_INFO[dataset]['gt_categories']

    if dataset == 'nuscenes_mini':
        gts = gt_with_speed(nusc, any_frame)
    else:
        gts = gt_boxes_ego(nusc, any_frame.sample_token, any_frame.R_e2g, any_frame.t_e2g)
        for g in gts:
            g['speed'] = float('nan')

    rows = []
    for cam, frame in frame_by_cam.items():
        objs = obj_by_cam.get(cam, [])
        if not objs:
            continue
        H, W = frame.img_height, frame.img_width

        claims, matches = {}, [None] * len(objs)
        for i, o in enumerate(objs):
            gi, frac, amb = associate_gt_lidar(gts, o['binary_mask'], frame, pts_sweep, category_prefix=gt_categories)
            matches[i] = None if (gi is None or amb) else gi
            if matches[i] is not None:
                claims[i] = (gi, int(o['binary_mask'].sum()))
        for i in resolve_duplicate_gt(claims):
            matches[i] = None

        pl_model = pl_model_by_cam.get(cam)
        for i, o in enumerate(objs):
            occ = occluder_mask([oo['binary_mask'] for j, oo in enumerate(objs) if j != i], (H, W))
            gt = None if matches[i] is None else gts[matches[i]]
            row, verts, oc_ego = evaluate_object(o, frame, grid, pts_sweep, H, W, occ, gt, pl_model, device)
            row.update(cam=cam, prompt=o['prompt'], sam3_index=o.get('sam3_index'),
                       scene=getattr(frame, 'scene_name', None), frame_idx=getattr(frame, 'frame_idx', None))

            if o['prompt'] in RIDER_CLASSES:
                merged = rider_merged_obb(verts, oc_ego, body_by_cam.get(cam, []), frame)
                if merged is not None:
                    mc, md, my = merged
                    s = fs.freespace_stats(grid, fs.box_lattice(mc, md[0], md[1], md[2], my, grid.vs),
                                           margins=(), volume=True, claim=0.0)
                    row['pipeline_obb_free_frac'] = s['frac_free']
                    row['pipeline_obb_free_m3'] = s['frac_free'] * s['vol_m3']
                    row['pipeline_obb_below_ground'] = row['ground_z'] - (mc[2] - md[2] / 2)
            rows.append(row)
    return rows


# ── Driver ───────────────────────────────────────────────────────────────────────────────────

def run(dataset: str, run_root: Path, mode_name: str, out_csv: Path, scene: Optional[str] = None,
       device: str = 'cuda') -> int:
    """
    Writes `out_csv` INCREMENTALLY, one keyframe's rows at a time (a busy scene can take longer than one job's time
    limit; a timed-out job would otherwise lose everything). If `out_csv` already has rows from a previous, timed-out
    attempt at the SAME scene, resumes after the highest `frame_idx` already written instead of redoing it -- just
    resubmit the same command after a timeout.
    """
    import csv as _csv
    import pandas as pd
    from nuscenes.nuscenes import NuScenes
    info = DATASET_INFO[dataset]
    version, data_root = (('v1.0-mini', DEV_ROOT / 'data' / 'nuScenes_mini') if dataset == 'nuscenes_mini'
                          else ('v1.0-trainval', DEV_ROOT / 'data' / 'ecp'))
    nusc = NuScenes(version=version, dataroot=str(data_root), verbose=False)

    if dataset == 'nuscenes_mini':
        assert scene is not None, 'nuscenes_mini needs --scene (one array task per scene)'
        ckpt_root = run_root / 'scenes' / scene / 'checkpoints'
        frames_per_cam = frames_per_cam_nuscenes(nusc, scene, info['cameras'])
    else:
        ckpt_root = run_root / 'checkpoints'
        frames_per_cam = frames_per_cam_ecp(nusc, info['cameras'])

    n = len(next(iter(frames_per_cam.values())))
    obj_dir_by_cam = {c: find_ckpt_dir(ckpt_root / c, f'objects__{mode_name}') for c in info['cameras']}
    pl_states_by_cam = {c: load_pl_states(ckpt_root / c) for c in info['cameras']}

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    start_i, n_done, writer, fieldnames, fh = 0, 0, None, None, None
    if out_csv.exists() and out_csv.stat().st_size > 0:
        prev = pd.read_csv(out_csv)
        if len(prev) and 'frame_idx' in prev:
            start_i = int(prev['frame_idx'].max()) + 1
            n_done = len(prev)
            fieldnames = list(prev.columns)
            print(f'Resuming {out_csv}: {n_done} row(s) already written, continuing from keyframe {start_i}/{n}.',
                 flush=True)

    mode_ = 'a' if start_i > 0 else 'w'
    fh = open(out_csv, mode_, newline='')
    n_written = n_done
    try:
        for i in range(start_i, n):
            frame_by_cam = {c: frames_per_cam[c][i] for c in info['cameras']}
            obj_by_cam = {c: load_obj_ckpt(obj_dir_by_cam[c], i) for c in info['cameras']}
            body_by_cam = {c: load_body_ckpt(ckpt_root / c, i) for c in info['cameras']}
            pl_model_by_cam = {c: _restore_pseudolabeler(pl_states_by_cam[c].get(i), device, DEV_ROOT)
                              for c in info['cameras']}
            rows = evaluate_keyframe(dataset, nusc, frame_by_cam, obj_by_cam, body_by_cam, pl_model_by_cam, device)
            if rows:
                if writer is None:
                    fieldnames = fieldnames or sorted({k for r in rows for k in r})
                    writer = _csv.DictWriter(fh, fieldnames=fieldnames, extrasaction='ignore')
                    if mode_ == 'w':
                        writer.writeheader()
                writer.writerows(rows)
                fh.flush()
                n_written += len(rows)
            print(f'[{dataset}{("/" + scene) if scene else ""}] keyframe {i + 1}/{n}: {len(rows)} object(s)', flush=True)
    finally:
        fh.close()
    print(f'Wrote {n_written} object row(s) total -> {out_csv}')
    return n_written


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', required=True, choices=list(DATASET_INFO))
    p.add_argument('--run-root', required=True, type=Path,
                   help='The prepare-shared run directory, e.g. .../all_cam_shared (NOT .../all_cam_mode_2 -- '
                        'that one only holds the submission JSON, the checkpoints live in the shared run).')
    p.add_argument('--mode-name', default='moge_affine_local', help='pointmap mode NAME (not the number).')
    p.add_argument('--scene', default=None, help='nuscenes_mini: one scene name (one array task per scene).')
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--device', default='cuda')
    return p.parse_args()


if __name__ == '__main__':
    a = _parse_args()
    run(a.dataset, a.run_root, a.mode_name, a.out, scene=a.scene, device=a.device)
