"""
Cheap companion to batch_eval.py: box-level geometry + free-space/below-ground for EVERY GT box and EVERY
detection's OBB, with the fields needed for nuScenes-style center-distance matching (score, center, dims, yaw).

Does NOT render any mesh or sample mesh surface/volume -- those were the slow part of batch_eval.py (the hours-long
run) and don't depend on which GT a detection gets matched to. This only needs a box query against the already-built
free-space grid (sub-millisecond each), so the whole dataset runs in minutes, not hours. Recall / IoU / mesh
free-space / below-ground / not-implied are already correct in batch_eval.py's output regardless of matching
protocol -- scripts/rematch.py joins those back in by (scene/source, cam, frame_idx, sam3_index).

Usage: identical flags to batch_eval.py, writes two CSVs instead of one (--out is used as a stem):
    bash container/run_in_container.sh python -m autolabeling.batch_boxes --dataset nuscenes_mini \\
        --run-root /workspace/autolabeling/output/nuscenes_mini/all_cam_shared --scene scene-0061 \\
        --mode-name moge_affine_local --out /workspace/contribution_ideas/phase0_mesh_freespace/results/boxes/scene-0061
Writes <out>_det.csv (one row per detection) and <out>_gt.csv (one row per GT box, shared across all cameras of
that keyframe).
"""
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .batch_eval import (
    DATASET_INFO, DEV_ROOT, box_of, find_ckpt_dir, frames_per_cam_ecp, frames_per_cam_nuscenes,
    gt_boxes_ego, gt_with_speed, load_body_ckpt, load_obj_ckpt, load_pl_states, pl_ground_z,
    rider_merged_obb, single_sweep,
)
from .pipeline import _restore_pseudolabeler
from .utils import freespace as fs


def box_stats(grid: fs.FreeSpaceGrid, center, l: float, w: float, h: float, yaw: float) -> Dict[str, float]:
    s = fs.freespace_stats(grid, fs.box_lattice(center, l, w, h, yaw, grid.vs), margins=(), volume=True, claim=0.0)
    return dict(free_frac=s['frac_free'], free_m3=s['frac_free'] * s['vol_m3'])


def run(dataset: str, run_root: Path, mode_name: str, out_stem: Path, scene: Optional[str] = None,
       device: str = 'cuda') -> None:
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
    gt_categories = info['gt_categories']

    det_rows, gt_rows = [], []
    for i in range(n):
        any_frame = frames_per_cam[info['cameras'][0]][i]
        pts_sweep, sensor = single_sweep(nusc, any_frame)
        grid = fs.build_grid(pts_sweep, sensor, vs=0.1)

        if dataset == 'nuscenes_mini':
            gts = gt_with_speed(nusc, any_frame)
        else:
            gts = gt_boxes_ego(nusc, any_frame.sample_token, any_frame.R_e2g, any_frame.t_e2g)
            for g in gts:
                g['speed'] = float('nan')

        canon_pl = _restore_pseudolabeler(pl_states_by_cam[info['cameras'][0]].get(i), device, DEV_ROOT)
        for gi, g in enumerate(gts):
            if not g['category'].startswith(gt_categories):
                continue
            w, l, h = g['size']
            st = box_stats(grid, g['center'], l, w, h, g['yaw'])
            gz = pl_ground_z(canon_pl, np.asarray(g['center'][:2], np.float64), device)
            gt_rows.append(dict(scene=getattr(any_frame, 'scene_name', None), frame_idx=getattr(any_frame, 'frame_idx', i), gt_index=gi,
                               category=g['category'], center_x=g['center'][0], center_y=g['center'][1],
                               center_z=g['center'][2], length=l, width=w, height=h, yaw=g['yaw'],
                               speed=g['speed'], below_ground=gz - (g['center'][2] - h / 2), **st))

        for cam in info['cameras']:
            frame = frames_per_cam[cam][i]
            objs = load_obj_ckpt(obj_dir_by_cam[cam], i)
            if not objs:
                continue
            bodies = load_body_ckpt(ckpt_root / cam, i)
            pl_model = _restore_pseudolabeler(pl_states_by_cam[cam].get(i), device, DEV_ROOT)
            for o in objs:
                raw = o['obb_raw']
                oc, ol, ow, oh, oy = box_of(raw['center'], raw['dims'], raw['yaw'])
                gz = pl_ground_z(pl_model, oc[:2], device)
                row = dict(scene=getattr(frame, 'scene_name', None), cam=cam, frame_idx=getattr(frame, 'frame_idx', i),
                          sam3_index=o.get('sam3_index'), cls=o['prompt'], score=float(o['score']),
                          center_x=oc[0], center_y=oc[1], center_z=oc[2], length=ol, width=ow, height=oh, yaw=oy,
                          below_ground=gz - (oc[2] - oh / 2))
                row.update(box_stats(grid, oc, ol, ow, oh, oy))
                if o['prompt'] in ('bicycle', 'motorcycle'):
                    merged = rider_merged_obb(o['vertices'], oc, bodies, frame)
                    if merged is not None:
                        mc, md, my = merged
                        row['pipeline_center_x'], row['pipeline_center_y'], row['pipeline_center_z'] = mc
                        row['pipeline_length'], row['pipeline_width'], row['pipeline_height'] = md
                        row['pipeline_yaw'] = my
                        row.update({f'pipeline_{k}': v for k, v in box_stats(grid, mc, md[0], md[1], md[2], my).items()})
                        row['pipeline_below_ground'] = gz - (mc[2] - md[2] / 2)
                if 'pipeline_center_x' not in row:
                    for k in ('center_x', 'center_y', 'center_z', 'length', 'width', 'height', 'yaw',
                             'free_frac', 'free_m3', 'below_ground'):
                        row[f'pipeline_{k}'] = row[k]
                det_rows.append(row)
        print(f'[{dataset}{("/" + scene) if scene else ""}] keyframe {i + 1}/{n} '
             f'(frame_idx={getattr(any_frame, "frame_idx", i)}): running total {len(det_rows)} detection(s), '
             f'{len(gt_rows)} GT box(es)', flush=True)

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(det_rows).to_csv(out_stem.with_name(out_stem.name + '_det.csv'), index=False)
    pd.DataFrame(gt_rows).to_csv(out_stem.with_name(out_stem.name + '_gt.csv'), index=False)
    print(f'Wrote {len(det_rows)} detection(s), {len(gt_rows)} GT box(es) -> {out_stem}_{{det,gt}}.csv')


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', required=True, choices=list(DATASET_INFO))
    p.add_argument('--run-root', required=True, type=Path)
    p.add_argument('--mode-name', default='moge_affine_local')
    p.add_argument('--scene', default=None)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--device', default='cuda')
    return p.parse_args()


if __name__ == '__main__':
    a = _parse_args()
    run(a.dataset, a.run_root, a.mode_name, a.out, scene=a.scene, device=a.device)
