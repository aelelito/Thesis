#!/usr/bin/env python3
"""
Compare pointmap modes object by object against GT, from the checkpoints of several pipeline runs.

    python compare_modes.py --dataset nuscenes_mini --runs sweep1_m1 sweep1_m2 sweep1_m7 [--labels m1 m2 m7] [--baseline m2]
    python compare_modes.py --dataset nuscenes_mini --shared-run cmp1_shared --modes 1 2 6 7 8 --baseline m2
      (the second form is for sweeps made with submit_sweep.sh: all modes share one checkpoint folder and each keeps its
       SAM3D Objects results in objects__<mode name>__<hash>)

Each run folder is autolabeling/output/<dataset>/<run>/ (nuScenes: scenes/<scene>/checkpoints, ECP: checkpoints).
All runs must have processed the same frames (same --frame-start/--frame-end). What is done:
  * every detection is matched to a GT box ONCE, from LiDAR (the GT box containing most of the LiDAR points on its mask,
    >= 30% of them, one GT box per detection, ambiguous ones dropped) -- independent of the mode, so every mode is scored
    against the same box;
  * errors: center / range / lateral [m], length-width-height ratios, yaw [deg] (utils/diagnostics.py);
  * only objects matched in EVERY run are compared; medians per run, then paired differences to the baseline run with a
    bootstrap 95% CI, win rate and exact sign test; strata by range, truncation (mask touching the image edge) and
    (modes 6/8) ill-conditioned object affines.
Writes the per-object table to <output-root>/_compare_<labels>.csv.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

DATASETS = {
    'nuscenes_mini': dict(version='v1.0-mini', root='/workspace/data/nuScenes_mini', gt=('vehicle',)),
    'ecp': dict(version='v1.0-trainval', root='/workspace/data/ecp', gt=('car', 'trailer', 'motorcycle', 'bicycle')),
}
EGO = dict(use_ego_body_filter=True, ego_box_half_x=4., ego_box_half_y=1.5, ego_box_z_min=.5, ego_box_z_max=2.5)


def _stage(ckpt_root: Path, prefix: str):
    dirs = sorted(ckpt_root.glob(prefix + '*'), key=lambda p: p.stat().st_mtime)
    if not dirs:
        return None
    if len(dirs) > 1:
        print(f'  [note] {ckpt_root}: {len(dirs)} {prefix}* folders, using the newest ({dirs[-1].name})')
    return dirs[-1]


def _units(run_root: Path, dataset: str) -> dict:
    """unit name -> checkpoints folder (nuScenes: one per scene; ECP: a single 'annotated' unit)."""
    if (run_root / 'scenes').is_dir():
        return {d.name: d / 'checkpoints' for d in sorted((run_root / 'scenes').iterdir()) if (d / 'checkpoints').is_dir()}
    if (run_root / 'checkpoints').is_dir():
        return {'annotated': run_root / 'checkpoints'}
    raise SystemExit(f'{run_root}: no scenes/ or checkpoints/ folder')


def _idx(stage: Path) -> set:
    return {int(p.name.split('.')[0]) for p in stage.glob('*.pkl.gz')}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', choices=list(DATASETS), default='nuscenes_mini')
    ap.add_argument('--runs', nargs='+', default=None, help='run folder names under <output-root> (one per mode)')
    ap.add_argument('--shared-run', default=None, help='ONE run folder whose checkpoints hold several modes (submit_sweep.sh)')
    ap.add_argument('--modes', nargs='+', default=None, help='with --shared-run: the pointmap modes to compare (numbers or names)')
    ap.add_argument('--stages', nargs='+', default=None,
                    help='with --shared-run, in place of --modes: EXACT objects__<mode>__<hash> checkpoint folder '
                         'names, one per label. Use this instead of --modes whenever two or more runs share the '
                         'same pointmap mode (so --modes\' objects__<mode>__* globbing cannot tell them apart) -- '
                         'e.g. the token-drop probe, several force_drop_modalities variants of mode 2. Read the '
                         'exact name off each run\'s own log line "checkpoint stage: objects__...".')
    ap.add_argument('--labels', nargs='+', default=None, help='short names for the runs (default: the folder names)')
    ap.add_argument('--baseline', default=None, help='label of the run the others are compared to (default: the first)')
    ap.add_argument('--output-root', type=Path, default=None)
    ap.add_argument('--scenes', nargs='*', default=None, help='restrict to these scenes')
    ap.add_argument('--frame-start', type=int, default=0, help='the --frame-start the runs used (checkpoint i = keyframe start+i)')
    ap.add_argument('--max-frames', type=int, default=None, help='only the first N frames per scene (quick check)')
    ap.add_argument('--csv', type=Path, default=None)
    args = ap.parse_args()

    if bool(args.runs) == bool(args.shared_run):
        raise SystemExit('give either --runs or --shared-run (with --modes/--stages)')
    exact_stages = None
    if args.shared_run:
        assert bool(args.modes) != bool(args.stages), '--shared-run needs exactly one of --modes or --stages'
        if args.stages:
            exact_stages = args.stages
            mode_names = None
            labels = args.labels or [f's{i}' for i in range(len(exact_stages))]
            args.runs = [args.shared_run] * len(exact_stages)
            assert len(labels) == len(exact_stages), '--labels needs one name per --stages entry'
        else:
            from src.autolabeling.models.sam3d_objects import resolve_pointmap_mode
            mode_names = [resolve_pointmap_mode(int(m) if str(m).isdigit() else m) for m in args.modes]
            labels = args.labels or [f'm{m}' for m in args.modes]
            args.runs = [args.shared_run] * len(mode_names)
            assert len(labels) == len(mode_names), '--labels needs one name per mode'
    else:
        mode_names = None
        labels = args.labels or args.runs
        assert len(labels) == len(args.runs), '--labels needs one name per run'
    base = args.baseline or labels[0]
    assert base in labels, f'--baseline must be one of {labels}'
    cfg = DATASETS[args.dataset]
    out_root = args.output_root or Path('/workspace/autolabeling/output') / args.dataset

    from nuscenes.nuscenes import NuScenes
    from src.autolabeling.pipeline import _load, _obj_from_ckpt, _postprocess_frame, _sam3_from_ckpt
    from src.autolabeling.utils.compare import add_metrics, common_objects, medians, paired_table
    from src.autolabeling.utils.diagnostics import (associate_gt_lidar, gt_boxes_ego, object_errors,
                                                    resolve_duplicate_gt)
    from src.autolabeling.utils.lidar import load_lidar_pts_aggregated

    nusc = NuScenes(version=cfg['version'], dataroot=cfg['root'], verbose=False)
    units_per_run = [_units(out_root / r, args.dataset) for r in args.runs]
    units = [u for u in units_per_run[0] if all(u in up for up in units_per_run) and (not args.scenes or u in args.scenes)]
    if not units:
        raise SystemExit('the runs share no scene/unit')

    rows, stats = [], dict(det=0, matched=0, ambiguous=0, nomatch=0, duplicate=0, frames=0)
    for unit in units:
        if args.dataset == 'ecp':
            from src.autolabeling.data.ecp_loader import collect_annotated_frames
            frames = collect_annotated_frames(nusc, 'CAM_FRONT')
        else:
            from src.autolabeling.data.nuscenes_loader import collect_frames
            frames = collect_frames(nusc, [unit], 'CAM_FRONT', args.frame_start, None)
        if exact_stages:                                 # one shared checkpoint folder, exact objects__<mode>__<hash> names
            sam_stage = _stage(units_per_run[0][unit], 'sam3__')
            obj_dirs = []
            for name in exact_stages:
                p = units_per_run[0][unit] / name
                if not p.is_dir():
                    raise SystemExit(f'--stages: {p} does not exist (check the exact name against that '
                                      f'run\'s own log line "checkpoint stage: ...")')
                obj_dirs.append(p)
            stages = [(sam_stage, p) for p in obj_dirs]
        elif mode_names:                                 # one shared checkpoint folder, one objects__<mode>__ folder per mode
            sam_stage = _stage(units_per_run[0][unit], 'sam3__')
            stages = [(sam_stage, _stage(units_per_run[0][unit], f'objects__{n}__')) for n in mode_names]
        else:
            stages = [(_stage(up[unit], 'sam3__'), _stage(up[unit], 'objects__')) for up in units_per_run]
        if any(o is None for _, o in stages) or stages[0][0] is None:
            print(f'skipping {unit}: missing checkpoints in one of the runs')
            continue
        idx = sorted(set.intersection(*[_idx(o) for _, o in stages]) & _idx(stages[0][0]))
        if args.max_frames:
            idx = idx[:args.max_frames]
        for i in idx:
            frame = frames[i]
            sam = _sam3_from_ckpt(_load(stages[0][0] / f'{i:06d}.pkl.gz'))
            per_run = []
            for (_, ostage) in stages:
                objs = _obj_from_ckpt(_load(ostage / f'{i:06d}.pkl.gz'))
                for r in objs:                                            # identify the detection across runs
                    if r.get('sam3_index') is None:
                        r['sam3_index'] = next((j for j, d in enumerate(sam.get(r['prompt'], []))
                                                if np.array_equal(d['binary_mask'], r['binary_mask'])), None)
                _postprocess_frame(frame, [], objs)                       # OBB (from obb_raw when stored)
                per_run.append({(r['prompt'], r['sam3_index']): r for r in objs if r['sam3_index'] is not None})
            keys = sorted(set.intersection(*[set(d) for d in per_run]))
            gts = gt_boxes_ego(nusc, frame.sample_token, frame.R_e2g, frame.t_e2g)
            pts = load_lidar_pts_aggregated(nusc, frame, 0, 0, **EGO)     # ONE sweep
            H, W = frame.img_height, frame.img_width
            assoc = {}
            for k in keys:
                m = sam[k[0]][k[1]]['binary_mask']
                gi, frac, amb = associate_gt_lidar(gts, m, frame, pts, category_prefix=cfg['gt'])
                assoc[k] = (gi, amb, int(m.sum()))
            lose = resolve_duplicate_gt({k: (v[0], v[2]) for k, v in assoc.items() if v[0] is not None and not v[1]})
            stats['frames'] += 1
            for k in keys:
                gi, amb, area = assoc[k]
                stats['det'] += 1
                if gi is None:
                    stats['nomatch'] += 1
                    continue
                if amb:
                    stats['ambiguous'] += 1
                    continue
                if k in lose:
                    stats['duplicate'] += 1
                    continue
                stats['matched'] += 1
                m = sam[k[0]][k[1]]['binary_mask']
                trunc = bool(m[:2].any() or m[-2:].any() or m[:, :2].any() or m[:, -2:].any())
                ill = np.nan
                for d in per_run:
                    r = d[k]
                    if r.get('fit_ab') is not None and r.get('bg_ab') is not None:
                        a_o, a_b = r['fit_ab'][0], r['bg_ab'][0]
                        ill = float(a_o <= 0 or a_o < a_b / 2 or a_o > a_b * 2)
                        break
                for lab, d in zip(labels, per_run):
                    r = d[k]
                    e = object_errors(r['obb_center'], r['obb_dims'], r['obb_yaw'], gts[gi])
                    rows.append(dict(obj=f'{unit}:{i}:{k[0]}:{k[1]}', unit=unit, frame=i, cls=k[0], run=lab, matched=True,
                                     truncated=trunc, ill_conditioned=ill, mask_px=area, label=r.get('o3_mode'),
                                     ssi_scale=r.get('ssi_scale'), **e))
    if not rows:
        raise SystemExit('no matched objects in common -- nothing to compare')
    df = add_metrics(pd.DataFrame(rows))
    both = common_objects(df, labels)
    both = both.assign(range_bin=pd.cut(both['gt_range'], [0, 15, 30, 1000], labels=['<15 m', '15-30 m', '>30 m']),
                       size_bin=pd.cut(both['mask_px'], [0, 5000, 20000, 1e9], labels=['small (<5k px)', 'medium', 'large (>20k px)']))
    pd.set_option('display.width', 220)
    pd.set_option('display.max_columns', 40)
    pd.set_option('display.float_format', lambda v: '%.3f' % v)

    print(f'\nruns: {dict(zip(labels, exact_stages or mode_names or args.runs))}   baseline: {base}')
    print(f'frames {stats["frames"]} | detections {stats["det"]}: matched {stats["matched"]}, ambiguous {stats["ambiguous"]}, '
          f'no GT match {stats["nomatch"]}, duplicate claims dropped {stats["duplicate"]}')
    n_obj = both['obj'].nunique()
    print(f'objects matched in EVERY run: {n_obj}   (truncated by the image edge: {both[both.run == base].truncated.sum()})')
    print('\n== medians per run (lower is better; ratios: closer to 1 is better) ==')
    print(medians(both, labels).to_string())
    nt = both[~both.truncated]
    print(f'\n== same, without objects cut by the image edge (n = {nt["obj"].nunique()}) ==')
    print(medians(nt, labels).to_string())
    print(f'\n== paired differences to the baseline "{base}" (run - baseline; NEGATIVE = the run is better); 95% CI by bootstrap ==')
    pt = paired_table(nt, labels, base)
    print(pt.drop(columns=['vs']).to_string(index=False))
    print('\n== strata (median center_err m / size_err, n objects) ==')
    for name, col in (('range', 'range_bin'), ('mask size (small mask = the crop is mostly background)', 'size_bin'),
                      ('cut by image edge', 'truncated'), ('ill-conditioned object affine', 'ill_conditioned')):
        g = both.dropna(subset=[col]) if col == 'ill_conditioned' else both
        if g.empty:
            print(f'  {name}: not available for these runs')
            continue
        t = g.groupby([col, 'run'], observed=True)[['center_err', 'size_err']].median().unstack('run').reindex(columns=labels, level=1)
        t['n'] = g[g.run == base].groupby(col, observed=True)['obj'].nunique()
        print(f'  by {name}:')
        print('   ' + t.to_string().replace('\n', '\n   '))
    csv = args.csv or out_root / f'_compare_{"_".join(labels)}.csv'
    df.to_csv(csv, index=False)
    print(f'\nper-object table: {csv}')


if __name__ == '__main__':
    main()
