"""
Re-does the match between detections and GT using the nuScenes devkit's OWN criterion -- greedy, per class,
confidence-ranked, BEV CENTER DISTANCE (not IoU; nuScenes deliberately doesn't use 3D IoU, see the devkit's
`nuscenes.eval.detection.algo.accumulate`) -- at each of the four official thresholds {0.5, 1, 2, 4} m, and reports
claims 1/2/3/5 under that population instead of the LiDAR-containment rule `batch_eval.py` used.

Adapted to this project's scope: nuScenes' own evaluator merges all cameras into one 3D detection set per sample;
this analysis deliberately does NOT cross-camera-merge (a duplicate seen worse from a side camera is itself a
finding), so matching runs PER CAMERA against the keyframe's full GT list -- same distance criterion and the same
confidence-ranked greedy assignment, just scoped per camera instead of per sample.

Cheap by design: `batch_boxes.py` already computed every box's free-space/below-ground without touching any mesh,
so re-matching needs no GPU and no checkpoints -- it only joins CSVs. Claims 1/2/3/5 are recomputed fresh per
threshold; claim 4 (the floor control) needs the mesh at the SPECIFIC matched GT's pose, so it is NOT redone here --
it stays as reported against the LiDAR-containment match in contribution_ideas/phase0_mesh_freespace/results/summary.md.

Usage:
    bash container/run_in_container.sh python contribution_ideas/phase0_mesh_freespace/scripts/rematch.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent.parent
RESULTS_DIR = HERE / 'results'
BOXES_DIR = RESULTS_DIR / 'boxes'
OUT_MD = RESULTS_DIR / 'summary_nuscenes_matching.md'
THRESHOLDS = (0.5, 1.0, 2.0, 4.0)
MIN_CATEGORY_N = 20

sys.path.insert(0, str(HERE.parent.parent / 'autolabeling' / 'src'))
sys.path.insert(0, str(HERE.parent.parent))
from contribution_ideas.phase0_mesh_freespace.scripts.aggregate import (  # noqa: E402
    mask_fit_table, not_implied_table, paired_stats,
)

MAPPING_YAML = HERE.parent.parent / 'evaluation' / 'class_mapping' / '8class.yaml'


def load_category_map() -> dict:
    """{raw nuScenes GT category -> our class name, space-separated}. ECP's category.json names already match our
    class names except for the underscore in construction_vehicle, handled by `norm_cls` below."""
    raw = yaml.safe_load(MAPPING_YAML.read_text())['mapping_nuscenes']
    return {k: norm_cls(v) for k, v in raw.items() if v is not None}


def norm_cls(s) -> str:
    return None if s is None else str(s).replace('_', ' ')


# ── Load everything ──────────────────────────────────────────────────────────────────────────

def load_boxes():
    det = pd.concat([pd.read_csv(p) for p in sorted(BOXES_DIR.glob('*_det.csv'))], ignore_index=True)
    gt = pd.concat([pd.read_csv(p) for p in sorted(BOXES_DIR.glob('*_gt.csv'))], ignore_index=True)
    cat_map = load_category_map()
    gt['cls'] = gt['category'].map(cat_map)
    # ECP: category.json names are already coarse (car, bicycle, ...); just normalise the underscore in
    # construction_vehicle. Always attempted for whatever the nuScenes-style map left unmapped -- do NOT gate this
    # on "does any category contain an underscore" (most ECP classes don't; that gate silently dropped every ECP
    # GT box the first time this ran).
    gt['cls'] = gt['cls'].fillna(gt['category'].map(norm_cls))
    gt = gt.dropna(subset=['cls'])
    det['dataset'] = det['scene'].apply(lambda s: 'ecp' if str(s).startswith('scene-euro') else 'nuscenes_mini')
    gt['dataset'] = gt['scene'].apply(lambda s: 'ecp' if str(s).startswith('scene-euro') else 'nuscenes_mini')
    return det, gt


def load_mesh_metrics() -> pd.DataFrame:
    """The GT-independent columns from the big batch_eval.py CSVs (recall, iou, clean, mesh/obb free-space,
    below-ground, not_implied, n_violating, surf_unknown) -- correct regardless of matching protocol."""
    keep = ['scene', 'cam', 'frame_idx', 'cls', 'sam3_index', 'recall', 'iou', 'clean', 'surf_free', 'surf_unknown',
           'mesh_free_frac', 'mesh_free_m3', 'obb_free_frac', 'obb_free_m3',
           'pipeline_obb_free_frac', 'pipeline_obb_free_m3', 'mesh_below_ground', 'obb_below_ground',
           'pipeline_obb_below_ground', 'n_violating', 'not_implied']
    dfs = [pd.read_csv(p)[keep] for p in sorted(RESULTS_DIR.glob('*.csv')) if p.parent == RESULTS_DIR]
    return pd.concat(dfs, ignore_index=True)


# ── nuScenes-style greedy center-distance matching, per (scene, frame_idx, cam, cls) ───────────

def match_one_group(dets: pd.DataFrame, gts: pd.DataFrame, threshold: float) -> dict:
    """{det row index -> gt_index} for one (scene, frame_idx, cam, cls) group, greedy by score descending."""
    if len(dets) == 0 or len(gts) == 0:
        return {}
    taken = set()
    out = {}
    for idx in dets.sort_values('score', ascending=False).index:
        dx, dy = dets.loc[idx, 'center_x'], dets.loc[idx, 'center_y']
        d2 = (gts['center_x'] - dx) ** 2 + (gts['center_y'] - dy) ** 2
        order = d2.sort_values().index
        for gi in order:
            if gi in taken:
                continue
            if d2[gi] > threshold ** 2:
                break
            taken.add(gi)
            out[idx] = gts.loc[gi, 'gt_index']
            break
    return out


def match_all(det: pd.DataFrame, gt: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Adds a `gt_index` column to a COPY of det: the matched GT's index at this threshold, or NaN."""
    det = det.copy()
    det['gt_index'] = np.nan
    for (scene, frame_idx, cam, cls), ddf in det.groupby(['scene', 'frame_idx', 'cam', 'cls']):
        gdf = gt[(gt['scene'] == scene) & (gt['frame_idx'] == frame_idx) & (gt['cls'] == cls)]
        m = match_one_group(ddf, gdf, threshold)
        for idx, gi in m.items():
            det.loc[idx, 'gt_index'] = gi
    return det


def attach_gt_stats(det: pd.DataFrame, gt: pd.DataFrame) -> pd.DataFrame:
    g = gt.set_index(['scene', 'frame_idx', 'gt_index'])[['free_frac', 'free_m3', 'below_ground', 'speed']]
    g.columns = ['gt_free_frac', 'gt_free_m3', 'gt_below_ground', 'gt_speed']
    det = det.join(g, on=['scene', 'frame_idx', 'gt_index'])
    det['has_gt'] = det['gt_index'].notna()
    det['observed'] = det['surf_unknown'] < 0.95
    det['matched'] = det['has_gt'] & det['observed']
    return det


# ── Tables (mirrors aggregate.py's claims 2/3, claim 1/5 reused directly) ──────────────────────

def freespace_table(df, group_cols):
    m = df[df['matched']]
    rows = []
    for key, g in m.groupby(group_cols):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_cols, key), n=len(g))
        for name, frac_col in (('mesh', 'mesh_free_frac'), ('obb', 'pipeline_obb_free_frac')):
            m3_col = frac_col.replace('_frac', '_m3')
            row[f'{name}_frac_median'] = g[frac_col].median()
            st = paired_stats(g['gt_free_frac'].values, g[frac_col].values)
            row[f'{name}_excess_frac_median'], row[f'{name}_excess_frac_p'] = st['median_diff'], st['p']
            st3 = paired_stats(g['gt_free_m3'].values, g[m3_col].values)
            row[f'{name}_excess_m3_median'], row[f'{name}_excess_m3_p'] = st3['median_diff'], st3['p']
        row['gt_frac_median'] = g['gt_free_frac'].median()
        rows.append(row)
    return pd.DataFrame(rows)


def ground_table(df, group_cols):
    m = df[df['has_gt']]
    rows = []
    for key, g in m.groupby(group_cols):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_cols, key), n=len(g))
        for name, col in (('mesh', 'mesh_below_ground'), ('obb', 'pipeline_obb_below_ground')):
            row[f'{name}_median'] = g[col].median()
            st = paired_stats(g['gt_below_ground'].values, g[col].values)
            row[f'{name}_excess_median'], row[f'{name}_excess_p'] = st['median_diff'], st['p']
        row['gt_median'] = g['gt_below_ground'].median()
        rows.append(row)
    return pd.DataFrame(rows)


def to_markdown(df):
    return ('_(no rows)_\n' if df.empty else df.round(4).to_markdown(index=False) + '\n')


def main():
    det_raw, gt_raw = load_boxes()
    mesh = load_mesh_metrics()
    det_raw = det_raw.merge(mesh, on=['scene', 'cam', 'frame_idx', 'cls', 'sam3_index'], how='left', validate='one_to_one')
    print(f'{len(det_raw)} detections, {len(gt_raw)} GT boxes (after category mapping) loaded for matching.')

    lines = ['# nuScenes-style (center-distance) matching -- claims 1/2/3/5 at each threshold', '',
            'Greedy, per class, confidence-ranked, BEV center distance -- the devkit\'s own criterion, NOT IoU. Run '
            'per camera (this analysis does not cross-camera-merge). Claim 4 (the floor control) is not redone here '
            '-- it needs the mesh at the matched GT\'s specific pose; see summary.md for that one, matched by LiDAR '
            'containment instead.', '']
    agreement_prev = None
    for th in THRESHOLDS:
        det = match_all(det_raw, gt_raw, th)
        det = attach_gt_stats(det, gt_raw)
        n_matched = int(det['has_gt'].sum())
        print(f'\n################ threshold {th} m: {n_matched} / {len(det)} detections matched ################')
        lines.append(f'## Threshold {th} m -- {n_matched} / {len(det)} detections matched')

        t1 = mask_fit_table(det, ['dataset'])
        t2 = freespace_table(det, ['dataset'])
        t3 = ground_table(det, ['dataset'])
        t5 = not_implied_table(det, ['dataset'])
        for name, t in (('1. Mask fit', t1), ('2. Free-space vs GT floor', t2), ('3. Below-ground', t3),
                       ('5. Not implied by in-mask depth', t5)):
            print(f'--- {name} ---')
            print(t.round(4).to_string(index=False))
            lines += [f'### {name}', to_markdown(t)]

        by_cat = det[det.groupby(['dataset', 'cls'])['cls'].transform('size') >= MIN_CATEGORY_N]
        if len(by_cat):
            t2c = freespace_table(by_cat, ['dataset', 'cls'])
            print('--- 2. Free-space vs GT floor (by category) ---')
            print(t2c.round(4).to_string(index=False))
            lines += ['### 2. Free-space vs GT floor (by category)', to_markdown(t2c)]

    OUT_MD.write_text('\n'.join(lines))
    print(f'\nWrote {OUT_MD}')


if __name__ == '__main__':
    main()
