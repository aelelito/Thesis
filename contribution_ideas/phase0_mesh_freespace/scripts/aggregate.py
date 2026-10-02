"""
Aggregation for the full-dataset mesh / free-space / ground evaluation (PLAN.md's "Reporting" section).

Reads every CSV in contribution_ideas/phase0_mesh_freespace/results/ (one row per object, written by
autolabeling.batch_eval), tags each with its dataset, and produces the tables PLAN.md specifies: mask fit, free-space
touch of mesh / pipeline-OBB vs the GT floor (fraction AND absolute volume -- the fraction alone can hide the worst
cases, per phase0's own finding), below-ground reach, the GT-floor shape-isolation control, and the "not implied by
in-mask depth" share -- per dataset, and per category where n allows it. Paired stats (median difference, bootstrap
CI, exact sign test) reuse `autolabeling.utils.compare.paired_stats`, the same machinery behind
`notes/sam3d_objects_mode_decision.md`.

Usage:
    bash container/run_in_container.sh python contribution_ideas/phase0_mesh_freespace/scripts/aggregate.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, '/workspace/autolabeling/src')
from scipy.stats import binomtest


def paired_stats(a, b, n_boot: int = 2000, seed: int = 0) -> dict:
    """
    Same statistic as autolabeling.utils.compare.paired_stats (median of b - a over paired objects, bootstrap 95% CI,
    wins/losses, exact-sign-test p) but via scipy's binomtest instead of a raw binomial-coefficient sum -- that sum
    overflows Python floats once n reaches a few hundred, which the mode-decision sweeps (n ~ 100-200) never hit but
    this dataset-wide evaluation (n in the thousands) does.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    d = (b - a)[ok]
    n = len(d)
    if n == 0:
        return dict(n=0, median_diff=np.nan, ci_lo=np.nan, ci_hi=np.nan, wins=0, losses=0, ties=0, win_rate=np.nan, p=np.nan)
    rng = np.random.default_rng(seed)
    boots = np.median(d[rng.integers(0, n, (n_boot, n))], axis=1)
    wins, losses = int((d < -1e-9).sum()), int((d > 1e-9).sum())
    p = 1.0 if (wins + losses) == 0 else float(binomtest(min(wins, losses), wins + losses, 0.5, alternative='two-sided').pvalue)
    return dict(n=n, median_diff=float(np.median(d)), ci_lo=float(np.percentile(boots, 2.5)), ci_hi=float(np.percentile(boots, 97.5)),
               wins=wins, losses=losses, ties=n - wins - losses, win_rate=wins / max(wins + losses, 1), p=p)

RESULTS_DIR = Path(__file__).resolve().parent.parent / 'results'
OUT_MD = RESULTS_DIR / 'summary.md'
MIN_CATEGORY_N = 20           # below this a category's own row is too noisy to report on its own
OBSERVED_UNKNOWN_MAX = 0.95   # surf_unknown below this = "the LiDAR actually saw something of this object";
                              # above it, no-evidence is not the same as no-violation, so it's excluded, not scored 0.
RIDER_CLASSES = {'bicycle', 'motorcycle'}  # GT box includes the rider; the floor control compares the GENERATED
                                           # mesh's own shape (bike-only) against the GT box, so it's excluded here
                                           # the same way PLAN.md excludes it -- not a bug, a geometry mismatch.


# ── Load ─────────────────────────────────────────────────────────────────────────────────────

def load_all() -> pd.DataFrame:
    dfs = []
    for p in sorted(RESULTS_DIR.glob('*.csv')):
        d = pd.read_csv(p)
        d['dataset'] = 'ecp' if p.stem == 'ecp' else 'nuscenes_mini'
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)
    df['observed'] = df['surf_unknown'] < OBSERVED_UNKNOWN_MAX
    df['matched'] = df['has_gt'] & df['observed']     # the population every free-space table below uses
    return df


# ── Per-table builders ───────────────────────────────────────────────────────────────────────

def mask_fit_table(df: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    """Claim 1: recall for every object (occlusion-robust); IoU only on the clean subset."""
    rows = []
    for key, g in df.groupby(group_cols):
        key = key if isinstance(key, tuple) else (key,)
        clean = g[g['clean']]
        rows.append(dict(zip(group_cols, key), n=len(g), recall_median=g['recall'].median(),
                         n_clean=len(clean), clean_frac=len(clean) / len(g),
                         iou_median=clean['iou'].median() if len(clean) else np.nan))
    return pd.DataFrame(rows)


def freespace_table(df: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    """Claim 2: free-space touch of the mesh and the pipeline OBB, against the GT-box floor."""
    m = df[df['matched']]
    rows = []
    for key, g in m.groupby(group_cols):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_cols, key), n=len(g))
        for name, frac_col in (('mesh', 'mesh_free_frac'), ('obb', 'pipeline_obb_free_frac')):
            m3_col = frac_col.replace('_frac', '_m3')
            row[f'{name}_frac_median'] = g[frac_col].median()
            st = paired_stats(g['gt_free_frac'].values, g[frac_col].values)      # median(pred - GT)
            row[f'{name}_excess_frac_median'], row[f'{name}_excess_frac_p'] = st['median_diff'], st['p']
            st3 = paired_stats(g['gt_free_m3'].values, g[m3_col].values)
            row[f'{name}_excess_m3_median'], row[f'{name}_excess_m3_p'] = st3['median_diff'], st3['p']
        row['gt_frac_median'] = g['gt_free_frac'].median()
        row['gt_m3_median'] = g['gt_free_m3'].median()
        rows.append(row)
    return pd.DataFrame(rows)


def ground_table(df: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    """Claim 3: below-ground reach (PseudoLabeler ground). Uses every GT match -- the ground estimate comes from
    LiDAR returns NEAR the object, not from the object's own surface, so it doesn't need `observed`."""
    m = df[df['has_gt']]
    rows = []
    for key, g in m.groupby(group_cols):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_cols, key), n=len(g))
        for name, col in (('mesh', 'mesh_below_ground'), ('obb', 'pipeline_obb_below_ground')):
            row[f'{name}_median'] = g[col].median()
            row[f'{name}_frac_below_1cm'] = float((g[col] > 0.01).mean())
            st = paired_stats(g['gt_below_ground'].values, g[col].values)
            row[f'{name}_excess_median'], row[f'{name}_excess_p'] = st['median_diff'], st['p']
        row['gt_median'] = g['gt_below_ground'].median()
        rows.append(row)
    return pd.DataFrame(rows)


def floor_table(df: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    """Claim 4: the mesh re-centred to the GT pose/yaw, own shape kept. Splits the violation into a placement part
    (surf_free -> floor_free: what correct placement alone would remove) and a shape part (floor_free -> the true
    GT box: what remains once placement is already correct, i.e. what only touching shape could fix).
    Excludes bicycles/motorcycles (GT box includes the rider; see RIDER_CLASSES above)."""
    m = df[df['matched'] & df['floor_free'].notna() & ~df['cls'].isin(RIDER_CLASSES)]
    rows = []
    for key, g in m.groupby(group_cols):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_cols, key), n=len(g),
                   surf_free_median=g['surf_free'].median(), floor_free_median=g['floor_free'].median(),
                   gt_free_median=g['gt_free_frac'].median())
        st_place = paired_stats(g['floor_free'].values, g['surf_free'].values)
        row['placement_excess_median'], row['placement_p'] = st_place['median_diff'], st_place['p']
        st_shape = paired_stats(g['gt_free_frac'].values, g['floor_free'].values)
        row['shape_excess_median'], row['shape_p'] = st_shape['median_diff'], st_shape['p']
        rows.append(row)
    return pd.DataFrame(rows)


def not_implied_table(df: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    """Claim 5: of the mesh surface that touches free space, the share a 'fit the mesh to its own in-mask LiDAR
    points' correction would NOT already remove. Only defined where there is a violation to begin with."""
    m = df[df['matched'] & (df['n_violating'] > 0)]
    rows = []
    for key, g in m.groupby(group_cols):
        key = key if isinstance(key, tuple) else (key,)
        rows.append(dict(zip(group_cols, key), n=len(g), not_implied_median=g['not_implied'].median(),
                         frac_majority_not_implied=float((g['not_implied'] > 0.5).mean())))
    return pd.DataFrame(rows)


# ── Formatting ───────────────────────────────────────────────────────────────────────────────

def _sig(p):
    return '**' if (pd.notna(p) and p < 0.05) else ''


def to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return '_(no rows)_\n'
    return df.round(4).to_markdown(index=False) + '\n'


def restrict_categories(df: pd.DataFrame) -> pd.DataFrame:
    counts = df.groupby(['dataset', 'cls']).size()
    keep = counts[counts >= MIN_CATEGORY_N].index
    return df[df.set_index(['dataset', 'cls']).index.isin(keep)]


def main():
    df = load_all()
    print(f'Loaded {len(df)} object rows from {len(list(RESULTS_DIR.glob("*.csv")))} CSV file(s).')
    print(f'  has_gt: {df.has_gt.sum()}   observed (surf_unknown < {OBSERVED_UNKNOWN_MAX}): {df.observed.sum()}   '
         f'matched (both): {df.matched.sum()}')

    by_cat = restrict_categories(df)

    sections = [
        ('1. Mask fit', mask_fit_table, ['dataset'], ['dataset', 'cls']),
        ('2. Free-space touch vs the GT-box floor', freespace_table, ['dataset'], ['dataset', 'cls']),
        ('3. Below-ground reach (PseudoLabeler)', ground_table, ['dataset'], ['dataset', 'cls']),
        ('4. Shape-isolated free-space (GT-floor control)', floor_table, ['dataset'], ['dataset', 'cls']),
        ('5. Not implied by in-mask depth', not_implied_table, ['dataset'], ['dataset', 'cls']),
    ]

    lines = ['# Mesh / free-space / ground evaluation -- summary', '',
            f'n = {len(df)} objects total, {df.has_gt.sum()} GT-matched, {df.matched.sum()} GT-matched AND observed '
            f'(surf_unknown < {OBSERVED_UNKNOWN_MAX}). `**` marks p < 0.05 (sign test). Categories need at least '
            f'{MIN_CATEGORY_N} matched objects to get their own row.', '']
    for title, fn, g1, g2 in sections:
        print(f'\n=== {title} (per dataset) ===')
        t1 = fn(df, g1)
        print(t1.round(4).to_string(index=False))
        print(f'\n=== {title} (per dataset x category, n >= {MIN_CATEGORY_N}) ===')
        t2 = fn(by_cat, g2)
        print(t2.round(4).to_string(index=False))
        lines += [f'## {title}', '### per dataset', to_markdown(t1), '### per dataset x category', to_markdown(t2)]

    OUT_MD.write_text('\n'.join(lines))
    print(f'\nWrote {OUT_MD}')


if __name__ == '__main__':
    main()
