"""Paired comparison of pointmap modes on the same objects (numpy/pandas only)."""
import math

import numpy as np
import pandas as pd

RIDER_CLASSES = ('bicycle', 'motorcycle')          # their size is not comparable to GT without the rider merge
METRICS = ['center_err', 'abs_range_err', 'lat_err', 'size_err', 'long_ratio', 'short_ratio', 'h_ratio', 'yaw_err']


def add_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add abs_range_err and size_err (mean |ratio - 1| over long side, short side, height; NaN for rider classes)."""
    df = df.copy()
    df['abs_range_err'] = df['range_err'].abs()
    se = (df['long_ratio'].sub(1).abs() + df['short_ratio'].sub(1).abs() + df['h_ratio'].sub(1).abs()) / 3.0
    df['size_err'] = se.where(~df['cls'].isin(RIDER_CLASSES))
    return df


def sign_test_p(n_better: int, n_worse: int) -> float:
    """Two-sided exact sign test (ties excluded)."""
    n = n_better + n_worse
    if n == 0:
        return 1.0
    k = min(n_better, n_worse)
    return float(min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n))


def paired_stats(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    """
    b - a over the same objects (lower error is better, so a negative difference favours b). Returns the median
    difference with a bootstrap 95% CI (resampling objects), how many objects b wins / loses / ties, the win rate
    and the exact sign-test p-value. NaN pairs are dropped.
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
    return dict(n=n, median_diff=float(np.median(d)), ci_lo=float(np.percentile(boots, 2.5)), ci_hi=float(np.percentile(boots, 97.5)),
                wins=wins, losses=losses, ties=n - wins - losses,
                win_rate=wins / max(wins + losses, 1), p=sign_test_p(wins, losses))


def common_objects(df: pd.DataFrame, runs: list) -> pd.DataFrame:
    """Rows of the matched objects that exist in EVERY run."""
    m = df[df['matched'] & df['run'].isin(runs)]
    keys = m.groupby('obj')['run'].nunique()
    return m[m['obj'].isin(keys[keys == len(runs)].index)]


def medians(df: pd.DataFrame, runs: list, metrics=METRICS) -> pd.DataFrame:
    t = df.groupby('run')[metrics].median().reindex(runs)
    t.insert(0, 'n', df.groupby('run')['obj'].nunique().reindex(runs))
    return t


def paired_table(df: pd.DataFrame, runs: list, base: str, metrics=('center_err', 'abs_range_err', 'size_err', 'yaw_err')) -> pd.DataFrame:
    rows = []
    wide = {mt: df.pivot(index='obj', columns='run', values=mt) for mt in metrics}
    for r in runs:
        if r == base:
            continue
        for mt in metrics:
            st = paired_stats(wide[mt][base].values, wide[mt][r].values)
            rows.append(dict(run=r, vs=base, metric=mt, **st))
    return pd.DataFrame(rows)
