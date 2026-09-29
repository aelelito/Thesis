"""
Paired comparison statistics.

    bash container/run_in_container.sh python tests/test_compare.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from autolabeling.utils.compare import add_metrics, common_objects, paired_stats, paired_table, sign_test_p

n = 0


def check(name, cond, detail=''):
    global n
    assert cond, f'FAILED: {name} {detail}'
    n += 1
    print('  ok ', name)


print('sign test')
check('8 wins, 0 losses -> p = 2/256', abs(sign_test_p(8, 0) - 2 / 256) < 1e-12)
check('5 vs 5 -> p = 1', sign_test_p(5, 5) == 1.0)
check('no data -> p = 1', sign_test_p(0, 0) == 1.0)

print('paired stats')
rng = np.random.default_rng(1)
base = rng.uniform(0.2, 1.0, 60)
better = base - 0.10 + rng.normal(0, 0.01, 60)           # consistently 0.10 lower
same = base + rng.normal(0, 0.05, 60)
st = paired_stats(base, better)
check('a consistent improvement: median diff ~ -0.10, CI excludes 0, all wins, tiny p',
      abs(st['median_diff'] + 0.10) < 0.01 and st['ci_hi'] < 0 and st['wins'] == 60 and st['p'] < 1e-10, st)
st = paired_stats(base, same)
check('pure noise: CI contains 0 and p is not small', st['ci_lo'] < 0 < st['ci_hi'] and st['p'] > 0.05, st)
st = paired_stats([1, 2, np.nan, 4], [1, 1, 3, 5])
check('NaN pairs are dropped; ties counted', st['n'] == 3 and st['ties'] == 1 and st['wins'] == 1 and st['losses'] == 1, st)
check('empty input does not crash', paired_stats([], [])['n'] == 0)

print('table helpers')
rows = []
for obj, cls in (('a', 'car'), ('b', 'car'), ('c', 'bicycle'), ('d', 'car')):
    for run, ce in (('m1', .5), ('m2', .3)):
        rows.append(dict(obj=obj, cls=cls, run=run, matched=True, center_err=ce, range_err=-ce, lat_err=0.1,
                         long_ratio=1.2, short_ratio=0.8, h_ratio=1.0, yaw_err=2.0))
rows.append(dict(obj='d', cls='car', run='m3', matched=True, center_err=.1, range_err=0, lat_err=0, long_ratio=1, short_ratio=1, h_ratio=1, yaw_err=0))
rows.append(dict(obj='e', cls='car', run='m1', matched=True, center_err=.5, range_err=.5, lat_err=0, long_ratio=1, short_ratio=1, h_ratio=1, yaw_err=1))
df = add_metrics(pd.DataFrame(rows))
check('size_err = mean |ratio-1| (0.2, 0.2, 0.0 -> 0.1333)', abs(df[df.cls == 'car'].size_err.iloc[0] - (0.2 + 0.2 + 0.0) / 3) < 1e-9)
check('rider classes get no size_err', df[df.cls == 'bicycle'].size_err.isna().all())
both = common_objects(df, ['m1', 'm2'])
check('common_objects keeps only objects present in every run (drops e: only in m1)', set(both.obj) == {'a', 'b', 'c', 'd'})
pt = paired_table(both, ['m1', 'm2'], base='m1')
ce = pt[pt.metric == 'center_err'].iloc[0]
check('paired table: m2 has lower center error on all 4 objects', ce.wins == 4 and abs(ce.median_diff + 0.2) < 1e-9, ce.to_dict())
check('size_err pairs skip the bicycle (n = 3)', pt[pt.metric == 'size_err'].iloc[0].n == 3)
print(f'\nall {n} checks passed')
