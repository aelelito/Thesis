"""
Both shipped YAML configs parse and only contain keys the pipeline reads.

    bash container/run_in_container.sh python tests/test_configs.py
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import autolabeling.pipeline as P
from autolabeling.models.sam3d_objects import POINTMAP_MODES, resolve_pointmap_mode

REMOVED = ('motion_compensation', 'obb_filter')
n = 0
for name in ('nuscenes', 'ecp'):
    raw = yaml.safe_load(open(ROOT / 'configs' / f'{name}.yaml'))
    for k in REMOVED:
        assert k not in raw, f'{name}: {k} still present'
    o = raw['sam3d_objects']
    for k in ('hull_anchoring', 'pl_ground_inlier_thres'):
        assert k not in o, f'{name}: {k} still present'
    assert 'max_range_m' not in raw['lidar_filters']
    assert 'correction_mode' not in raw['sam3d_body']
    assert set(raw['sam3d_body']) == {'b1_depth_correction', 'b2_ground_anchoring'}
    assert raw['ground_removal']['terraseg_ckpt']
    assert raw['lidar_aggregation']['use_aggregation'] is True
    assert P._resolve_mode(o["pointmap_mode"]) in POINTMAP_MODES.values()
    for m in (1, 2, 3, 4, *POINTMAP_MODES.values()):
        resolve_pointmap_mode(m)
    for cls in ('pedestrian', 'car', 'bus', 'truck', 'trailer', 'bicycle', 'motorcycle', 'construction vehicle'):
        assert cls in o['hdbscan'], f'{name}: hdbscan missing {cls}'
    n += 1
print(f'all configs OK ({n})')
