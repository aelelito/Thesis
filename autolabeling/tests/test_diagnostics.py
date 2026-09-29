"""
GT association + per-object errors, on synthetic geometry.

    bash container/run_in_container.sh python tests/test_diagnostics.py
"""
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from autolabeling.utils.diagnostics import associate_gt, associate_gt_lidar, object_errors, resolve_duplicate_gt
from autolabeling.utils.geometry import project

n = 0


def check(name, cond, detail=''):
    global n
    assert cond, f'FAILED: {name} {detail}'
    n += 1
    print('  ok ', name)


H, W = 480, 640
K = np.array([[500., 0, 320], [0, 500., 240], [0, 0, 1]])
R_c2e = np.array([[0, 0, 1.], [-1, 0, 0], [0, -1, 0]])          # cam (x right, y down, z fwd) -> ego (x fwd, y left, z up)
frame = NS(R_c2e=R_c2e, t_c2e=np.zeros(3), K=K)


def box(cx, cy, l=4.0, w=1.8, h=1.6, yaw=0.0, cat='vehicle.car'):
    c = np.array([cx, cy, h / 2])
    R = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    lx = np.array([1, 1, 1, 1, -1, -1, -1, -1]) * l / 2
    ly = np.array([1, -1, -1, 1, 1, -1, -1, 1]) * w / 2
    lz = np.array([1, 1, -1, -1, 1, 1, -1, -1]) * h / 2
    return dict(center=c, size=(w, l, h), yaw=yaw, corners_3d=c + (R @ np.stack([lx, ly, lz])).T,
                category=cat, num_lidar_pts=10)


def mask_of(g):
    pc = (R_c2e.T @ (g['corners_3d']).T).T
    m = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(m, cv2.convexHull(project(pc, K).astype(np.int32)), 1)
    return m.astype(bool)


near, far, ped = box(12, 0), box(30, 4), box(15, -3, 0.6, 0.6, 1.7, cat='human.pedestrian')
gts = [far, ped, near]

print('association')
i, iou = associate_gt(gts, mask_of(near), frame)
check('a mask matches the GT box it was rendered from', i == 2 and iou > 0.9, (i, iou))
i, _ = associate_gt(gts, mask_of(far), frame)
check('...also for a far box', i == 1 - 1, i)
i, _ = associate_gt(gts, mask_of(ped), frame)
check('pedestrians are ignored for vehicle masks (no vehicle overlaps)', i is None, i)
i, iou = associate_gt(gts, np.zeros((H, W), bool), frame)
check('empty mask -> no match', i is None)
behind = box(-10, 0)
i, _ = associate_gt([behind], mask_of(near), frame)
check('a box behind the camera is skipped', i is None)

print('LiDAR association')
def surface_points(g, n=400, seed=0):
    rng = np.random.default_rng(seed)
    l, w, h = g['size'][1], g['size'][0], g['size'][2]
    loc = (rng.random((n, 3)) - 0.5) * np.array([l, w, h])
    c, s = np.cos(g['yaw']), np.sin(g['yaw'])
    return g['center'] + np.stack([c * loc[:, 0] - s * loc[:, 1], s * loc[:, 0] + c * loc[:, 1], loc[:, 2]], 1)

near2, far2 = box(10, 0.5, 4.0, 1.8, 1.6), box(22, -1.0, 4.0, 1.8, 1.6)     # far car is partly hidden behind the near one in the image
pts = np.vstack([surface_points(near2), surface_points(far2, seed=1)])
gts2 = [near2, far2]
m_near, m_far = mask_of(near2), mask_of(far2)
visible_far = m_far & ~m_near                                              # what is left of the far car after occlusion
assert visible_far.sum() > 200 and (m_far & m_near).sum() > 200, 'test setup: the far car must be partly occluded'
i, frac, amb = associate_gt_lidar(gts2, visible_far, frame, pts)
check('occluded far car: the LiDAR in its visible pixels sits in the FAR box', i == 1 and not amb, (i, frac, amb))
i, frac, amb = associate_gt_lidar(gts2, m_near, frame, pts)
check('near car -> near box', i == 0, (i, frac, amb))
i, frac, amb = associate_gt_lidar(gts2, m_far, frame, pts[:2])
check('too few LiDAR points -> no match', i is None)
dup = box(10.3, 0.6, 4.0, 1.8, 1.6)                                          # a second GT box on top of the near car
i, frac, amb = associate_gt_lidar([near2, dup], m_near, frame, np.vstack([surface_points(near2), surface_points(dup, seed=2)]))
check('two boxes claiming the same points -> flagged ambiguous', i is not None and amb, (i, frac, amb))
i, frac, amb = associate_gt_lidar([ped], mask_of(ped), frame, surface_points(ped), category_prefix='vehicle')
check('other categories are ignored', i is None)

print('match hygiene')
# a small mask: 3 of its 10 points bleed from the near car's box, 7 belong to something else -> below min_fraction
bleed_pts = np.vstack([surface_points(near2, n=400)[:400], surface_points(far2, n=400, seed=5)])
i, frac, amb = associate_gt_lidar(gts2, m_near, frame, bleed_pts)
check('sanity: a well-covered mask still matches with the default min_fraction', i == 0, (i, frac))
# 4 points on the near car + 20 points far behind it along the same viewing rays: all 24 fall in the near car's mask
far_behind = np.array([[40.0, 2.0 + dy, 0.8] for dy in np.linspace(-0.4, 0.4, 20)])
mixed = np.vstack([surface_points(near2, n=4), far_behind])
i, frac, amb = associate_gt_lidar(gts2, m_near, frame, mixed, min_fraction=0.0)
check('setup: with no fraction limit those 4 points would claim the near box', i == 0 and frac < 0.3, (i, frac))
i, frac, amb = associate_gt_lidar(gts2, m_near, frame, mixed)
check('default min_fraction (0.3): only ~17% of the in-mask points are inside the box -> no match', i is None, (i, frac))
check('resolve_duplicate_gt: the larger mask keeps the GT box, the smaller loses it',
      resolve_duplicate_gt({'a': (5, 20000), 'b': (5, 900), 'c': (7, 500)}) == {'b'})
check('resolve_duplicate_gt: no duplicates -> nothing dropped', resolve_duplicate_gt({'a': (1, 10), 'b': (2, 10)}) == set())
check('resolve_duplicate_gt: three claimants -> the two smaller lose',
      resolve_duplicate_gt({'a': (5, 300), 'b': (5, 900), 'c': (5, 100)}) == {'a', 'c'})

print('errors')
g = near
e = object_errors(g['center'], [4.0, 1.8, 1.6], g['yaw'], g)
check('perfect prediction -> zeros / ones', e['center_err'] < 1e-9 and abs(e['long_ratio'] - 1) < 1e-9
      and abs(e['short_ratio'] - 1) < 1e-9 and abs(e['h_ratio'] - 1) < 1e-9 and e['yaw_err'] < 1e-6, e)
e = object_errors(g['center'] + np.array([2.0, 0, 0]), [4.0, 1.8, 1.6], g['yaw'], g)
check('2 m too far along the viewing ray -> range_err = +2, lat_err = 0', abs(e['range_err'] - 2) < 1e-6 and e['lat_err'] < 1e-6, e)
e = object_errors(g['center'] + np.array([0, 1.5, 0]), [4.0, 1.8, 1.6], g['yaw'], g)
check('sideways error goes to lat_err', abs(e['lat_err'] - 1.5) < 1e-6 and abs(e['range_err']) < 1e-6, e)
e = object_errors(g['center'], [3.0, 1.35, 1.2], g['yaw'], g)
check('a box 25% too small -> ratios 0.75', all(abs(e[k] - 0.75) < 1e-9 for k in ('long_ratio', 'short_ratio', 'h_ratio')), e)
e = object_errors(g['center'], [1.8, 4.0, 1.6], g['yaw'] + np.pi / 2, g)
check('length/width swapped is handled (footprint sides sorted)', abs(e['long_ratio'] - 1) < 1e-9 and abs(e['short_ratio'] - 1) < 1e-9)
check('...but a 90 degree yaw is a 90 degree error', abs(e['yaw_err'] - 90) < 1e-6, e['yaw_err'])
e = object_errors(g['center'], [4.0, 1.8, 1.6], g['yaw'] + np.pi, g)
check('a front/back flip (180 deg) is not counted as a heading error', e['yaw_err'] < 1e-6)
e = object_errors(g['center'], [4.0, 1.8, 1.6], g['yaw'] + np.radians(20), g)
check('20 degree yaw error', abs(e['yaw_err'] - 20) < 1e-6, e['yaw_err'])
print(f'\nall {n} checks passed')
