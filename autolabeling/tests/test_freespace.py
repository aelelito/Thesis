"""
Free-space grid, mesh queries and mesh <-> mask metrics, on synthetic geometry with known answers.

    bash container/run_in_container.sh python tests/test_freespace.py
"""
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from autolabeling.utils import freespace as fs
from autolabeling.utils import mesh_mask as mm

n = 0


def check(name, cond, detail=''):
    global n
    assert cond, f'FAILED: {name} {detail}'
    n += 1
    print('  ok ', name)


def near(a, b, tol):
    return abs(a - b) <= tol


def wall_sweep(x_wall=10.0, half=6.0, spacing=0.05):
    """A flat wall at x = x_wall seen by a sensor at the origin: one return per (y, z) grid point."""
    y, z = np.meshgrid(np.arange(-half, half, spacing), np.arange(-1.0, 3.0, spacing))
    return np.stack([np.full(y.size, x_wall), y.ravel(), z.ravel()], axis=1)


def cube(center, size):
    c, h = np.asarray(center, float), size / 2
    v = np.array([[x, y, z] for x in (-h, h) for y in (-h, h) for z in (-h, h)]) + c
    f = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                  [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    return v, f


grid = fs.build_grid(wall_sweep(), np.zeros(3), vs=0.1, xy_half=20.0)

st = grid.state(np.array([[5.0, 0.0, 0.5], [10.0, 0.0, 0.5], [13.0, 0.0, 0.5]]))
check('in front of a return = free, on it = occupied, behind it = unknown', list(st) == [fs.FREE, fs.OCC, fs.UNKNOWN], st)

d = grid.dist_to_occupied(np.array([[5.0, 0.0, 0.5], [9.5, 0.0, 0.5]]))
check('distance to the surface along the beam', near(d[0], 5.0, 0.2) and near(d[1], 0.5, 0.2), d)

a = fs.freespace_stats(grid, fs.box_lattice([6.0, 0.0, 0.5], 2.0, 2.0, 1.0, 0.0))
b = fs.freespace_stats(grid, fs.box_lattice([12.0, 0.0, 0.5], 2.0, 2.0, 1.0, 0.0))
check('a box floating in front of the wall is (almost) all free, also with a 0.3 m margin',
      a['frac_free'] > 0.95 and a['frac_free_m0.3'] > 0.95, a)
check('a box behind the wall is never free (unknown)', b['frac_free'] == 0 and b['frac_unknown'] > 0.95, b)

s = fs.freespace_stats(grid, fs.box_lattice([9.85, 0.0, 0.5], 0.2, 2.0, 1.0, 0.0))     # spans x = 9.75 .. 9.95, wall at 10
check('the margin removes points that only sit at the surface', s['frac_free'] > 0.5 and s['frac_free_m0.3'] == 0, s)

v, f = cube([0, 0, 0], 2.0)
pts = fs.mesh_volume_points(v, f, vs=0.1)
check('filled volume of a 2 m cube is about 8 m3 (a voxelised shell adds about half a voxel per side: 9.3)', near(len(pts) * 0.1 ** 3, 8.0, 1.5), len(pts) * 0.001)
S = fs.sample_mesh_surface(*cube([1, 2, 3], 2.0), 2000)
check('surface samples lie on the cube', np.isclose(np.abs(S - [1, 2, 3]).max(axis=1), 1.0, atol=1e-6).all())

K = np.array([[800.0, 0, 320], [0, 800.0, 240], [0, 0, 1]])
v, f = cube([1.0, 0.5, 10.0], 2.0)                       # camera frame: z forward, y down
sil, depth = mm.render_mesh(v, f, K, 480, 640, backend='pytorch3d')
sil2, depth2 = mm.render_mesh(v, f, K, 480, 640, backend='painter')
check('painter and pytorch3d silhouettes agree', (sil ^ sil2).sum() < 0.02 * sil.sum() and abs(np.nanmedian(depth) - np.nanmedian(depth2)) < 0.3, ((sil ^ sil2).sum(), sil.sum()))
u, w = K[0, 0] * v[:, 0] / v[:, 2] + K[0, 2], K[1, 1] * v[:, 1] / v[:, 2] + K[1, 2]
ys, xs = np.where(sil)
check('silhouette extent equals the pinhole projection of the cube (this also checks the axis conventions)',
      near(xs.min(), u.min(), 1.5) and near(xs.max(), u.max(), 1.5) and near(ys.min(), w.min(), 1.5) and near(ys.max(), w.max(), 1.5),
      (xs.min(), xs.max(), ys.min(), ys.max(), u.min(), u.max(), w.min(), w.max()))
check('depth of the front face', near(np.nanmin(depth), 9.0, 0.05), np.nanmin(depth))

sil = np.zeros((20, 20), bool); sil[5:15, 5:15] = True
mask = np.zeros((20, 20), bool); mask[5:15, 5:12] = True
m = mm.mask_agreement(sil, mask)
check('recall / leak / iou', near(m['recall'], 1.0, 1e-9) and near(m['leak'], 0.3, 1e-9) and near(m['iou'], 0.7, 1e-9), m)
occ = np.zeros((20, 20), bool); occ[:, 12:] = True
check('leak behind another detection is explained', mm.mask_agreement(sil, mask, occ)['leak_free'] == 0.0)
dep = np.full((20, 20), np.nan, np.float32); dep[5:15, 5:15] = 10.0
r = mm.depth_residuals(dep, mask, np.array([6.0, 7.0, 0.0]), np.array([6.0, 7.0, 0.0]), np.array([10.5, 10.5, 10.5]))
check('depth residual (return behind the mesh surface = +0.5)', near(r['median'], 0.5, 1e-6) and r['n_in_mask'] == 2, r)

# ── ground estimate, penetration depth, box projection ──
rng = np.random.default_rng(0)
gx, gy = np.meshgrid(np.arange(-10, 10, 0.3), np.arange(-10, 10, 0.3))
ground = np.stack([gx.ravel(), gy.ravel(), -1.7 + 0.02 * rng.standard_normal(gx.size)], axis=1)
car = np.stack([rng.uniform(-2, 2, 500), rng.uniform(-0.9, 0.9, 500), rng.uniform(-1.7, -0.3, 500)], axis=1)
z = fs.local_ground_z(np.vstack([ground, car]), [0, 0, -1.0], 4.0, 1.8, 0.0)
check('ground height next to an object (returns on the object itself are ignored)', near(z, -1.7, 0.03), z)
check('too few ring points -> NaN', np.isnan(fs.local_ground_z(car, [0, 0, -1.0], 4.0, 1.8, 0.0)))
sv = fs.freespace_stats(grid, fs.box_lattice([6.0, 0.0, 0.5], 2.0, 2.0, 1.0, 0.0))
check('penetration depth of a box floating 4 m in front of the wall', near(sv['viol_depth_p50'], 4.0, 1.2) and near(sv['viol_depth_p90'], 4.9, 1.2), sv)
from autolabeling.utils import freespace_viz as fv
fr = NS(R_c2e=np.array([[0, 0, 1.], [-1, 0, 0], [0, -1, 0]]), t_c2e=np.zeros(3), K=np.array([[500., 0, 320], [0, 500., 240], [0, 0, 1]]))
px = fv.project_corners(fr, fv.box_corners([10.0, 0.0, 0.0], 2.0, 2.0, 2.0, 0.0))
check('box centre projects to the principal point, depth = distance ahead', near(px[:, 0].mean(), 320, 1.0) and near(px[:, 1].mean(), 240, 1.0) and near(px[:, 2].mean(), 10.0, 1e-6), px)

# ── controls ──
from autolabeling.utils import controls as ct
V0, F0 = cube([0, 0, 0], 1.0)
V0 = V0 * np.array([4.0, 1.8, 1.5])                                        # a 4 x 1.8 x 1.5 box centred at the origin, yaw 0
V1 = ct.align_mesh_to_box(V0, [0, 0, 0], 0.0, (4.0, 1.8, 1.5), [10, 5, 1], np.pi / 2 + 0.05)
check('align: centre lands on the target', np.allclose(V1.mean(axis=0), [10, 5, 1], atol=1e-9), V1.mean(axis=0))
V2 = ct.align_mesh_to_box(V0, [0, 0, 0], 0.0, (4.0, 1.8, 1.5), [10, 5, 1], 0.0, (5.0, 2.0, 1.4))
check('align: stretched to the target size', np.allclose(np.ptp(V2, axis=0), [5.0, 2.0, 1.4]), np.ptp(V2, axis=0))
yy, zz = np.meshgrid(np.linspace(-1, 1, 30), np.linspace(-0.5, 1.5, 30))
plane = np.stack([np.full(yy.size, 9.5), yy.ravel(), zz.ravel()], axis=1)  # a surface 0.5 m in front of the wall
wall_pts = wall_sweep()
cur = ct.ray_shift_curve(grid, plane, [9.5, 0, 0.5], wall_pts[(np.abs(wall_pts[:, 1]) < 1) & (wall_pts[:, 2] > -0.5) & (wall_pts[:, 2] < 1.5)])
i = int(np.argmin(cur['free']))
check('shift curve: the surface in front of the wall is free at 0, on the wall after moving it 0.5 m back',
      cur['free'][np.argmin(np.abs(cur['shifts']))] > 0.9 and cur['shifts'][i] >= 0.4 and cur['free'][i] < 0.1, (cur['shifts'], cur['free']))
check('shift curve: LiDAR returns are closest to the surface at +0.5', near(cur['shifts'][int(np.nanargmin(cur['lidar_dist']))], 0.5, 0.11), cur['lidar_dist'])
P, ends = ct.violating_points(grid, plane)
check('violating points and their beam ends (the wall)', len(P) > 0.9 * len(plane) and near(float(ends[:, 0].mean()), 10.0, 0.2), (len(P), ends[:, 0].mean()))
fr2 = NS(R_c2e=np.array([[0, 0, 1.], [-1, 0, 0], [0, -1, 0]]), t_c2e=np.zeros(3), K=np.array([[300., 0, 320], [0, 300., 240], [0, 0, 1]]))
Hh, Ww = 480, 640
full = np.ones((Hh, Ww), bool)
pcw = (fr2.R_c2e.T @ wall_pts.T).T
ret = np.stack([pcw[:, 0] / pcw[:, 2] * 300 + 320, pcw[:, 1] / pcw[:, 2] * 300 + 240, pcw[:, 2]], axis=1)
ret = ret[(ret[:, 0] > 0) & (ret[:, 0] < Ww) & (ret[:, 1] > 0) & (ret[:, 1] < Hh)]
c1 = ct.classify_violations(P, ends, fr2, full, ret, vol_pts=fs.box_lattice([10.2, 0, 0.5], 0.6, 2.0, 2.0, 0.0))
check('a point in front of the in-mask depth is IMPLIED by that depth; its beam ends inside the given volume',
      c1['red_implied'] > 0.9 and c1['red_end_in_mesh'] > 0.9, c1)
c2 = ct.classify_violations(P, ends, fr2, np.zeros((Hh, Ww), bool), ret)
check('outside the mask nothing is implied', c2['red_outside_mask'] > 0.9 and c2['red_implied'] == 0, c2)
c3 = ct.classify_violations(P, ends, fr2, full, ret[:0])
check('without any in-mask return the points are in_mask_no_return_near', c3['red_in_mask_no_return_near'] > 0.9, c3)

# ── SAM3D shift freeze (mask perturbation with the anchor held fixed) ──
import torch
from autolabeling.models.sam3d_objects import SAM3DObjectsModel
from sam3d_objects.data.dataset.tdfy.img_and_mask_transforms import ObjectCentricSSI

norm = ObjectCentricSSI(use_scene_scale=True, allow_scale_and_shift_override=True)
model = SAM3DObjectsModel.__new__(SAM3DObjectsModel)
model._frozen_ssi_mask = None
model._inference = NS(_pipeline=NS(ss_preprocessor=NS(pointmap_normalizer=norm, rgb_pointmap_normalizer=norm)))
model._install_ssi_freeze()
model._install_ssi_freeze()                                   # installing twice must not stack
pm = torch.zeros(3, 40, 40)
pm[2] = torch.linspace(5, 15, 40)[:, None].expand(40, 40)     # depth grows with the row
m_orig = torch.zeros(1, 40, 40); m_orig[:, 5:15, 5:15] = 1    # rows 5-14: depth about 6.3
m_new = torch.zeros(1, 40, 40); m_new[:, 25:35, 5:15] = 1     # a different mask: depth about 12.7
free_shift = norm.normalize(pm, m_new).shift
model._frozen_ssi_mask = m_orig[0].numpy()
frozen_shift = norm.normalize(pm, m_new).shift
orig_shift = norm.normalize(pm, m_orig).shift
model._frozen_ssi_mask = None
check('without freezing the shift follows the mask', abs(float(free_shift[2]) - 12.7) < 0.5, free_shift)
check('frozen: the shift comes from the original mask although another mask is passed', torch.allclose(frozen_shift, orig_shift), (frozen_shift, orig_shift))
check('unfreezing restores the normal behaviour', abs(float(norm.normalize(pm, m_new).shift[2]) - 12.7) < 0.5)
print(f'\nall {n} checks passed')
