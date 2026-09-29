"""
CPU-only checks for the clean pipeline's pure-logic parts (no GPU, no model weights).

Run inside the project container from the repo root:
    bash container/run_in_container.sh python tests/test_clean_pipeline.py

Covers: per-sweep TerraSeg-then-aggregate loader, the ego-body filter, pointmap-mode
resolution, the modes 1-4 pointmap construction in SAM3DObjectsModel.run_frame (models
mocked), the fallback ladder labels, and the NaN-outside-mask fix (mode 3).
"""
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS, SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from autolabeling.utils.lidar import (
    filter_lidar_pts, load_lidar_pts_aggregated, load_lidar_pts_nonground_aggregated,
)
from autolabeling.models.sam3d_objects import SAM3DObjectsModel, resolve_pointmap_mode
from fakes import FakeGroundFilter, FakeNusc

_passed = []


def check(name, cond, detail=''):
    assert cond, f'FAILED: {name} {detail}'
    _passed.append(name)
    print(f'  ok  {name}')


# ══════════════════════════════════════════════════════════════════════════════
# LiDAR loaders
# ══════════════════════════════════════════════════════════════════════════════
print('LiDAR loaders')
with tempfile.TemporaryDirectory() as d:
    nusc = FakeNusc(d)
    ANCH = 2
    frame = SimpleNamespace(
        lidar_sd_token=f'sd{ANCH}', R_e2g=np.eye(3), t_e2g=np.array([float(ANCH), 0, 0]),
        lidar_path=nusc.sd[f'sd{ANCH}']['filename'], R_l2e=np.eye(3), t_l2e=np.zeros(3))

    gf = FakeGroundFilter()
    ng = load_lidar_pts_nonground_aggregated(nusc, frame, 2, 2, gf)
    check('ground filter called once per sweep', len(gf.calls) == 5, len(gf.calls))
    # TerraSeg must see each sweep in ITS OWN ego frame: static object x = 10 - sweep_ego_x
    seen_x = sorted(float(c[c[:, 2] > 0.5][:, 0].min()) for c in gf.calls)
    check('TerraSeg input is in each sweep\'s own frame', np.allclose(seen_x, [6, 7, 8, 9, 10]), seen_x)
    check('ego-body point removed BEFORE TerraSeg (own frame)',
          all(not np.any(np.all(np.isclose(c, [0, 0, 1.5]), axis=1)) for c in gf.calls))
    check('output is ground-free', np.all(ng[:, 2] > 0.1))
    check('5 sweeps x 2 object points', len(ng) == 10, len(ng))
    uniq = np.unique(np.round(ng, 4), axis=0)
    check('static object collapses onto 2 points in the anchor frame (ego-motion compensated)',
          len(uniq) == 2 and np.allclose(sorted(uniq[:, 0]), [8.0, 8.2]), uniq)

    gf0 = FakeGroundFilter()
    ng0 = load_lidar_pts_nonground_aggregated(nusc, frame, 0, 0, gf0)
    check('n_before=n_after=0 -> single sweep only', len(gf0.calls) == 1 and len(ng0) == 2, len(ng0))
    check('single-sweep result is in the anchor frame', np.allclose(sorted(ng0[:, 0]), [8.0, 8.2]))

    full = load_lidar_pts_aggregated(nusc, frame, 2, 2)
    check('full aggregation keeps ground (4 world pts x 5 sweeps)', len(full) == 20, len(full))
    check('full aggregation: ego-body point removed in every path',
          not np.any(np.all(np.isclose(full, [0, 0, 1.5]), axis=1)))
    full0 = load_lidar_pts_aggregated(nusc, frame, 0, 0)
    check('single-sweep fallback of full loader is ego-filtered too', len(full0) == 4, len(full0))
    check('no lidar -> None', load_lidar_pts_nonground_aggregated(
        nusc, SimpleNamespace(lidar_sd_token=None, lidar_path=None, R_l2e=None), 2, 2, gf) is None)

e = filter_lidar_pts(np.array([[0, 0, 1.5], [10, 0, 1.5], [0, 0, 0.2], [0, 0, 3.0]]))
check('ego-body filter: box+z-band only', len(e) == 3 and not np.any(np.all(e == [0, 0, 1.5], axis=1)))
check('filter_lidar_pts has no range filter (far points survive)',
      len(filter_lidar_pts(np.array([[500.0, 500.0, 1.0]]))) == 1)


# ══════════════════════════════════════════════════════════════════════════════
# Pointmap modes
# ══════════════════════════════════════════════════════════════════════════════
print('pointmap mode names')
check('1..11 resolve', [resolve_pointmap_mode(i) for i in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)] ==
      ['sparse_lidar', 'moge_affine_local', 'moge_affine_local_masked', 'completionformer_full',
       'moge_affine_local_raw', 'moge_affine_local_piecewise', 'moge_affine_composite', 'moge_affine_local_regslope',
       'completionformer_raw', 'completionformer_ground', 'ldcm_full'])
check('numeric string resolves', resolve_pointmap_mode('3') == 'moge_affine_local_masked')
check('name resolves', resolve_pointmap_mode('sparse_lidar') == 'sparse_lidar')
for bad in (0, 12, 'o3_local_affine', 'baseline', True, None):
    try:
        resolve_pointmap_mode(bad)
        check(f'rejects {bad!r}', False)
    except ValueError:
        check(f'rejects {bad!r}', True)

H, W, FX, CX, CY = 60, 80, 50.0, 40.0, 30.0
K = np.array([[FX, 0, CX], [0, FX, CY], [0, 0, 1.0]])
MASK = np.zeros((H, W), bool)
MASK[20:40, 30:50] = True
A_TRUE, B_TRUE = 2.5, 3.0
HDB = {'car': dict(min_cluster_size=3, min_samples=1, cluster_eps=0.5)}


def zmoge_map():
    u = np.arange(W, dtype=np.float32)[None, :].repeat(H, 0)
    return (4.0 + 0.05 * u).astype(np.float32)


def cam_pts(px, depth):
    """3D points (camera frame == ego frame here: R_c2e = I, t_c2e = 0) at pixels px, given depth."""
    px, depth = np.asarray(px, float), np.asarray(depth, float)
    return np.stack([(px[:, 0] - CX) / FX * depth, (px[:, 1] - CY) / FX * depth, depth], 1)


def make_model(mode, erode=1, **kw):
    m = SAM3DObjectsModel('r', 'c', {'car': 'objects'}, device='cpu', pointmap_mode=mode,
                          hdbscan_params=HDB, mask_erode_px=erode, **kw)
    Zm = zmoge_map()
    m._compute_moge_pointmap = lambda img, K_: (
        np.stack([Zm * 0 + 1, Zm * 0 + 2, Zm], -1).astype(np.float32), Zm)
    seen = []
    m._inference = lambda img, mask, seed, pointmap: seen.append(pointmap.numpy().copy()) or {}
    m._mesh_to_r3 = lambda out: (np.zeros((3, 3), np.float32), np.zeros((1, 3), np.int32))
    return m, seen


FRAME = SimpleNamespace(
    scene_name='fake', frame_idx=0, K=K, R_c2e=np.eye(3), t_c2e=np.zeros(3), lidar_path='x',
    load_images=lambda: (np.zeros((H, W, 3), np.uint8), np.zeros((H, W, 3), np.uint8)))
SAM3 = {'car': [{'binary_mask': MASK, 'score': 0.9, 'prompt': 'car'}]}


def object_cluster(n_side=6, u0=36, v0=26):
    """36 tightly packed LiDAR points inside the mask; depth follows the TRUE affine of Z_moge."""
    uu, vv = np.meshgrid(np.arange(u0, u0 + n_side), np.arange(v0, v0 + n_side))
    px = np.stack([uu.ravel(), vv.ravel()], 1)
    depth = A_TRUE * zmoge_map()[px[:, 1], px[:, 0]] + B_TRUE
    return cam_pts(px, depth), px


def background_pts(n=20):
    rng = np.random.default_rng(0)
    px = np.stack([rng.integers(0, 25, n), rng.integers(0, H, n)], 1)      # all left of the mask
    return cam_pts(px, rng.uniform(30, 35, n))


obj_pts, obj_px = object_cluster()
PTS = np.vstack([obj_pts, background_pts()])

print('mode 2  moge_affine_local')
m2, seen2 = make_model(2)
r2 = m2.run_frame(FRAME, SAM3, pts_ego=PTS)
check('mode 2: one result, label local', len(r2) == 1 and r2[0]['o3_mode'] == 'local', r2 and r2[0]['o3_mode'])
p2 = seen2[0]
check('mode 2: pointmap finite everywhere (background gets a value)', np.isfinite(p2).all())
u0, v0 = obj_px[0]
zmog = zmoge_map()[v0, u0]
check('mode 2: recovered affine (Z = 2.5*Zmoge + 3.0) inside the mask',
      np.isclose(p2[v0, u0, 2], A_TRUE * zmog + B_TRUE, atol=1e-3), (p2[v0, u0, 2], A_TRUE * zmog + B_TRUE))
ur, vr = obj_px[np.argmax(obj_px[:, 0])]                       # right-most cluster pixel: u > cx
check('mode 2: R3->P3D convention (x negated: pixel right of cx -> negative x)',
      ur > CX and p2[vr, ur, 0] < 0, (ur, p2[vr, ur, 0]))

print('selection  only=')
m2s, seen2s = make_model(2)
SAM3_TWO = {'car': [SAM3['car'][0], dict(SAM3['car'][0])]}
rs = m2s.run_frame(FRAME, SAM3_TWO, pts_ego=PTS, only={('car', 1)})
check('only: exactly the selected detection is inferred', len(rs) == 1 and len(seen2s) == 1)
rs = m2s.run_frame(FRAME, SAM3_TWO, pts_ego=PTS, only=set())
check('only=empty set: nothing inferred', rs == [])
rs = m2s.run_frame(FRAME, SAM3_TWO, pts_ego=PTS)
check('only=None: all detections inferred', len(rs) == 2)

print('mode 5  moge_affine_local_raw')
ou, ov = int(obj_px[:, 0].min()) + 1, int(obj_px[:, 1].max()) + 3
OUTLIER = cam_pts(np.array([[ou, ov]]), np.array([70.0]))
PTS_OUT = np.vstack([PTS, OUTLIER])
m2o, seen2o = make_model(2)
r2o = m2o.run_frame(FRAME, SAM3, pts_ego=PTS_OUT)
check('mode 2 ignores an in-mask outlier (HDBSCAN drops it) -> label local', r2o[0]['o3_mode'] == 'local')
m5, seen5 = make_model(5)
r5 = m5.run_frame(FRAME, SAM3, pts_ego=PTS_OUT)
check('mode 5: one result, label local_raw (no HDBSCAN rung)', len(r5) == 1 and r5[0]['o3_mode'] == 'local_raw', r5 and r5[0]['o3_mode'])
check('mode 5: outlier contaminates the fit (pointmap differs from mode 2)',
      not np.allclose(seen5[0][v0, u0, 2], seen2o[0][v0, u0, 2], atol=1e-2))
r5c = m5.run_frame(FRAME, SAM3, pts_ego=PTS)
check('mode 5 on clean points matches mode 2 inside the mask',
      np.allclose(seen5[-1][MASK][:, 2], p2[MASK][:, 2], atol=1e-2))
check('mode 5: pointmap finite everywhere (no NaN masking)', np.isfinite(seen5[-1]).all())

print('mode 7  moge_affine_composite')
zm7 = zmoge_map()
BG_L, BG_R = (2.0, 3.0), (3.0, 0.0)                         # background truth: left / right half of the image
AB_A, AB_B = (2.5, 2.0), (2.0, 1.0)                         # object truths
mA = np.zeros((H, W), bool); mA[20:40, 10:25] = True
mB = np.zeros((H, W), bool); mB[20:40, 50:65] = True
def cluster_ab(u0, v0, ab, n=6):
    uu, vv = np.meshgrid(np.arange(u0, u0 + n), np.arange(v0, v0 + n))
    px = np.stack([uu.ravel(), vv.ravel()], 1)
    return cam_pts(px, ab[0] * zm7[px[:, 1], px[:, 0]] + ab[1])
def bg_cloud(masks, step=3):
    gu, gv = np.meshgrid(np.arange(1, W, step), np.arange(1, H, step))
    px = np.stack([gu.ravel(), gv.ravel()], 1)
    keep = ~np.any([m[px[:, 1], px[:, 0]] for m in masks], axis=0)
    px = px[keep]
    ab = np.where((px[:, 0] < W // 2)[:, None], np.array(BG_L), np.array(BG_R))
    return cam_pts(px, ab[:, 0] * zm7[px[:, 1], px[:, 0]] + ab[:, 1])
obj_cloud = np.vstack([cluster_ab(12, 26, AB_A), cluster_ab(53, 26, AB_B)])
full_cloud = np.vstack([obj_cloud, bg_cloud([mA, mB])])
SAM3_AB = {'car': [{'binary_mask': mA, 'score': .9, 'prompt': 'car'}, {'binary_mask': mB, 'score': .9, 'prompt': 'car'}]}
m7, seen7 = make_model(7, comp_grid=(2, 1), comp_min_pts=20, comp_margin=0.0, bg_feather_px=0)
r7 = m7.run_frame(FRAME, SAM3_AB, pts_ego=obj_cloud, pts_ego_full=full_cloud)
check('one pointmap is shared by both objects', len(seen7) == 2 and np.array_equal(seen7[0], seen7[1]))
check('both objects have their own fit (label local) and report it',
      [r['o3_mode'] for r in r7] == ['local', 'local'] and np.allclose(r7[0]['fit_ab'], AB_A, atol=1e-2) and np.allclose(r7[1]['fit_ab'], AB_B, atol=1e-2),
      [(r['o3_mode'], r['fit_ab']) for r in r7])
Z7 = seen7[0][..., 2]
check('inside mask A: A\'s own affine', np.isclose(Z7[30, 17], AB_A[0] * zm7[30, 17] + AB_A[1], atol=0.02), (Z7[30, 17], AB_A[0] * zm7[30, 17] + AB_A[1]))
check('inside mask B: B\'s own affine', np.isclose(Z7[30, 57], AB_B[0] * zm7[30, 57] + AB_B[1], atol=0.02))
check('background, left section: the LEFT background affine', np.isclose(Z7[5, 4], BG_L[0] * zm7[5, 4] + BG_L[1], atol=0.05), (Z7[5, 4], BG_L[0] * zm7[5, 4] + BG_L[1]))
check('background, right section: the RIGHT background affine', np.isclose(Z7[5, 76], BG_R[0] * zm7[5, 76] + BG_R[1], atol=0.05), (Z7[5, 76], BG_R[0] * zm7[5, 76] + BG_R[1]))
lo, hi = sorted([BG_L[0] * zm7[5, 40] + BG_L[1], BG_R[0] * zm7[5, 40] + BG_R[1]])
check('between the two section centres the field blends smoothly (no seam)', lo < Z7[5, 40] < hi, (lo, Z7[5, 40], hi))
check('no NaN anywhere (dense)', np.isfinite(seen7[0]).all())
# overlap: the NEARER object is painted on top
mC = np.zeros((H, W), bool); mC[20:40, 10:30] = True; mD = np.zeros((H, W), bool); mD[20:40, 20:40] = True
AB_C, AB_D = (2.5, 2.0), (2.5, 6.0)
ov_obj = np.vstack([cluster_ab(11, 26, AB_C), cluster_ab(33, 26, AB_D)])
m7b, seen7b = make_model(7, comp_grid=(2, 1), comp_min_pts=20, comp_margin=0.0, bg_feather_px=0)
m7b.run_frame(FRAME, {'car': [{'binary_mask': mD, 'score': .9, 'prompt': 'car'}, {'binary_mask': mC, 'score': .9, 'prompt': 'car'}]},
              pts_ego=ov_obj, pts_ego_full=np.vstack([ov_obj, bg_cloud([mC, mD])]))
check('overlap of two masks: the nearer object\'s affine wins', np.isclose(seen7b[0][30, 25, 2], AB_C[0] * zm7[30, 25] + AB_C[1], atol=0.02),
      (seen7b[0][30, 25, 2], AB_C[0] * zm7[30, 25] + AB_C[1], AB_D[0] * zm7[30, 25] + AB_D[1]))
# a detection without LiDAR takes the background calibration
mE = np.zeros((H, W), bool); mE[5:15, 60:75] = True
m7c, seen7c = make_model(7, comp_grid=(2, 1), comp_min_pts=20, comp_margin=0.0, bg_feather_px=0)
r7c = m7c.run_frame(FRAME, {'car': [{'binary_mask': mA, 'score': .9, 'prompt': 'car'}, {'binary_mask': mE, 'score': .9, 'prompt': 'car'}]},
                    pts_ego=cluster_ab(12, 26, AB_A), pts_ego_full=np.vstack([cluster_ab(12, 26, AB_A), bg_cloud([mA, mE])]))
check('a detection with no in-mask LiDAR is labelled bg_only and has no own fit', r7c[1]['o3_mode'] == 'bg_only' and r7c[1]['fit_ab'] is None, r7c[1]['o3_mode'])
check('...and its region carries the background calibration',
      np.isclose(seen7c[0][10, 70, 2], BG_R[0] * zm7[10, 70] + BG_R[1], atol=0.05), (seen7c[0][10, 70, 2], BG_R[0] * zm7[10, 70] + BG_R[1]))
m7d, seen7d = make_model(7)
r7d = m7d.run_frame(FRAME, SAM3_AB, pts_ego=None)
check('no LiDAR: unscaled MoGe for every object', all(r['o3_mode'] == 'unscaled_fallback' for r in r7d) and len(seen7d) == 2)

print('mode 8  moge_affine_local_regslope')
flat = cam_pts(np.stack(np.meshgrid(np.arange(12, 18), np.arange(26, 32)), -1).reshape(-1, 2), np.full(36, 12.0))   # constant depth: slope unidentifiable
cloud8 = np.vstack([flat, cluster_ab(53, 26, AB_B)])
full8 = np.vstack([cloud8, bg_cloud([mA, mB])])
m6c, seen6c = make_model(6, bg_pad_factor=2.8)      # the synthetic image is small: widen the local background region
r6c = m6c.run_frame(FRAME, SAM3_AB, pts_ego=cloud8, pts_ego_full=full8)
m8, seen8 = make_model(8, bg_pad_factor=2.8)
r8 = m8.run_frame(FRAME, SAM3_AB, pts_ego=cloud8, pts_ego_full=full8)
check('the flat object is regularised (label +reg), the sane one is not', r8[0]['o3_mode'].endswith('+reg') and not r8[1]['o3_mode'].endswith('+reg'), [r['o3_mode'] for r in r8])
check('its slope now comes from the background fit (~2.0 on the left), not ~0', abs(r8[0]['fit_ab'][0] - r8[0]['bg_ab'][0]) < 1e-6 and 1.5 < r8[0]['fit_ab'][0] < 2.6, r8[0]['fit_ab'])
d8 = seen8[0][30, 17, 2] - seen8[0][30, 13, 2]
check('inside its mask the depth now varies like MoGe scaled by that slope (was flat in mode 6)',
      abs(d8 - r8[0]['fit_ab'][0] * (zm7[30, 17] - zm7[30, 13])) < 1e-3 and abs(seen6c[0][30, 17, 2] - seen6c[0][30, 13, 2]) < 0.05,
      (d8, seen6c[0][30, 17, 2] - seen6c[0][30, 13, 2]))
check('the shift SAM3D derives is unchanged by the regularisation', abs(r8[0]['ssi_shift'][2] - r6c[0]['ssi_shift'][2]) < 0.15, (r8[0]['ssi_shift'][2], r6c[0]['ssi_shift'][2]))
check('the well-conditioned object gets exactly mode 6\'s pointmap', np.allclose(seen8[1], seen6c[1], atol=1e-5))

print('diagnostic fields')
m2d, seen2d = make_model(2)
r2d = m2d.run_frame(FRAME, SAM3_TWO if 'SAM3_TWO' in dir() else SAM3, pts_ego=PTS)
check('results carry the detection index', all('sam3_index' in r for r in r2d) and r2d[0]['sam3_index'] == 0)
check('results carry SAM3D scale/shift (or None if the package is missing)',
      all(('ssi_scale' in r and 'ssi_shift' in r) for r in r2d))
if r2d[0]['ssi_scale'] is not None:
    check('scale is positive and finite, shift is a 3-vector', r2d[0]['ssi_scale'] > 0 and np.isfinite(r2d[0]['ssi_scale']) and r2d[0]['ssi_shift'].shape == (3,))
    inmask_z = seen2d[0][MASK][:, 2]
    check('shift z = median depth inside the mask (torch takes the lower middle value of an even count)', abs(r2d[0]['ssi_shift'][2] - np.median(inmask_z)) < 0.1, (r2d[0]['ssi_shift'][2], np.median(inmask_z)))

print('OBB from the full mesh, then slim')
from autolabeling.utils.meshes import attach_full_obbs, slim_object_results, FULL_MESH_CLASSES
from autolabeling.fitting.obb import compute_obb_gravity_aligned
rng_m = np.random.default_rng(3)
body_pts = rng_m.normal(0, 1.0, (20000, 3)) * np.array([1.0, 0.4, 0.3])          # a dense blob, ego-frame-like
outliers = np.array([[6.0, 0, 0], [-6.0, 0, 0]])                                   # 2 extreme points a subsample will miss
full_v = np.vstack([body_pts, outliers]).astype(np.float32)
R_id = np.eye(3); t_id = np.zeros(3)
fr_id = NS(R_c2e=R_id, t_c2e=t_id)
def mk(prompt): return [{'prompt': prompt, 'vertices': full_v.copy(), 'faces': np.zeros((5, 3), np.int32)}]
objs_c = attach_full_obbs(mk('car'), fr_id)
_, _, d_full, y_full = compute_obb_gravity_aligned(full_v, R_id, t_id, ground_z=None)
check('attach_full_obbs stores the box fitted on ALL vertices', np.allclose(objs_c[0]['obb_raw']['dims'], d_full) and abs(objs_c[0]['obb_raw']['yaw'] - y_full) < 1e-9)
slim_object_results(objs_c, 2000)
_, _, d_slim, _ = compute_obb_gravity_aligned(objs_c[0]['vertices'], R_id, t_id, ground_z=None)
check('after slimming the vertices are reduced and faces dropped', len(objs_c[0]['vertices']) == 2000 and objs_c[0]['faces'] is None)
check('a refit on the slim vertices WOULD differ (the 2 extreme points are gone) -- the stored box does not',
      d_slim[0] < d_full[0] - 1.0 and np.allclose(objs_c[0]['obb_raw']['dims'], d_full))
objs_b = slim_object_results(mk('bicycle'), 2000)
check('bicycles/motorcycles keep ALL vertices (their box is refitted with the rider), faces dropped',
      len(objs_b[0]['vertices']) == len(full_v) and objs_b[0]['faces'] is None and 'bicycle' in FULL_MESH_CLASSES)
import pickle
back = pickle.loads(pickle.dumps(objs_c))
check('the stored box survives a checkpoint round trip', np.allclose(back[0]['obb_raw']['corners'], objs_c[0]['obb_raw']['corners']))

print('token-drop probe toggle')
_fuser = NS(embedder_list=[(None, [('image', 'cropped'), ('rgb_image', 'full')]), (None, [('mask', 'cropped'), ('rgb_image_mask', 'full')]),
                           (None, [('pointmap', 'cropped'), ('rgb_pointmap', 'full')])], force_drop_modalities=None)
mp, _ = make_model(2)
mp._inference = NS(_pipeline=NS(condition_embedders={'ss_condition_embedder': _fuser, 'slat_condition_embedder': NS(embedder_list=[], force_drop_modalities=None)}))
mp.set_force_drop(['pointmap', 'rgb_pointmap'])
check('sets the drop list on the STAGE-1 embedder', _fuser.force_drop_modalities == ['pointmap', 'rgb_pointmap'])
check('leaves the stage-2 embedder alone', mp._inference._pipeline.condition_embedders['slat_condition_embedder'].force_drop_modalities is None)
mp.set_force_drop(None)
check('None restores normal behaviour', _fuser.force_drop_modalities is None)
mp.set_force_drop([])
check('empty list also restores normal behaviour', _fuser.force_drop_modalities is None)
try:
    mp.set_force_drop(['pointmapp'])
    check('a misspelled input name is rejected', False)
except ValueError as e:
    check('a misspelled input name is rejected', 'pointmapp' in str(e))
check('...and nothing was changed by the failed call', _fuser.force_drop_modalities is None)

print('mode 6  moge_affine_local_piecewise')
A_BG, B_BG = 1.2, 10.0
bu, bv = np.meshgrid(np.r_[8:24:2, 58:74:2], np.arange(6, 56, 4))
bpx = np.stack([bu.ravel(), bv.ravel()], 1)
bdepth = A_BG * zmoge_map()[bpx[:, 1], bpx[:, 0]] + B_BG
BG = cam_pts(bpx, bdepth)
BG_BAD = cam_pts(bpx[:4], bdepth[:4] + 15.0)              # a few wrong returns (other surfaces)
FULL = np.vstack([obj_pts, BG, BG_BAD])
m6, seen6 = make_model(6, bg_pad_factor=3.0, bg_feather_px=0)
r6 = m6.run_frame(FRAME, SAM3, pts_ego=PTS, pts_ego_full=FULL)
p6 = seen6[0]
check('mode 6: label = object ladder + local background', r6[0]['o3_mode'] == 'local+bg_local', r6[0]['o3_mode'])
check('mode 6: pointmap finite everywhere', np.isfinite(p6).all())
zm = zmoge_map()
check('mode 6: inside the mask the object affine', np.isclose(p6[v0, u0, 2], A_TRUE * zm[v0, u0] + B_TRUE, atol=1e-3))
vb0, ub0 = 30, 12
check('mode 6: outside the mask the BACKGROUND affine (robust to the wrong returns)',
      np.isclose(p6[vb0, ub0, 2], A_BG * zm[vb0, ub0] + B_BG, atol=0.05), (p6[vb0, ub0, 2], A_BG * zm[vb0, ub0] + B_BG))
check('mode 6: differs from mode 2 outside the mask, equal inside',
      not np.isclose(p6[vb0, ub0, 2], p2[vb0, ub0, 2], atol=0.5) and np.allclose(p6[MASK], p2[MASK], atol=1e-3))
m6o, seen6o = make_model(6)
r6o = m6o.run_frame(FRAME, SAM3, pts_ego=obj_pts, pts_ego_full=obj_pts)
check('mode 6 without any background LiDAR: label bg_none, identical to mode 2',
      r6o[0]['o3_mode'] == 'local+bg_none' and np.allclose(seen6o[0], p2, atol=1e-3), r6o[0]['o3_mode'])
m6f, seen6f = make_model(6, bg_pad_factor=3.0, bg_feather_px=3)
m6f.run_frame(FRAME, SAM3, pts_ego=PTS, pts_ego_full=FULL)
check('mode 6: feather only changes pixels near the seam',
      np.allclose(seen6f[0][:, :20], p6[:, :20], atol=1e-3) and not np.allclose(seen6f[0], p6))
m6g, seen6g = make_model(6, bg_pad_factor=1.5)
r6g = m6g.run_frame(FRAME, SAM3, pts_ego=PTS, pts_ego_full=FULL)
check('mode 6: no points in the padded box -> whole-image background fit', r6g[0]['o3_mode'] == 'local+bg_image', r6g[0]['o3_mode'])

print('mode 3  moge_affine_local_masked')
m3, seen3 = make_model(3)
r3 = m3.run_frame(FRAME, SAM3, pts_ego=PTS)
p3 = seen3[0]
check('mode 3: label unchanged (local)', r3[0]['o3_mode'] == 'local')
check('mode 3: NaN outside the ORIGINAL (un-eroded) mask, all 3 channels',
      np.isnan(p3[~MASK]).all())
check('mode 3: finite everywhere inside the mask incl. its border pixels',
      np.isfinite(p3[MASK]).all())
check('mode 3: inside values identical to mode 2', np.allclose(p3[MASK], p2[MASK]))

print('fallback ladder (mode 2)')
# local_raw: 5 points in the mask but > 10 m apart in depth -> HDBSCAN finds no cluster
px_raw = np.array([[32, 25], [36, 27], [40, 29], [44, 31], [47, 33]])
raw_pts = cam_pts(px_raw, np.array([5, 15, 25, 35, 45.0]))
m, seen = make_model(2)
r = m.run_frame(FRAME, SAM3, pts_ego=raw_pts)
check('spread-out in-mask points -> local_raw', r[0]['o3_mode'] == 'local_raw', r[0]['o3_mode'])
# global_fallback: LiDAR exists but none inside the mask
m, seen = make_model(2)
r = m.run_frame(FRAME, SAM3, pts_ego=background_pts(40))
check('LiDAR only outside the mask -> global_fallback', r[0]['o3_mode'] == 'global_fallback', r[0]['o3_mode'])
# unscaled_fallback: no LiDAR at all
m, seen = make_model(2)
r = m.run_frame(FRAME, SAM3, pts_ego=None)
check('no LiDAR -> unscaled_fallback', r[0]['o3_mode'] == 'unscaled_fallback', r[0]['o3_mode'])
check('  ... and uses the raw MoGe pointmap', np.allclose(seen[0][..., 2], zmoge_map()))
# mode 3 keeps the NaN mask even on the unscaled fallback
m, seen = make_model(3)
r = m.run_frame(FRAME, SAM3, pts_ego=None)
check('mode 3 unscaled_fallback still NaN-masked', np.isnan(seen[0][~MASK]).all() and np.isfinite(seen[0][MASK]).all())
# too few points even globally
m, seen = make_model(2)
r = m.run_frame(FRAME, SAM3, pts_ego=background_pts(2))
check('< min_affine_pts LiDAR anywhere -> unscaled_fallback', r[0]['o3_mode'] == 'unscaled_fallback')

print('mode 1  sparse_lidar')
m1, seen1 = make_model(1)
r1 = m1.run_frame(FRAME, SAM3, pts_ego=PTS)
p1 = seen1[0]
check('mode 1: label is the mode name', r1[0]['o3_mode'] == 'sparse_lidar')
n_valid = int(np.isfinite(p1[..., 2]).sum())
check('mode 1: sparse -- finite only where LiDAR landed', 0 < n_valid <= len(PTS), n_valid)
check('mode 1: metric depth kept at the LiDAR pixel', np.isclose(p1[v0, u0, 2], obj_pts[0, 2], atol=1e-3))
check('mode 1: no MoGe call needed (pointmap not dense)', n_valid < H * W // 4)

print('mode 4  completionformer_full')
m4, seen4 = make_model(4)
captured = {}


def fake_dense(img, K_, u, v, Z):
    captured.update(u=u.copy(), v=v.copy(), Z=Z.copy())
    dense = np.full((H, W), 12.0, np.float32)
    return m4._depth_to_ptmap(dense, K_, H, W), dense


m4._compute_dense_completion_pointmap = fake_dense
stray = cam_pts(np.array([[33, 22], [45, 35], [38, 30], [47, 22]]), np.array([30, 40, 50, 60.0]))
PTS4 = np.vstack([obj_pts, stray, background_pts()])
r4 = m4.run_frame(FRAME, SAM3, pts_ego=PTS4)
check('mode 4: label is the mode name', r4[0]['o3_mode'] == 'completionformer_full')
check('mode 4: one shared dense pointmap fed to inference', np.isfinite(seen4[0]).all() and np.allclose(seen4[0][..., 2], 12.0))
in_mask_after = MASK[np.round(captured['v']).astype(int), np.round(captured['u']).astype(int)]
kept_Z = captured['Z'][in_mask_after]
check('mode 4: far in-mask strays (depth 30-60 m) removed; cluster (~18 m) kept', bool(np.all(kept_Z < 25)), kept_Z.max())
check('mode 4: dominant near cluster survives (HDBSCAN may drop a few edge points)',
      len(kept_Z) >= 0.75 * len(obj_pts), len(kept_Z))
check('mode 4: out-of-mask returns kept unchanged as anchors', int((~in_mask_after).sum()) == len(background_pts()))

print('mode 10  completionformer_ground (hybrid: ground-free in-mask, full-cloud background)')
# a background cloud clearly distinguishable from PTS4's own background (depth 30-35, cols 0-25):
full_bg = cam_pts(np.stack([np.arange(55, 75), np.full(20, 10)], 1), np.full(20, 72.0))


def _hybrid_check(mode_no, mode_name, method_name):
    m, seen = make_model(mode_no)
    captured = {}

    def fake(img, K_, u, v, Z):
        captured.update(u=u.copy(), v=v.copy(), Z=Z.copy())
        dense = np.full((H, W), 12.0, np.float32)
        return m._depth_to_ptmap(dense, K_, H, W), dense

    setattr(m, method_name, fake)
    r = m.run_frame(FRAME, SAM3, pts_ego=PTS4, pts_ego_full=full_bg)
    check(f'{mode_name}: label is the mode name', r[0]['o3_mode'] == mode_name)
    in_mask_after = MASK[np.round(captured['v']).astype(int), np.round(captured['u']).astype(int)]
    kept_Z, bg_Z = captured['Z'][in_mask_after], captured['Z'][~in_mask_after]
    check(f'{mode_name}: in-mask cleaning matches mode 4 (far strays removed, cluster kept)',
          len(kept_Z) >= 0.75 * len(obj_pts) and bool(np.all(kept_Z < 25)), (len(kept_Z), kept_Z.max() if len(kept_Z) else None))
    check(f"{mode_name}: background comes from pts_ego_full, NOT PTS4's own background (depth 30-35)",
          len(bg_Z) == len(full_bg) and bool(np.all(bg_Z > 60)), (len(bg_Z), bg_Z.min() if len(bg_Z) else None))


_hybrid_check(10, 'completionformer_ground', '_compute_dense_completion_pointmap')

print('mode 11  ldcm_full (same hybrid anchors as mode 10, fused by a different network)')
_hybrid_check(11, 'ldcm_full', '_compute_dense_ldcm_pointmap')

print(f'\nall {len(_passed)} checks passed')
