"""
CPU-only end-to-end check of pipeline.py's orchestration with every model mocked.

    bash container/run_in_container.sh python tests/test_pipeline_orchestration.py

Verifies stage wiring (which LiDAR cloud goes where), caching/resume behaviour, config
hashing (a changed mode/aggregation never resumes stale results), the removal of all OBB
filters, and multi-camera sharing of the ground-free cloud.
"""
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import autolabeling.pipeline as P
from fakes import FakeGroundFilter, FakeNusc

_n = []


def check(name, cond, detail=''):
    assert cond, f'FAILED: {name} {detail}'
    _n.append(name)
    print(f'  ok  {name}')


# ── spies / mocks installed into the pipeline module ─────────────────────────────
LOG = {}


def reset():
    LOG.clear()
    LOG.update(seg_loads=0, seg_frames=0, body_loads=0, body_frames=0, obj_loads=0, obj_frames=0,
               obj_modes=[], obj_full_lens=[], obj_cloud_lens=[], obj_cloud_ground=[], ts_loads=0, ts_sweeps=0,
               b1_cloud=[], b2_cloud=[], pl_fits=0)


class FakeSeg:
    def __init__(self, **kw): pass
    def load(self): LOG['seg_loads'] += 1
    def unload(self): pass

    def run_frame(self, frame):
        LOG['seg_frames'] += 1
        m = np.zeros((480, 640), bool); m[150:210, 170:220] = True
        return {'pedestrian': [{'binary_mask': m, 'score': 0.9, 'prompt': 'pedestrian'}], 'car': []}


class FakeBody:
    def __init__(self, **kw): pass
    def load(self): LOG['body_loads'] += 1
    def unload(self): pass

    def run_frame(self, frame, ped_dets):
        LOG['body_frames'] += 1
        rng = np.random.default_rng(1)
        verts = (rng.uniform(-1, 1, (30, 3)) * [0.25, 0.85, 0.2] + [-1.0, 0.0, 12.0]).astype(np.float32)
        joints = np.zeros((70, 3), np.float32)
        joints[0] = [0, 0, 0.2]; joints[5] = [-0.2, 0, 0]; joints[6] = [0.2, 0, 0]
        return [{'vertices': verts, 'faces': np.zeros((5, 3), np.int32), 'joints_3d': joints,
                 'cam_t': np.array([-1.0, 0.0, 12.0], np.float32), 'bbox': np.array([170, 150, 220, 210.]),
                 'score': 1.0, 'sam3_mask_idx': 0, 'binary_mask': ped_dets[0]['binary_mask']}]


def cube(center, s):
    c = np.array(center, np.float32)
    return np.array([c + [dx, dy, dz] for dx in (-s, s) for dy in (-s, s) for dz in (-s, s)], np.float32)


class FakeObjects:
    def __init__(self, **kw): self.mode = kw['pointmap_mode']
    def load(self): LOG['obj_loads'] += 1
    def unload(self): pass

    def run_frame(self, frame, frame_sam3, pts_ego=None, pts_ego_full=None):
        LOG['obj_frames'] += 1
        LOG['obj_full_lens'].append(None if pts_ego_full is None else len(pts_ego_full))
        LOG['obj_modes'].append(self.mode)
        LOG['obj_cloud_lens'].append(None if pts_ego is None else len(pts_ego))
        LOG['obj_cloud_ground'].append(None if pts_ego is None else bool(np.any(pts_ego[:, 2] <= 0.1)))
        m = np.zeros((480, 640), bool); m[100:200, 300:400] = True
        f = np.zeros((12, 3), np.int32)
        return [
            {'vertices': cube([-2, -1, 9], 0.8), 'faces': f, 'binary_mask': m, 'score': .9, 'prompt': 'car', 'o3_mode': 'local'},
            # things the old OBB filters existed to drop -- must now survive:
            {'vertices': cube([0, 0, 15], 0.005), 'faces': f, 'binary_mask': m, 'score': .9, 'prompt': 'car', 'o3_mode': 'local_raw'},   # degenerate size
            {'vertices': cube([0, 0, 0.4], 0.2), 'faces': f, 'binary_mask': m, 'score': .9, 'prompt': 'car', 'o3_mode': 'unscaled_fallback'},  # 0.4 m from camera ("hood")
        ]


class CountingGF(FakeGroundFilter):
    def __init__(self, *a, **k):
        super().__init__()
        LOG['ts_loads'] += 1

    def segment(self, pts):
        LOG['ts_sweeps'] += 1
        return super().segment(pts)

    def unload(self): pass


def _spy(name):
    orig = getattr(P, name)

    def spy(*a, **k):
        # (body_results, frame, ped_dets, pts_ego_vis, ...) for B1; (body_results, frame, pts_ego, ...) for B2
        LOG['b1_cloud' if name == '_apply_b1_depth_correction' else 'b2_cloud'].append(
            len(a[3]) if name == '_apply_b1_depth_correction' else len(a[2]))
        if name == '_apply_b1_depth_correction':
            return orig(*a, **k)
    return spy


P.SAM3Segmentor, P.SAM3DBodyModel, P.SAM3DObjectsModel, P.TerraSegGroundFilter = FakeSeg, FakeBody, FakeObjects, CountingGF
P._fit_pseudolabeler = lambda pts, device, dev_root: (LOG.__setitem__('pl_fits', LOG['pl_fits'] + 1) or torch.nn.Linear(1, 1))
P._restore_pseudolabeler = lambda state, device, dev_root: None
P._apply_b1_depth_correction = _spy('_apply_b1_depth_correction')
P._apply_b2_ground_anchoring = _spy('_apply_b2_ground_anchoring')


# ── config / frames ─────────────────────────────────────────────────────────────
def make_cfg(mode=2, agg=True, b1=True, b2=True, mesh_points=None):
    cfg = NS(
        models=NS(dev_root='/nonexistent', sam3_ckpt='x', sam3d_body_repo='br', sam3d_obj_cfg='/a/b/c/pipeline.yaml'),
        thresholds=NS(sam3_score=.5, min_mask_px=100, body_bbox=.8, sam3_iou_merge=.4),
        prompts=NS(pedestrian='body', car='objects'),
        cross_class_dedup=NS(iou_thresh=.5, priority=NS(pedestrian=0, car=1)),
        sam3d_objects=NS(pointmap_mode=mode, cformer_ckpt=None, lidar_lines=32, mask_erode_px=3,
                         proximity_min_pts=30, proximity_ratio=.7,
                         hdbscan=NS(pedestrian=NS(min_cluster_size=3, min_samples=1, cluster_eps=.2),
                                    car=NS(min_cluster_size=3, min_samples=1, cluster_eps=.5))),
        lidar_filters=NS(use_ego_body_filter=True, ego_box_half_x=4., ego_box_half_y=1.5, ego_box_z_min=.5, ego_box_z_max=2.5),
        lidar_aggregation=NS(use_aggregation=agg, n_before=2, n_after=2),
        ground_removal=NS(terraseg_ckpt=None),
        sam3d_body=NS(b1_depth_correction=b1, b2_ground_anchoring=b2),
        cross_camera_merge=NS(enabled=False),
    )
    if mesh_points is not None:
        cfg.sam3d_objects.mesh_points = mesh_points
    return cfg


def make_frames(nusc, tmp, cams=('CAM_FRONT',)):
    anch = 2
    img = Path(tmp) / 'img.png'
    Image.fromarray(np.zeros((480, 640, 3), np.uint8)).save(img)
    R_c2e = np.array([[0, 0, 1.], [-1, 0, 0], [0, -1, 0]])     # cam (x right,y down,z fwd) -> ego (x fwd,y left,z up)
    K = np.array([[500., 0, 320], [0, 500., 240], [0, 0, 1]])
    out = {}
    for cam in cams:
        out[cam] = [NS(
            sample_token='tok0', img_path=str(img), K=K, R_c2e=R_c2e, t_c2e=np.zeros(3),
            R_e2g=np.eye(3), t_e2g=np.array([float(anch), 0, 0]), scene_name='scene', frame_idx=0,
            camera_name=cam, img_width=640, img_height=480,
            lidar_path=nusc.sd[f'sd{anch}']['filename'], R_l2e=np.eye(3), t_l2e=np.zeros(3),
            lidar_sd_token=f'sd{anch}',
            load_images=lambda: (np.zeros((480, 640, 3), np.uint8),) * 2)]
    return out


with tempfile.TemporaryDirectory() as tmp:
    nusc = FakeNusc(tmp)
    frames = make_frames(nusc, tmp)['CAM_FRONT']
    ckpt = Path(tmp) / 'ckpt'
    cache = Path(tmp) / 'lidar_cache'

    print('run 1: mode 2, aggregation ±2, B1+B2')
    reset()
    body, objs = P.run_pipeline(make_cfg(mode=2), frames, device='cpu', checkpoint_dir=ckpt, nusc=nusc, lidar_cache_dir=cache)
    check('TerraSeg loaded once, ran once per sweep (5 sweeps)', LOG['ts_loads'] == 1 and LOG['ts_sweeps'] == 5, (LOG['ts_loads'], LOG['ts_sweeps']))
    check('ground-free cloud cached per keyframe token', len(list(cache.rglob('tok0.npy'))) == 1)
    check('objects got the GROUND-FREE cloud (10 pts, none at ground level)',
          LOG['obj_cloud_lens'] == [10] and LOG['obj_cloud_ground'] == [False], (LOG['obj_cloud_lens'], LOG['obj_cloud_ground']))
    check('B1 got a projection of the ground-free cloud (<= 10 visible pts)', len(LOG['b1_cloud']) == 1 and 0 < LOG['b1_cloud'][0] <= 10, LOG['b1_cloud'])
    check('B2 got the FULL cloud incl. ground (20 pts)', LOG['b2_cloud'] == [20], LOG['b2_cloud'])
    check('PseudoLabeler fitted once', LOG['pl_fits'] == 1)
    check('mode reaches SAM3DObjectsModel as its canonical name', LOG['obj_modes'] == ['moge_affine_local'], LOG['obj_modes'])
    check('OBB fitted for pedestrian + all 3 objects', len(body[0]) == 1 and len(objs[0]) == 3 and all('obb_corners' in r for r in objs[0]))
    check('NO OBB filter: degenerate (1 cm) box and 0.4 m-from-camera "hood" box both survive',
          sorted(r['o3_mode'] for r in objs[0]) == ['local', 'local_raw', 'unscaled_fallback'])
    check('velocity/dynamic fields present and zero', all(r['velocity_mps'] == 0.0 and r['is_dynamic'] is False for r in body[0] + objs[0]))
    dirs = sorted(p.name for p in ckpt.iterdir())
    check('stage dirs are hash-named', any(d.startswith('sam3__') for d in dirs) and any(d.startswith('body__') for d in dirs)
          and any(d.startswith('objects__moge_affine_local__') for d in dirs), dirs)

    print('run 2: identical config -> everything resumes from checkpoints')
    reset()
    body2, objs2 = P.run_pipeline(make_cfg(mode=2), frames, device='cpu', checkpoint_dir=ckpt, nusc=nusc, lidar_cache_dir=cache)
    check('no model loaded, nothing recomputed',
          LOG['seg_loads'] == LOG['body_loads'] == LOG['obj_loads'] == LOG['ts_loads'] == 0, dict(LOG))
    check('restored objects keep their binary_mask (cross-camera merge needs it)', all(r.get('binary_mask') is not None for r in objs2[0]))
    check('restored objects keep their o3_mode label', sorted(r['o3_mode'] for r in objs2[0]) == ['local', 'local_raw', 'unscaled_fallback'])
    check('restored pedestrian keeps its binary_mask', body2[0][0].get('binary_mask') is not None)
    check('restored OBBs identical', np.allclose(objs2[0][0]['obb_corners'], objs[0][0]['obb_corners']))

    print('run 3: only the pointmap mode changed (2 -> 3)')
    reset()
    P.run_pipeline(make_cfg(mode=3), frames, device='cpu', checkpoint_dir=ckpt, nusc=nusc, lidar_cache_dir=cache)
    check('objects recomputed under the new mode (NOT resumed from mode 2)', LOG['obj_frames'] == 1 and LOG['obj_modes'] == ['moge_affine_local_masked'], LOG['obj_modes'])
    check('SAM3, body and the ground-free cloud reused', LOG['seg_loads'] == LOG['body_loads'] == LOG['ts_loads'] == 0, dict(LOG))
    dirs = sorted(p.name for p in ckpt.iterdir())
    check('separate objects dirs per mode', sum(d.startswith('objects__') for d in dirs) == 2, dirs)
    reset()
    P.run_pipeline(make_cfg(mode=2), frames, device='cpu', checkpoint_dir=ckpt, nusc=nusc, lidar_cache_dir=cache)
    check('modes other than 6 are not handed the full cloud', LOG['obj_full_lens'] in ([], [None]), LOG['obj_full_lens'])
    reset()
    P.run_pipeline(make_cfg(mode=6), frames, device='cpu', checkpoint_dir=ckpt, nusc=nusc, lidar_cache_dir=cache)
    check('mode 6 gets the FULL cloud (ground included, 20 pts) next to the ground-free one (10)',
          LOG['obj_modes'] == ['moge_affine_local_piecewise'] and LOG['obj_full_lens'] == [20] and LOG['obj_cloud_lens'] == [10],
          (LOG['obj_modes'], LOG['obj_full_lens'], LOG['obj_cloud_lens']))
    for _m, _name in ((7, 'moge_affine_composite'), (8, 'moge_affine_local_regslope')):
        reset()
        P.run_pipeline(make_cfg(mode=_m), frames, device='cpu', checkpoint_dir=ckpt, nusc=nusc, lidar_cache_dir=cache)
        check(f'mode {_m} ({_name}) gets the FULL cloud next to the ground-free one',
              LOG['obj_modes'] == [_name] and LOG['obj_full_lens'] == [20] and LOG['obj_cloud_lens'] == [10],
              (LOG['obj_modes'], LOG['obj_full_lens'], LOG['obj_cloud_lens']))
    _rt = P._obj_from_ckpt(P._obj_to_ckpt([{'vertices': np.zeros((3, 3), np.float32), 'faces': None, 'score': .9, 'prompt': 'car',
                                             'o3_mode': 'local', 'binary_mask': np.zeros((4, 4), bool), 'sam3_index': 2,
                                             'ssi_scale': 6.5, 'ssi_shift': np.array([1., 2., 3.]), 'fit_ab': (2.5, 1.0), 'bg_ab': (2.0, 3.0)}]))[0]
    check('sam3_index, SAM3D scale/shift and the affines survive a checkpoint round trip',
          _rt['sam3_index'] == 2 and _rt['ssi_scale'] == 6.5 and np.allclose(_rt['ssi_shift'], [1, 2, 3]) and _rt['fit_ab'] == (2.5, 1.0) and _rt['bg_ab'] == (2.0, 3.0))
    _old = P._obj_from_ckpt([{'vertices': np.zeros((3, 3), np.float32), 'faces': None, 'score': .9, 'prompt': 'car'}])[0]
    check('old checkpoints without those fields still load', _old['sam3_index'] is None and _old['fit_ab'] is None)
    reset()
    P.run_pipeline(make_cfg(mode='1'), frames, device='cpu', checkpoint_dir=ckpt, nusc=nusc, lidar_cache_dir=cache)
    check("mode given as numeric string '1' -> sparse_lidar", LOG['obj_modes'] == ['sparse_lidar'])

    print('run 4: aggregation switched off')
    reset()
    P.run_pipeline(make_cfg(mode=2, agg=False), frames, device='cpu', checkpoint_dir=ckpt, nusc=nusc, lidar_cache_dir=cache)
    check('everything downstream of LiDAR recomputed (new hash), TerraSeg sees ONE sweep',
          LOG['ts_loads'] == 1 and LOG['ts_sweeps'] == 1 and LOG['body_frames'] == 1 and LOG['obj_frames'] == 1, dict(LOG))
    check('objects got the single-sweep ground-free cloud (2 pts)', LOG['obj_cloud_lens'] == [2], LOG['obj_cloud_lens'])
    check('B2 got the single-sweep FULL cloud (4 pts)', LOG['b2_cloud'] == [4], LOG['b2_cloud'])

    print('B1 / B2 are independent')
    reset()
    P.run_pipeline(make_cfg(mode=2, b1=False, b2=True), frames, device='cpu', checkpoint_dir=Path(tmp) / 'c_b2only', nusc=nusc, lidar_cache_dir=cache)
    check('B2 only: B1 never called, B2 called', LOG['b1_cloud'] == [] and len(LOG['b2_cloud']) == 1)
    reset()
    P.run_pipeline(make_cfg(mode=2, b1=True, b2=False), frames, device='cpu', checkpoint_dir=Path(tmp) / 'c_b1only', nusc=nusc, lidar_cache_dir=cache)
    check('B1 only: B2 and PseudoLabeler never run', LOG['b2_cloud'] == [] and LOG['pl_fits'] == 0 and len(LOG['b1_cloud']) == 1)

    print('mesh slimming (sam3d_objects.mesh_points)')
    ckpt_m = Path(tmp) / 'ckpt_mesh'
    reset()
    _, o_full = P.run_pipeline(make_cfg(mode=2), frames, device='cpu', checkpoint_dir=ckpt_m, nusc=nusc, lidar_cache_dir=cache)
    check('without mesh_points the full mesh and faces are kept', len(o_full[0][0]['vertices']) == 8 and o_full[0][0]['faces'] is not None)
    dirs_before = sorted(d.name for d in ckpt_m.iterdir())
    reset()
    _, o_slim = P.run_pipeline(make_cfg(mode=2, mesh_points=4), frames, device='cpu', checkpoint_dir=ckpt_m, nusc=nusc, lidar_cache_dir=cache)
    check('adding mesh_points does NOT change the stage hash (old checkpoints stay valid, nothing recomputed)',
          sorted(d.name for d in ckpt_m.iterdir()) == dirs_before and LOG['obj_frames'] == 0, (LOG['obj_frames'], dirs_before))
    check('old full-mesh checkpoints are slimmed on load: 4 vertices, no faces',
          all(len(r['vertices']) == 4 and r['faces'] is None for r in o_slim[0]))
    check('slimming is deterministic (same subset every time)',
          np.array_equal(o_slim[0][0]['vertices'],
                         P.slim_object_results([{'vertices': o_full[0][0]['vertices'], 'faces': None}], 4)[0]['vertices']))
    reset()
    ckpt_n = Path(tmp) / 'ckpt_mesh_fresh'
    _, o_fresh = P.run_pipeline(make_cfg(mode=2, mesh_points=4), frames, device='cpu', checkpoint_dir=ckpt_n, nusc=nusc, lidar_cache_dir=cache)
    reset()
    _, o_res = P.run_pipeline(make_cfg(mode=2, mesh_points=4), frames, device='cpu', checkpoint_dir=ckpt_n, nusc=nusc, lidar_cache_dir=cache)
    check('fresh run and resumed run give identical OBBs',
          LOG['obj_frames'] == 0 and all(np.allclose(a['obb_corners'], b['obb_corners']) for a, b in zip(o_fresh[0], o_res[0])))
    check('new checkpoints are slim on disk', all(len(r['vertices']) == 4 and r['faces'] is None
          for r in P._obj_from_ckpt(P._load(next((ckpt_n).glob('objects__*/000000.pkl.gz'))))))

    print('OBBs come from the full mesh and survive slimming + resume')
    reset()
    o_ref = P.run_pipeline(make_cfg(mode=2), frames, device='cpu', checkpoint_dir=Path(tmp) / 'ckpt_ref', nusc=nusc, lidar_cache_dir=cache)[1]
    reset()
    o_sl = P.run_pipeline(make_cfg(mode=2, mesh_points=4), frames, device='cpu', checkpoint_dir=Path(tmp) / 'ckpt_sl', nusc=nusc, lidar_cache_dir=cache)[1]
    check('slimmed run (4 of 8 vertices kept) gives the SAME boxes as the un-slimmed run',
          all(np.allclose(a['obb_corners'], b['obb_corners']) and np.isclose(a['obb_yaw'], b['obb_yaw']) for a, b in zip(o_ref[0], o_sl[0])))
    reset()
    o_sl2 = P.run_pipeline(make_cfg(mode=2, mesh_points=4), frames, device='cpu', checkpoint_dir=Path(tmp) / 'ckpt_sl', nusc=nusc, lidar_cache_dir=cache)[1]
    check('...and so does a resume from the slim checkpoint (no refit on the reduced vertices)',
          LOG['obj_frames'] == 0 and all(np.allclose(a['obb_corners'], b['obb_corners']) for a, b in zip(o_ref[0], o_sl2[0])))
    ck = P._load(next((Path(tmp) / 'ckpt_sl').glob('objects__*/000000.pkl.gz')))
    check('the checkpoint holds the reduced vertices AND the full-mesh box', all(len(r['vertices']) == 4 and r.get('obb_raw') is not None for r in ck))

    print('shared checkpoints: prepare once, then one cheap run per mode')
    shared = Path(tmp) / 'ckpt_shared'
    reset()
    b_prep, o_prep = P.run_pipeline(make_cfg(mode=2), frames, device='cpu', checkpoint_dir=shared, nusc=nusc, prepare_only=True)
    check('prepare_only returns the Body results and NO object results', o_prep == {} and len(b_prep[0]) == 1, (o_prep, b_prep.keys()))
    check('...never loads SAM3D Objects', LOG['obj_loads'] == 0 and LOG['obj_frames'] == 0)
    check('...but ran SAM3, TerraSeg and Body', LOG['seg_loads'] == 1 and LOG['ts_loads'] == 1 and LOG['body_loads'] == 1, dict(LOG))
    check('...and left the shared stages on disk (sam3, body, ground-free cloud) but no objects folder',
          any((shared).glob('sam3__*/000000.pkl.gz')) and any((shared).glob('body__*/000000.pkl.gz'))
          and any((shared / '_lidar').rglob('tok0.npy')) and not any(shared.glob('objects__*')))
    for _mo in (2, 7):
        reset()
        b_m, o_m = P.run_pipeline(make_cfg(mode=_mo), frames, device='cpu', checkpoint_dir=shared, nusc=nusc)
        check(f'mode {_mo} on the shared folder reloads NOTHING but SAM3D Objects',
              LOG['seg_loads'] == 0 and LOG['body_loads'] == 0 and LOG['ts_loads'] == 0 and LOG['pl_fits'] == 0
              and LOG['obj_loads'] == 1 and LOG['obj_frames'] == 1, dict(LOG))
        check(f'...and still returns objects and the cached pedestrians', len(o_m[0]) == 3 and len(b_m[0]) == 1)
    check('each mode keeps its own objects folder in the shared checkpoints', len(list(shared.glob('objects__*'))) == 2,
          [d.name for d in shared.glob('objects__*')])
    reset()
    P.run_pipeline(make_cfg(mode=2), frames, device='cpu', checkpoint_dir=shared, nusc=nusc)
    check('rerunning a finished mode loads and computes nothing', LOG['obj_loads'] == 0 and LOG['obj_frames'] == 0)

    print('no checkpointing')
    reset()
    b, o = P.run_pipeline(make_cfg(mode=2), frames, device='cpu', checkpoint_dir=None, nusc=nusc)
    check('runs with no checkpoint dir and no lidar cache dir (temp dir used)', len(o[0]) == 3)

    print('config errors')
    for bad, frag in (('o3_local_affine', 'old mode name'), ('o5_mask_hdbscan', 'old mode name'),
                      ('baseline', 'was removed'), (12, 'not in')):
        try:
            P.run_pipeline(make_cfg(mode=bad), frames, device='cpu', nusc=nusc)
            check(f'{bad!r} rejected', False)
        except ValueError as e:
            check(f'{bad!r} rejected with a helpful message', frag in str(e), str(e))

    print('multi-camera: ground-free cloud shared across cameras')
    reset()
    fpc = make_frames(nusc, tmp, cams=('CAM_FRONT_LEFT', 'CAM_FRONT'))
    ckpt2 = Path(tmp) / 'ckpt_multi'
    body_all, obj_all = P.run_multi_camera_pipeline(make_cfg(mode=2), fpc, device='cpu', checkpoint_dir=ckpt2, nusc=nusc)
    check('TerraSeg ran ONCE for two cameras of the same keyframe', LOG['ts_loads'] == 1 and LOG['ts_sweeps'] == 5, (LOG['ts_loads'], LOG['ts_sweeps']))
    check('both cameras produced results', set(obj_all) == {'CAM_FRONT_LEFT', 'CAM_FRONT'} and all(len(v[0]) == 3 for v in obj_all.values()))
    check('shared cache lives in <checkpoint_dir>/_lidar', len(list((ckpt2 / '_lidar').rglob('tok0.npy'))) == 1)
    check('per-camera checkpoint dirs', (ckpt2 / 'CAM_FRONT').is_dir() and (ckpt2 / 'CAM_FRONT_LEFT').is_dir())

print(f'\nall {len(_n)} checks passed')
