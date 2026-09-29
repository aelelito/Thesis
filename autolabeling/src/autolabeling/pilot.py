"""
Helpers for the mask / free-space pilot (notebook `testing/sam3d_mask_freespace_analysis.ipynb`).

Question: does the SAM3D Objects mesh fit its 2D mask, sit on the LiDAR depth, and stay out of space the LiDAR has
certified as empty? Each function here does one measurement for one object; the notebook only orchestrates and plots.

Coordinates: meshes come out of the pipeline in the camera frame (x right, y down, z forward); free-space and boxes
live in the ego frame of the keyframe.
"""
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional

import numpy as np
import yaml

from .utils import freespace as fs
from .utils import mesh_mask as mm
from .utils.geometry import cam_to_ego

DEV_ROOT = Path('/workspace')

# The four frames of the pilot (nuScenes scene idx 1 keyframe 11 was dropped: many objects, no added value). scene_idx / frame_idx index nusc.scene[...] and the keyframes of that scene.
SELECTIONS = [
    dict(tag='ecp10_f560',   dataset='ecp',           scene_idx=10, frame_idx=560,  camera='CAM_FRONT'),
    dict(tag='ecp10_f1080',  dataset='ecp',           scene_idx=10, frame_idx=1080, camera='CAM_FRONT_LEFT'),
    dict(tag='ecp10_f1240',  dataset='ecp',           scene_idx=10, frame_idx=1240, camera='CAM_FRONT_LEFT'),
    dict(tag='nusc3_kf3',    dataset='nuscenes_mini', scene_idx=3,  frame_idx=3,    camera='CAM_FRONT_RIGHT'),
]

DATASET_INFO = {
    'nuscenes_mini': dict(config='nuscenes.yaml', version='v1.0-mini', root='data/nuScenes_mini',
                          gt_categories=('vehicle',)),
    'ecp':           dict(config='ecp.yaml', version='v1.0-trainval', root='data/ecp',
                          gt_categories=('car', 'trailer', 'motorcycle', 'bicycle')),
}


# ── Setup ──────────────────────────────────────────────────────────────────────────

def _ns(d):
    return SimpleNamespace(**{k: _ns(v) for k, v in d.items()}) if isinstance(d, dict) else d


def load_cfg(dataset: str, pointmap_mode: int = 2, keep_pedestrians: bool = False):
    """
    The production config of `dataset`, with two changes for this analysis: full meshes with faces are kept
    (mesh_points 0), and (unless keep_pedestrians) the pedestrian prompt is removed so SAM3D Body does not run
    (only SAM3D Objects is analysed; bicycles then keep their own OBB, without the rider merge).
    """
    with open(DEV_ROOT / 'autolabeling' / 'configs' / DATASET_INFO[dataset]['config']) as f:
        raw = yaml.safe_load(f)
    raw['sam3d_objects']['mesh_points'] = 0
    raw['sam3d_objects']['pointmap_mode'] = pointmap_mode
    if not keep_pedestrians:
        raw['prompts'] = {k: v for k, v in raw['prompts'].items() if v != 'body'}
    return _ns(raw)


def open_nusc(dataset: str):
    from nuscenes.nuscenes import NuScenes
    info = DATASET_INFO[dataset]
    return NuScenes(version=info['version'], dataroot=str(DEV_ROOT / info['root']), verbose=False)


def get_frame(nusc, sel: dict):
    from .data.nuscenes_loader import collect_frames
    name = nusc.scene[sel['scene_idx']]['name']
    return collect_frames(nusc, [name], sel['camera'], sel['frame_idx'], sel['frame_idx'] + 1)[0]


def run_selection(sel: dict, nusc, cfg, ckpt_root: Path, device: Optional[str] = None):
    """Run the production pipeline (SAM3 -> TerraSeg -> SAM3D Objects) on one frame. Resumes from checkpoints."""
    from .pipeline import run_pipeline
    frame = get_frame(nusc, sel)
    _, objs = run_pipeline(cfg, [frame], device=device, checkpoint_dir=Path(ckpt_root) / sel['tag'], nusc=nusc)
    return frame, objs[0]


def load_cached_inputs(sel: dict, frame, ckpt_root: Path):
    """SAM3 detections and the ground-free aggregated cloud the pipeline cached for this frame (for re-running SAM3D)."""
    from .pipeline import _load, _sam3_from_ckpt
    ck = Path(ckpt_root) / sel['tag']
    sam3 = _sam3_from_ckpt(_load(next(ck.glob('sam3__*/000000.pkl.gz'))))
    pts_ng = np.load(next((ck / '_lidar').glob(f'nonground__*/{frame.sample_token}.npy'))).astype(np.float64)
    return sam3, pts_ng


def single_sweep(nusc, frame):
    """The keyframe's own LiDAR sweep (ground INCLUDED: ground returns certify the free space above the ground),
    ego-body returns removed, in the keyframe ego frame; plus the ray origin (LiDAR position in that frame)."""
    from .utils.lidar import load_lidar_pts_aggregated
    pts = load_lidar_pts_aggregated(nusc, frame, 0, 0, use_ego_body_filter=True, ego_box_half_x=4.0,
                                    ego_box_half_y=1.5, ego_box_z_min=0.5, ego_box_z_max=2.5)
    return pts, np.asarray(frame.t_l2e, np.float64)


def sweep_timing(nusc, frame) -> Dict[str, float]:
    """Camera vs LiDAR timestamp of the keyframe [ms]. The camera sees objects at t_cam, the sweep at ~t_lidar."""
    sample = nusc.get('sample', frame.sample_token)
    t_cam = nusc.get('sample_data', sample['data'][frame.camera_name])['timestamp']
    t_lid = nusc.get('sample_data', sample['data']['LIDAR_TOP'])['timestamp']
    return dict(camera_minus_lidar_ms=(t_cam - t_lid) / 1000.0)


def gt_with_speed(nusc, frame) -> List[dict]:
    """GT boxes of the keyframe (ego frame) with the annotation's speed [m/s] in the world frame (NaN if unknown).
    Speed ~0 = parked: the sweep/camera time offset cannot move it, so a free-space violation is not a timing effect."""
    from .utils.diagnostics import gt_boxes_ego
    gts = gt_boxes_ego(nusc, frame.sample_token, frame.R_e2g, frame.t_e2g)
    for g, tok in zip(gts, nusc.get('sample', frame.sample_token)['anns']):
        try:
            v = nusc.box_velocity(tok)
            g['speed'] = float(np.linalg.norm(v[:2])) if np.all(np.isfinite(v)) else float('nan')
        except Exception:
            g['speed'] = float('nan')
    return gts


# ── Per-object measurements ────────────────────────────────────────────────────────

def lidar_in_camera(frame, pts_ego: np.ndarray, H: int, W: int):
    """LiDAR returns (ego frame) -> pixel coordinates and camera depth, only those in front of the camera and in the image."""
    pc = (frame.R_c2e.T @ (np.asarray(pts_ego, np.float64) - frame.t_c2e).T).T
    front = pc[:, 2] > 0.5
    pc = pc[front]
    u = pc[:, 0] / pc[:, 2] * frame.K[0, 0] + frame.K[0, 2]
    v = pc[:, 1] / pc[:, 2] * frame.K[1, 1] + frame.K[1, 2]
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return u[ok], v[ok], pc[ok, 2]


def box_of(obb_center, obb_dims, obb_yaw):
    """(centre, length, width, height, yaw) from a pipeline OBB (dims = [length, width, height])."""
    return np.asarray(obb_center, np.float64), float(obb_dims[0]), float(obb_dims[1]), float(obb_dims[2]), float(obb_yaw)


def gt_box_of(g: dict):
    w, l, h = g['size']
    return np.asarray(g['center'], np.float64), float(l), float(w), float(h), float(g['yaw'])


def _prefixed(prefix: str, d: dict) -> dict:
    return {f'{prefix}_{k}': v for k, v in d.items()}


def analyze_object(r: dict, frame, grid: fs.FreeSpaceGrid, pts_sweep: np.ndarray, H: int, W: int,
                   gt: Optional[dict] = None, occluders: Optional[np.ndarray] = None, n_surface: int = 20000,
                   margins: Iterable[float] = (0.0, 0.1, 0.2, 0.3), device: str = 'cpu', claim: float = 0.2) -> dict:
    """
    All measurements for one SAM3D Objects result `r` (needs vertices, faces, binary_mask, obb_center/dims/yaw).
    Returns a flat dict (one DataFrame row) and, under key '_render', the silhouette and depth image for plotting.
    Free-space numbers are given for the mesh surface, the filled mesh volume, the fitted OBB volume and the GT box volume.
    """
    verts, faces, mask = r['vertices'], r['faces'], r['binary_mask']
    sil, depth = mm.render_mesh(verts, faces, frame.K, H, W, device=device)
    row = _prefixed('mask', mm.mask_agreement(sil, mask, occluders))
    u, v, z = lidar_in_camera(frame, pts_sweep, H, W)
    row.update(_prefixed('depth', mm.depth_residuals(depth, mask, u, v, z)))

    verts_ego = cam_to_ego(np.asarray(verts, np.float64), frame.R_c2e, frame.t_c2e)
    surf = fs.sample_mesh_surface(verts_ego, faces, n_surface)
    row.update(_prefixed('surf', fs.freespace_stats(grid, surf, margins, claim=claim)))
    vol = fs.mesh_volume_points(verts_ego, faces, grid.vs)
    row.update(_prefixed('vol', fs.freespace_stats(grid, vol, margins, volume=True, claim=claim)))
    row.update(_prefixed('obb', fs.freespace_stats(grid, fs.box_lattice(*box_of(r['obb_center'], r['obb_dims'], r['obb_yaw']), grid.vs),
                                                    margins, volume=True, claim=claim)))
    # how far below the local ground the object reaches (positive = below): mesh, its OBB and the GT box (the GT box is the baseline)
    oc, ol, ow, oh, oy = box_of(r['obb_center'], r['obb_dims'], r['obb_yaw'])
    gz = fs.local_ground_z(pts_sweep, oc, ol, ow, oy)
    row['ground_z'] = gz
    row['mesh_below_ground'] = gz - float(np.percentile(verts_ego[:, 2], 0.5))
    row['obb_below_ground'] = gz - (oc[2] - oh / 2)
    if gt is not None:
        row['gt_below_ground'] = gz - (gt['center'][2] - gt['size'][2] / 2)
        # independent of the ground estimate: how far the mesh bottom lies below the bottom of the GT box (GT boxes rest on the ground)
        row['mesh_below_gt_bottom'] = (gt['center'][2] - gt['size'][2] / 2) - float(np.percentile(verts_ego[:, 2], 0.5))
        row.update(_prefixed('gt', fs.freespace_stats(grid, fs.box_lattice(*gt_box_of(gt), grid.vs), margins, volume=True, claim=claim)))
        row['gt_speed'] = gt.get('speed', float('nan'))
        # share of the mesh volume outside the GT box (GT box enlarged by 0.2 m): mesh overshoot measured against the only GT we have
        c, l, w, h, yaw = gt_box_of(gt)
        d = vol - c
        cs, sn = np.cos(-yaw), np.sin(-yaw)
        lx, ly, lz = cs * d[:, 0] - sn * d[:, 1], sn * d[:, 0] + cs * d[:, 1], d[:, 2]
        inside = (np.abs(lx) <= l / 2 + 0.2) & (np.abs(ly) <= w / 2 + 0.2) & (np.abs(lz) <= h / 2 + 0.2)
        row['mesh_vol_outside_gt_box'] = float(1.0 - inside.mean())
    row['_verts_ego'] = verts_ego
    row['_render'] = (sil, depth)
    row['_surf'] = surf
    row['_vol'] = vol
    return row


def occluder_mask(other_masks: List[np.ndarray], shape) -> np.ndarray:
    occ = np.zeros(shape, bool)
    for m in other_masks:
        occ |= m.astype(bool)
    return occ


# ── Mask perturbation ──────────────────────────────────────────────────────────────

def mask_variants(erode_dilate: Iterable[float] = (0.05, 0.15, 0.3), shift: Iterable[float] = (0.1, 0.3)) -> Dict[str, callable]:
    """
    name -> function(binary mask) -> perturbed mask. Sizes are RELATIVE to the object so a small far car and a large near
    one get comparable perturbations: erode / dilate radius = fraction of the mask's shorter bounding-box side, shift =
    fraction of the bounding-box width (sideways). For scale: SAM3D's training augmentation used shifts <= 5 px and 2-5 px
    erode/dilate kernels, so the small fractions are inside that range and the large ones far outside it.
    'repeat' changes nothing (noise floor of the run itself); 'bbox' fills the bounding box.
    """
    import cv2

    def extent(m):
        ys, xs = np.where(m)
        return xs.min(), xs.max() + 1, ys.min(), ys.max() + 1

    def morph(frac, grow):
        def f(m):
            x0, x1, y0, y1 = extent(m)
            k = max(1, int(round(frac * min(x1 - x0, y1 - y0))))
            ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
            out = (cv2.dilate if grow else cv2.erode)(m.astype(np.uint8), ker).astype(bool)
            return out if out.any() else m.copy()          # an erosion must not delete the object
        return f

    def shifted(frac):
        def f(m):
            x0, x1, y0, y1 = extent(m)
            dx = max(1, int(round(frac * (x1 - x0))))
            out = np.zeros_like(m)
            out[:, dx:] = m[:, :-dx]
            return out
        return f

    def bbox(m):
        x0, x1, y0, y1 = extent(m)
        out = np.zeros_like(m)
        out[y0:y1, x0:x1] = True
        return out

    variants = {'repeat': lambda m: m.copy()}
    for fr in erode_dilate:
        variants[f'erode{int(fr * 100)}%'] = morph(fr, False)
    for fr in erode_dilate:
        variants[f'dilate{int(fr * 100)}%'] = morph(fr, True)
    for fr in shift:
        variants[f'shift{int(fr * 100)}%'] = shifted(fr)
    variants['bbox'] = bbox
    return variants


def build_objects_model(cfg, device: Optional[str] = None):
    """SAM3D Objects wrapper configured exactly like the production pipeline does (for re-running objects in the notebook)."""
    from .models.sam3d_objects import SAM3DObjectsModel, resolve_pointmap_mode
    o = cfg.sam3d_objects
    hd = {k: vars(v) for k, v in vars(o.hdbscan).items()}
    obj_cfg = (Path(cfg.models.sam3d_obj_cfg) if cfg.models.sam3d_obj_cfg
               else DEV_ROOT / 'models' / 'SAM3D' / 'sam-3d-objects' / 'checkpoints' / 'hf' / 'pipeline.yaml')
    model = SAM3DObjectsModel(
        repo_path=str(obj_cfg.parent.parent.parent), config_path=str(obj_cfg), prompts=vars(cfg.prompts), device=device,
        pointmap_mode=resolve_pointmap_mode(o.pointmap_mode), hdbscan_params=hd,
        proximity_min_pts=int(float(getattr(o, 'proximity_min_pts', 30))),
        proximity_ratio=float(getattr(o, 'proximity_ratio', 0.70)),
        mask_erode_px=int(float(getattr(o, 'mask_erode_px', 0))), mask_erode_min_px=int(float(getattr(o, 'mask_erode_min_px', 0))))
    model.load()
    return model


def run_mask_perturbation(obj_model, frame, sam3: dict, pts_ng: np.ndarray, only: set, variants: Dict[str, callable],
                          freeze_shift: bool = True) -> Dict[str, list]:
    """
    Re-run SAM3D Objects for the objects in `only` ({(prompt, sam3_index)}) once per mask variant. Everything else
    (image, pointmap built from the ORIGINAL mask, seed) is identical, so any change in the mesh comes from the
    mask SAM3D is handed. freeze_shift=True: SAM3D's shift and scale (its depth anchor, the median 3D point in the mask)
    are still computed from the ORIGINAL mask, so the perturbation cannot move the object through the anchor and only
    acts through crop, image and mask tokens. False: SAM3D computes them from the perturbed mask it receives.
    """
    from .utils.meshes import attach_full_obbs
    out = {}
    try:
        obj_model.freeze_ssi = freeze_shift
        for name, fn in variants.items():
            obj_model.mask_hook = fn
            res = obj_model.run_frame(frame, sam3, pts_ng, only=only)
            attach_full_obbs(res, frame)
            out[name] = res
    finally:
        obj_model.mask_hook = None
        obj_model.freeze_ssi = False
    return out


def perturbation_rows(runs: Dict[str, list], frame, grid: fs.FreeSpaceGrid, H: int, W: int, ref: str = 'repeat',
                      d_claim: float = 0.2, device: str = 'cpu') -> List[dict]:
    """
    One row per (variant, object). Compared with the `ref` run of the same object:
      iou_new / iou_orig : mesh silhouette vs the mask SAM3D was given / vs the original SAM3 mask
                           (mask is a constraint -> iou_new stays high and iou_orig drops; a hint -> the mesh stays put)
      d_center, size ratios, d_yaw : change of the fitted OBB   shift_dz : change of SAM3D's shift along z [m]
      surf_free : share of the mesh surface in free space (>= d_claim in front of the nearest return)
    """
    def key(r):
        return (r['prompt'], r['sam3_index'])

    base = {key(r): r for r in runs[ref]}
    rows = []
    for name, res in runs.items():
        for r in res:
            b = base[key(r)]
            sil, _ = mm.render_mesh(r['vertices'], r['faces'], frame.K, H, W, device=device)
            new_mask = r['infer_mask'] if r.get('infer_mask') is not None else r['binary_mask']
            a_new, a_orig = mm.mask_agreement(sil, new_mask), mm.mask_agreement(sil, r['binary_mask'])
            verts_ego = cam_to_ego(np.asarray(r['vertices'], np.float64), frame.R_c2e, frame.t_c2e)
            st = fs.freespace_stats(grid, fs.sample_mesh_surface(verts_ego, r['faces'], 10000), (d_claim,))
            ro, bo = r['obb_raw'], b['obb_raw']
            dy = (ro['yaw'] - bo['yaw']) % np.pi
            rows.append(dict(variant=name, prompt=r['prompt'], sam3_index=r['sam3_index'],
                             iou_new=a_new['iou'], iou_orig=a_orig['iou'], recall_new=a_new['recall'], leak_new=a_new['leak'],
                             d_center=float(np.linalg.norm(np.asarray(ro['center'])[:2] - np.asarray(bo['center'])[:2])),
                             len_ratio=float(ro['dims'][0] / bo['dims'][0]), wid_ratio=float(ro['dims'][1] / bo['dims'][1]),
                             hgt_ratio=float(ro['dims'][2] / bo['dims'][2]), d_yaw=float(np.degrees(min(dy, np.pi - dy))),
                             shift_dz=float(r['ssi_shift'][2] - b['ssi_shift'][2]),
                             surf_free=st[f'frac_free_m{d_claim:g}']))
    return rows


# ── Controls: what do the violating points mean? ───────────────────────────────────

def analyze_controls(r: dict, frame, grid: fs.FreeSpaceGrid, pts_sweep: np.ndarray, H: int, W: int, verts_ego: np.ndarray, faces,
                     surf: np.ndarray, vol: np.ndarray, gt: Optional[dict] = None, claim: float = 0.2,
                     shifts: Iterable[float] = tuple(np.round(np.arange(-0.5, 0.51, 0.1), 2))) -> dict:
    """
    Three controls for one object (see utils/controls.py). Returns a flat dict of scalars and, under '_curves', arrays for plotting.
      floor_*   : free-space share of the mesh moved to the GT position + yaw (floor_free) and also stretched to the GT size
                  (floor_scaled_free); needs a GT match. What a correctly placed mesh of this shape shows.
      shift_*   : slide the mesh along the LiDAR ray by -0.5 .. +0.5 m (positive = away from the sensor). best_shift_free = shift with the
                  least free-space share, best_shift_lidar = shift that puts the surface closest to the in-mask returns.
      red_*     : where the violating points project and whether a depth constraint on the mask would have seen them.
    """
    from .utils import controls as ct
    oc, ol, ow, oh, oy = box_of(r['obb_center'], r['obb_dims'], r['obb_yaw'])
    row, curves = {}, {}
    key = f'frac_free_m{claim:g}'

    if gt is not None:
        gc, gl, gw, gh, gy = gt_box_of(gt)
        for name, dims in (('floor', None), ('floor_scaled', (gl, gw, gh))):
            V = ct.align_mesh_to_box(verts_ego, oc, oy, (ol, ow, oh), gc, gy, dims)
            st = fs.freespace_stats(grid, fs.sample_mesh_surface(V, faces, len(surf), seed=1), (claim,), claim=claim)
            row[f'{name}_free'] = st[key]
            row[f'{name}_unknown'] = st['frac_unknown']

    # in-mask LiDAR returns near the mesh (background seen through gaps is dropped by the 1.5 m radius)
    u, v, z = lidar_in_camera(frame, pts_sweep, H, W)
    inm = r['binary_mask'][np.clip(np.round(v).astype(int), 0, H - 1), np.clip(np.round(u).astype(int), 0, W - 1)]
    ret_uvz = np.stack([u[inm], v[inm], z[inm]], axis=1)
    ret_ego = _returns_in_mask_ego(frame, pts_sweep, r['binary_mask'], H, W)
    from scipy.spatial import cKDTree
    near = cKDTree(surf).query(ret_ego)[0] <= 1.5 if len(ret_ego) else np.zeros(0, bool)
    cur = ct.ray_shift_curve(grid, surf, oc, ret_ego[near], shifts, claim)
    curves['shift'] = cur
    row['shift_free_at_0'] = float(cur['free'][np.argmin(np.abs(cur['shifts']))])
    i_f = int(np.argmin(cur['free']))
    row['best_shift_free'], row['free_at_best_shift'] = float(cur['shifts'][i_f]), float(cur['free'][i_f])
    if np.isfinite(cur['lidar_dist']).any():
        i_l = int(np.nanargmin(cur['lidar_dist']))
        row['best_shift_lidar'] = float(cur['shifts'][i_l])
        row['lidar_dist_at_0'] = float(cur['lidar_dist'][np.argmin(np.abs(cur['shifts']))])
        row['lidar_dist_best'] = float(cur['lidar_dist'][i_l])
    row['n_returns_near'] = int(near.sum())

    P, ends = ct.violating_points(grid, surf, claim)
    row.update(ct.classify_violations(P, ends, frame, r['binary_mask'], ret_uvz, vol, grid.vs))
    curves['red_points'] = P
    return row | {'_curves': curves}


def _returns_in_mask_ego(frame, pts_ego: np.ndarray, mask: np.ndarray, H: int, W: int) -> np.ndarray:
    """LiDAR returns (ego frame) whose pixel lies inside the mask."""
    pc = (frame.R_c2e.T @ (np.asarray(pts_ego, np.float64) - frame.t_c2e).T).T
    front = pc[:, 2] > 0.5
    pe, pc = np.asarray(pts_ego)[front], pc[front]
    u = np.round(pc[:, 0] / pc[:, 2] * frame.K[0, 0] + frame.K[0, 2]).astype(int)
    v = np.round(pc[:, 1] / pc[:, 2] * frame.K[1, 1] + frame.K[1, 2]).astype(int)
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    pe, u, v = pe[ok], u[ok], v[ok]
    return pe[mask[v, u].astype(bool)]
