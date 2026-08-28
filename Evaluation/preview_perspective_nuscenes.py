#!/home/lleba/miniconda3/envs/autolabeling/bin/python
"""
Quick single-frame dual-panel perspective preview for nuScenes mini.

Renders one annotated frame as front (top) + rear (bottom) and saves to
~/Desktop/preview_nuscenes.png.

Tune FRONT_CAM_POS / FRONT_LOOK_AT, BACK_CAM_POS / BACK_LOOK_AT, and FOV
at the top, then copy values into visualize_autolabeling_results.py:
    PERSP_FOV_DUAL, PERSP_FRONT_CAM_POS/LOOK_AT, PERSP_BACK_CAM_POS/LOOK_AT

Usage:
    conda activate autolabeling
    python preview_perspective_nuscenes.py
    python preview_perspective_nuscenes.py --frame 5
"""

import argparse
import json
import os
import numpy as np
import cv2
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

# ── Tune these ────────────────────────────────────────────────────────────────
FOV = 35.0

FRONT_CAM_POS = np.array([-15.0, 0.0,  5.0])
FRONT_LOOK_AT = np.array([ 30.0, 0.0,  2.0])

BACK_CAM_POS  = np.array([ 15.0, 0.0,  5.0])
BACK_LOOK_AT  = np.array([-30.0, 0.0,  2.0])
# ──────────────────────────────────────────────────────────────────────────────

DATA_ROOT  = '/media/lleba/ECP_Nuscenes_01/nuScenes_mini'
VERSION    = 'v1.0-mini'
LABEL_JSON = (
    '/media/lleba/ECP_Nuscenes_01/autolabeling/output/nuscenes_mini/'
    'lidar_integration_objects_o3_body_b1_all_cameras/'
    'autolabel_nuscenes_mini_train_8class.json'
)
OUT_PATH   = os.path.expanduser('~/Desktop/preview_nuscenes.png')

W         = 1920
PANEL_H   = 540    # each panel; total height = 1080
RANGE_M   = 60.0
BG_COLOR  = (12,  12,  12)
GT_COLOR  = (40,  40, 220)
PSEUDO_COLOR = (220, 80, 40)
BOX_EDGES = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]


def height_colormap(z):
    t = ((z.clip(-2.0, 4.0) + 2.0) / 6.0)
    out = np.zeros((len(t), 3), np.uint8)
    m = t < 0.25;  s = (t[m] / 0.25 * 255).astype(np.uint8)
    out[m, 0] = 255; out[m, 1] = s
    m = (t >= 0.25) & (t < 0.50);  s = ((t[m] - 0.25) / 0.25 * 255).astype(np.uint8)
    out[m, 0] = 255 - s; out[m, 1] = 255
    m = (t >= 0.50) & (t < 0.75);  s = ((t[m] - 0.50) / 0.25 * 255).astype(np.uint8)
    out[m, 1] = 255; out[m, 2] = s
    m = t >= 0.75;  s = ((t[m] - 0.75) / 0.25 * 255).astype(np.uint8)
    out[m, 1] = 255 - s; out[m, 2] = 255
    return out


def build_R(cam_pos, look_at):
    fwd   = look_at - cam_pos;  fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0]);  right /= np.linalg.norm(right)
    up    = np.cross(right, fwd)
    return np.stack([right, up, fwd], axis=0)


def project(pts, R_cam, cam_pos):
    pts_c = (R_cam @ (pts - cam_pos).T).T
    depth = pts_c[:, 2]
    valid = depth > 0.2
    pts_c = pts_c[valid];  depth = depth[valid]
    f = PANEL_H / (2.0 * np.tan(np.radians(FOV / 2.0)))
    u = f * pts_c[:, 0] / depth + W / 2.0
    v = -f * pts_c[:, 1] / depth + PANEL_H / 2.0
    return np.stack([u, v], axis=1), valid


def obb_corners(center_e, size, yaw):
    hw, hl, hh = size[0]/2, size[1]/2, size[2]/2
    c, s = np.cos(yaw), np.sin(yaw)
    loc = np.array([[ hl, hw, hh],[ hl,-hw, hh],[-hl,-hw, hh],[-hl, hw, hh],
                    [ hl, hw,-hh],[ hl,-hw,-hh],[-hl,-hw,-hh],[-hl, hw,-hh]])
    R = np.array([[c,-s,0],[s,c,0],[0,0,1]])
    return (R @ loc.T).T + center_e


def draw_boxes(canvas, boxes, color, R_cam, cam_pos, R_e2g, t_e2g):
    f = PANEL_H / (2.0 * np.tan(np.radians(FOV / 2.0)))
    for box in boxes:
        c_g = np.array(box['translation'])
        q_g = Quaternion(box['rotation'])
        center_e = R_e2g.T @ (c_g - t_e2g)
        yaw = (Quaternion(matrix=R_e2g).inverse * q_g).yaw_pitch_roll[0]
        corners = obb_corners(center_e, box['size'], yaw)
        pts_c   = (R_cam @ (corners - cam_pos).T).T
        depths  = pts_c[:, 2]
        if np.any(depths <= 0.2):
            continue
        u = (f * pts_c[:, 0] / depths + W / 2).astype(np.int32)
        v = (-f * pts_c[:, 1] / depths + PANEL_H / 2).astype(np.int32)
        for i, j in BOX_EDGES:
            cv2.line(canvas, (int(u[i]), int(v[i])), (int(u[j]), int(v[j])),
                     color, 2, cv2.LINE_AA)


def render_panel(pts_ego, pseudo_boxes, gt_boxes, R_e2g, t_e2g,
                 cam_pos, look_at, label):
    canvas = np.full((PANEL_H, W, 3), BG_COLOR, dtype=np.uint8)
    R_cam  = build_R(cam_pos, look_at)

    dist    = np.linalg.norm(pts_ego[:, :2], axis=1)
    pts_vis = pts_ego[dist < RANGE_M]
    if len(pts_vis):
        pix, valid = project(pts_vis, R_cam, cam_pos)
        colors = height_colormap(pts_vis[valid, 2])
        cols = pix[:, 0].astype(np.int32)
        rows = pix[:, 1].astype(np.int32)
        mask = (cols >= 0) & (cols < W) & (rows >= 0) & (rows < PANEL_H)
        canvas[rows[mask], cols[mask]] = colors[mask]

    draw_boxes(canvas, gt_boxes,     GT_COLOR,     R_cam, cam_pos, R_e2g, t_e2g)
    draw_boxes(canvas, pseudo_boxes, PSEUDO_COLOR, R_cam, cam_pos, R_e2g, t_e2g)

    (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.putText(canvas, label, (W - tw - 12, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (110, 110, 110), 2, cv2.LINE_AA)
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frame', type=int, default=0,
                        help='index into annotated frames to preview (default: 0)')
    args = parser.parse_args()

    print('Loading NuScenes …')
    nusc = NuScenes(version=VERSION, dataroot=DATA_ROOT, verbose=False)

    with open(LABEL_JSON) as f:
        pseudo_labels = json.load(f)['results']

    gt_by_sample = {}
    for ann in nusc.sample_annotation:
        gt_by_sample.setdefault(ann['sample_token'], []).append(ann)

    annotated_sd = []
    for scene in nusc.scene:
        tok = scene['first_sample_token']
        while tok:
            sample = nusc.get('sample', tok)
            if 'LIDAR_TOP' in sample['data'] and tok in pseudo_labels:
                annotated_sd.append((scene, nusc.get('sample_data', sample['data']['LIDAR_TOP'])))
            tok = sample['next']

    if not annotated_sd:
        print('No annotated frames found.')
        return

    idx = args.frame % len(annotated_sd)
    scene, sd = annotated_sd[idx]
    print(f'  Rendering frame {idx}/{len(annotated_sd)-1}: '
          f'scene={scene["name"]}  sample={sd["sample_token"][:8]}…')

    lidar_cal = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
    R_l2e = Quaternion(lidar_cal['rotation']).rotation_matrix.astype(np.float64)
    t_l2e = np.array(lidar_cal['translation'], dtype=np.float64)

    raw = np.fromfile(os.path.join(DATA_ROOT, sd['filename']), dtype=np.float32).reshape(-1, 5)[:, :3]
    pts_ego = (R_l2e @ raw.astype(np.float64).T).T + t_l2e

    ego_pose = nusc.get('ego_pose', sd['ego_pose_token'])
    R_e2g = Quaternion(ego_pose['rotation']).rotation_matrix.astype(np.float64)
    t_e2g = np.array(ego_pose['translation'], dtype=np.float64)

    pseudo_boxes = pseudo_labels.get(sd['sample_token'], [])
    gt_boxes     = gt_by_sample.get(sd['sample_token'], [])

    front = render_panel(pts_ego, pseudo_boxes, gt_boxes, R_e2g, t_e2g,
                         FRONT_CAM_POS, FRONT_LOOK_AT, 'FRONT')
    rear  = render_panel(pts_ego, pseudo_boxes, gt_boxes, R_e2g, t_e2g,
                         BACK_CAM_POS,  BACK_LOOK_AT,  'REAR')
    front[-2:, :] = 28

    canvas = np.concatenate([front, rear], axis=0)

    # Overlay current params
    info = [
        f'frame {idx}/{len(annotated_sd)-1}  (--frame N to change)',
        f'FOV            = {FOV}',
        f'FRONT_CAM_POS  = {FRONT_CAM_POS.tolist()}',
        f'FRONT_LOOK_AT  = {FRONT_LOOK_AT.tolist()}',
        f'BACK_CAM_POS   = {BACK_CAM_POS.tolist()}',
        f'BACK_LOOK_AT   = {BACK_LOOK_AT.tolist()}',
    ]
    for i, line in enumerate(info):
        cv2.putText(canvas, line, (14, 28 + i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1, cv2.LINE_AA)

    cv2.imwrite(OUT_PATH, canvas)
    print(f'  Saved → {OUT_PATH}')
    print()
    print('  When happy, copy into visualize_autolabeling_results.py:')
    print('    PERSP_FOV_DUAL, PERSP_FRONT_CAM_POS/LOOK_AT, PERSP_BACK_CAM_POS/LOOK_AT')


if __name__ == '__main__':
    main()
