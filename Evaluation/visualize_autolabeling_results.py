#!/home/lleba/miniconda3/envs/autolabeling/bin/python
"""
ECP LiDAR + pseudo-label BEV / perspective video generator.

Ego vehicle is always centred. Every LiDAR frame in the scene becomes one video
frame (or only annotated frames with --annotated-only). Pseudo-label boxes are
overlaid on annotated frames (colour-coded by class).

Two view modes
--------------
  bev         Top-down bird's-eye view. Boxes drawn as flat footprint rectangles.
  perspective Slightly-elevated perspective view (like a drone behind the car).
              Boxes drawn as full 3D wireframes.

NOTE on ECP / ecp2nuscenes format
----------------------------------
In ecp2nuscenes every recording frame is stored as a nuScenes *sample*
(keyframe). There are no separate sweep records. "Annotated" frames are those
whose sample_token appears in the pseudo-label results dict.

Usage
-----
    conda activate autolabeling

    # BEV, all frames, 3 labelled scenes:
    python visualize_bev.py

    # Perspective, annotated frames only:
    python visualize_bev.py --view perspective --annotated-only

    # One scene, BEV:
    python visualize_bev.py --scene scene-euro-citystrasbourg-scenariolatesession-00002_1

    # Custom FPS:
    python visualize_bev.py --fps 10
"""

import argparse
import json
import os
import subprocess
import sys

import cv2
import numpy as np
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

# ── Defaults ──────────────────────────────────────────────────────────────────
DATA_ROOT  = '/media/lleba/ECP_Nuscenes_01/output2/ecp2nuscenes'
VERSION    = 'v1.0-trainval'
LABEL_JSON = (
    '/media/lleba/ECP_Nuscenes_01/autolabeling/output/ecp/'
    'lidar_integration_objects_o3_body_b1_all_cameras/'
    'autolabel_ecp_annotated_8class.json'
)
OUT_DIR = '/home/lleba/Desktop/bev_videos'

# ── BEV canvas ────────────────────────────────────────────────────────────────
RANGE_M  = 30.0
BEV_W    = 1000
BEV_H    = 1000
BEV_SCALE = BEV_H / (2.0 * RANGE_M)   # px / m

# ── Perspective canvas & virtual cameras (ego frame: X=fwd, Y=left, Z=up) ─────
PERSP_W       = 2400
PERSP_H       = 1800           # total height
PERSP_PANEL_H = PERSP_H // 2  # 900 px per panel for dual-panel mode (even → libx264 safe)
PERSP_RANGE_M = 80.0           # clip LiDAR points beyond this BEV distance

# ── Dual-panel mode (nuScenes mini: front on top, rear on bottom) ─────────────
# Each panel is PERSP_PANEL_H tall. Camera sits behind/ahead at modest elevation.
PERSP_FOV_DUAL         = 35.0
PERSP_FRONT_CAM_POS    = np.array([-15.0, 0.0,  5.0])
PERSP_FRONT_LOOK_AT    = np.array([ 30.0, 0.0,  2.0])
PERSP_BACK_CAM_POS     = np.array([ 15.0, 0.0,  5.0])
PERSP_BACK_LOOK_AT     = np.array([-30.0, 0.0,  2.0])

# ── Single-panel mode (ECP: front view only, full PERSP_H, ego near bottom) ───
# Camera is 15 m behind and 12 m above ego, aimed well ahead → ego projects
# to ~93 % down from the top (near bottom edge). Tighter FOV = more zoom.
PERSP_FOV_SINGLE       = 35.0
PERSP_SINGLE_CAM_POS   = np.array([-15.0, 0.0,  5.0])
PERSP_SINGLE_LOOK_AT   = np.array([ 30.0, 0.0,  2.0])

# 12 edges of an axis-aligned box (indices into the 8-corner array):
#   corners 0-3: top face  (front-L, front-R, rear-R, rear-L)
#   corners 4-7: bottom face (same order)
BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),   # top
    (4, 5), (5, 6), (6, 7), (7, 4),   # bottom
    (0, 4), (1, 5), (2, 6), (3, 7),   # pillars
]

FPS = 20

# ── Camera strip ──────────────────────────────────────────────────────────────
# Default camera sets per dataset (auto-selected from --version).
ECP_CAMERAS      = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT']
NUSCENES_CAMERAS = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                    'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT']
# Index after which a visible group divider is drawn (nuScenes back/front split).
NUSCENES_CAM_DIVIDER_AFTER = 2   # after CAM_BACK_RIGHT, before CAM_FRONT_LEFT
CAM_DIVIDER_W = 6                # px width of the group divider bar

# ── Colours (BGR) ─────────────────────────────────────────────────────────────
BG_COLOR     = (12,  12,  12)
EGO_COLOR    = (255, 255, 255)
AXIS_COLOR   = (55,  55,  55)
INFO_COLOR   = (180, 180, 180)
GT_COLOR     = (40,  40,  220)   # red   — ground truth boxes
PSEUDO_COLOR = (220,  80,  40)   # blue  — pseudo-label boxes


# ── Colour helpers ────────────────────────────────────────────────────────────

POINT_RADIUS = 1   # px half-width of the square dilation kernel for each LiDAR point


def height_colormap(z: np.ndarray) -> np.ndarray:
    """Z (metres) → BGR colour, blue→cyan→green→yellow→red, clipped to [-2, 4] m."""
    t = ((z.clip(-2.0, 4.0) + 2.0) / 6.0)
    out = np.zeros((len(t), 3), np.uint8)

    m = t < 0.25
    s = (t[m] / 0.25 * 255).astype(np.uint8)
    out[m, 0] = 255; out[m, 1] = s

    m = (t >= 0.25) & (t < 0.50)
    s = ((t[m] - 0.25) / 0.25 * 255).astype(np.uint8)
    out[m, 0] = 255 - s; out[m, 1] = 255

    m = (t >= 0.50) & (t < 0.75)
    s = ((t[m] - 0.50) / 0.25 * 255).astype(np.uint8)
    out[m, 1] = 255; out[m, 2] = s

    m = t >= 0.75
    s = ((t[m] - 0.75) / 0.25 * 255).astype(np.uint8)
    out[m, 1] = 255 - s; out[m, 2] = 255

    return out


def paint_points(canvas: np.ndarray, cols: np.ndarray, rows: np.ndarray,
                 colors: np.ndarray, w: int, h: int) -> None:
    """
    Paint projected LiDAR points onto canvas with a square dilation of POINT_RADIUS.
    Much faster than drawing individual cv2.circles — fully vectorised with numpy.
    """
    r = int(POINT_RADIUS)
    # Write the point itself first (fast path for r=0)
    valid = (cols >= 0) & (cols < w) & (rows >= 0) & (rows < h)
    canvas[rows[valid], cols[valid]] = colors[valid]
    if r == 0:
        return
    # Dilate: for each offset in the square kernel, shift and write
    for dr in range(-r, r + 1):
        for dc in range(-r, r + 1):
            r2 = rows + dr;  c2 = cols + dc
            v  = valid & (r2 >= 0) & (r2 < h) & (c2 >= 0) & (c2 < w)
            canvas[r2[v], c2[v]] = colors[v]


# ── Geometry ──────────────────────────────────────────────────────────────────

def global_box_to_ego(box: dict, R_e2g: np.ndarray, t_e2g: np.ndarray) -> tuple:
    """Global-frame box → (center_ego (3,), size [w,l,h], yaw_ego float)."""
    c_g      = np.array(box['translation'])
    q_g      = Quaternion(box['rotation'])
    size     = box['size']
    center_e = R_e2g.T @ (c_g - t_e2g)
    q_ego    = Quaternion(matrix=R_e2g).inverse * q_g
    yaw      = q_ego.yaw_pitch_roll[0]
    return center_e, size, yaw


def obb_corners_3d(center_e: np.ndarray, size: list, yaw: float) -> np.ndarray:
    """
    (8, 3) corners of an OBB in ego frame.
    nuScenes size: [width(Y), length(X), height(Z)].
    Corners 0-3: top face; 4-7: bottom face.
    """
    hw, hl, hh = size[0] / 2, size[1] / 2, size[2] / 2
    c, s = np.cos(yaw), np.sin(yaw)
    loc = np.array([
        [ hl,  hw,  hh], [ hl, -hw,  hh], [-hl, -hw,  hh], [-hl,  hw,  hh],
        [ hl,  hw, -hh], [ hl, -hw, -hh], [-hl, -hw, -hh], [-hl,  hw, -hh],
    ])
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return (R @ loc.T).T + center_e


# ── BEV rendering ─────────────────────────────────────────────────────────────

def _bev_to_px(pts_xy: np.ndarray) -> np.ndarray:
    """(N,2) ego XY → (N,2) BEV canvas [col, row]."""
    col = BEV_W / 2.0 - pts_xy[:, 1] * BEV_SCALE
    row = BEV_H / 2.0 - pts_xy[:, 0] * BEV_SCALE
    return np.stack([col, row], axis=1)


def _draw_bev_boxes(canvas: np.ndarray, boxes: list, color: tuple,
                    R_e2g: np.ndarray, t_e2g: np.ndarray) -> None:
    """Draw flat BEV footprint boxes on canvas in-place."""
    for box in boxes:
        center_e, size, yaw = global_box_to_ego(box, R_e2g, t_e2g)
        if abs(center_e[0]) > RANGE_M + 10 or abs(center_e[1]) > RANGE_M + 10:
            continue
        corners3d  = obb_corners_3d(center_e, size, yaw)
        corners_xy = corners3d[:4, :2]
        pix = _bev_to_px(corners_xy).astype(np.int32)
        pts = pix.reshape(-1, 1, 2)
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.18, canvas, 0.82, 0, canvas)
        cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
        front = ((pix[0] + pix[1]) / 2).astype(int)
        rear  = ((pix[2] + pix[3]) / 2).astype(int)
        cv2.arrowedLine(canvas, tuple(rear), tuple(front), color, 1, tipLength=0.4, line_type=cv2.LINE_AA)


def render_bev(pts_ego: np.ndarray, pseudo_boxes: list, gt_boxes: list,
               R_e2g: np.ndarray, t_e2g: np.ndarray, scene_name: str,
               frame_num: int, total: int, timestamp_us: int,
               overlay: bool = True) -> np.ndarray:
    canvas = np.full((BEV_H, BEV_W, 3), BG_COLOR, dtype=np.uint8)

    # Ego marker + axis arrows
    cx, cy = BEV_W // 2, BEV_H // 2
    fwd_end  = (cx, cy - int(0.12 * BEV_H))
    left_end = (cx - int(0.12 * BEV_W), cy)
    cv2.arrowedLine(canvas, (cx, cy), fwd_end,  AXIS_COLOR, 1, tipLength=0.25, line_type=cv2.LINE_AA)
    cv2.arrowedLine(canvas, (cx, cy), left_end, AXIS_COLOR, 1, tipLength=0.25, line_type=cv2.LINE_AA)
    cv2.putText(canvas, 'fwd',  (cx + 4, fwd_end[1]  - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, AXIS_COLOR, 1)
    cv2.putText(canvas, 'left', (left_end[0] - 28, cy - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, AXIS_COLOR, 1)
    hw_px = int(1.0 * BEV_SCALE); hl_px = int(2.25 * BEV_SCALE)
    cv2.rectangle(canvas, (cx - hw_px, cy - hl_px), (cx + hw_px, cy + hl_px), EGO_COLOR, 1)
    cv2.circle(canvas, (cx, cy), 3, EGO_COLOR, -1)

    # Point cloud (height-coloured, dilated)
    mask = (np.abs(pts_ego[:, 0]) < RANGE_M) & (np.abs(pts_ego[:, 1]) < RANGE_M)
    vis  = pts_ego[mask]
    if len(vis):
        pix    = _bev_to_px(vis[:, :2]).astype(np.int32)
        colors = height_colormap(vis[:, 2])
        paint_points(canvas, pix[:, 0], pix[:, 1], colors, BEV_W, BEV_H)

    # GT boxes (red) then pseudo-label boxes (green) — pseudo drawn on top
    _draw_bev_boxes(canvas, gt_boxes,     GT_COLOR,     R_e2g, t_e2g)
    _draw_bev_boxes(canvas, pseudo_boxes, PSEUDO_COLOR, R_e2g, t_e2g)

    if overlay:
        _draw_overlay(canvas, scene_name, frame_num, total,
                      len(pseudo_boxes), len(gt_boxes), timestamp_us)
        _draw_legend(canvas, BEV_W)
    return canvas


# ── Perspective rendering ─────────────────────────────────────────────────────

def _build_camera_matrix(cam_pos: np.ndarray, look_at: np.ndarray) -> np.ndarray:
    """
    3×3 rotation matrix R that maps ego-frame vectors to camera-frame vectors.
    Camera axes: X=right, Y=camera-up (↑ in image = ↑ in world), Z=forward.
    """
    world_up = np.array([0.0, 0.0, 1.0])
    fwd   = look_at - cam_pos;  fwd   /= np.linalg.norm(fwd)
    right = np.cross(fwd, world_up);  right /= np.linalg.norm(right)
    up    = np.cross(right, fwd)
    return np.stack([right, up, fwd], axis=0)   # (3,3), rows = camera axes


def _project(pts_ego: np.ndarray, R_cam: np.ndarray, cam_pos: np.ndarray,
             w: int, h: int, fov_deg: float) -> tuple:
    """Perspective-project (N,3) ego points → (M,2) pixel coords + boolean mask."""
    pts_cam = (R_cam @ (pts_ego - cam_pos).T).T
    depth   = pts_cam[:, 2]
    valid   = depth > 0.2
    pts_cam = pts_cam[valid];  depth = depth[valid]
    f = h / (2.0 * np.tan(np.radians(fov_deg / 2.0)))
    u = f * pts_cam[:, 0] / depth + w / 2.0
    v = -f * pts_cam[:, 1] / depth + h / 2.0
    return np.stack([u, v], axis=1), valid


def _draw_persp_boxes(canvas: np.ndarray, boxes: list, color: tuple,
                      R_cam: np.ndarray, cam_pos: np.ndarray,
                      R_e2g: np.ndarray, t_e2g: np.ndarray,
                      panel_h: int, fov: float = PERSP_FOV_DUAL) -> None:
    """Draw 3D wireframe boxes projected into a perspective panel, in-place."""
    f = panel_h / (2.0 * np.tan(np.radians(fov / 2.0)))
    for box in boxes:
        center_e, size, yaw = global_box_to_ego(box, R_e2g, t_e2g)
        if np.linalg.norm(center_e[:2]) > PERSP_RANGE_M + 10:
            continue
        corners = obb_corners_3d(center_e, size, yaw)       # (8,3)
        pts_c   = (R_cam @ (corners - cam_pos).T).T
        depths  = pts_c[:, 2]
        if np.any(depths <= 0.2):
            continue
        u = (f * pts_c[:, 0] / depths + PERSP_W   / 2).astype(np.int32)
        v = (-f * pts_c[:, 1] / depths + panel_h  / 2).astype(np.int32)
        for i, j in BOX_EDGES:
            cv2.line(canvas, (int(u[i]), int(v[i])), (int(u[j]), int(v[j])),
                     color, 2, cv2.LINE_AA)


def _render_persp_panel(pts_ego: np.ndarray, pseudo_boxes: list, gt_boxes: list,
                        R_e2g: np.ndarray, t_e2g: np.ndarray,
                        cam_pos: np.ndarray, look_at: np.ndarray,
                        panel_h: int, fov: float,
                        panel_label: str) -> np.ndarray:
    """Render one perspective panel of size (panel_h × PERSP_W)."""
    canvas = np.full((panel_h, PERSP_W, 3), BG_COLOR, dtype=np.uint8)
    R_cam  = _build_camera_matrix(cam_pos, look_at)

    dist    = np.linalg.norm(pts_ego[:, :2], axis=1)
    pts_vis = pts_ego[dist < PERSP_RANGE_M]
    if len(pts_vis):
        pix, valid = _project(pts_vis, R_cam, cam_pos, PERSP_W, panel_h, fov)
        colors = height_colormap(pts_vis[valid, 2])
        paint_points(canvas, pix[:, 0].astype(np.int32), pix[:, 1].astype(np.int32),
                     colors, PERSP_W, panel_h)

    _draw_persp_boxes(canvas, gt_boxes,     GT_COLOR,     R_cam, cam_pos, R_e2g, t_e2g, panel_h, fov)
    _draw_persp_boxes(canvas, pseudo_boxes, PSEUDO_COLOR, R_cam, cam_pos, R_e2g, t_e2g, panel_h, fov)

    # Panel direction label (bottom-left corner)
    font_scale = 1.4
    thickness  = 3
    (tw, th), baseline = cv2.getTextSize(panel_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    cv2.putText(canvas, panel_label, (14, panel_h - baseline - 14),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return canvas


def render_perspective(pts_ego: np.ndarray, pseudo_boxes: list, gt_boxes: list,
                       R_e2g: np.ndarray, t_e2g: np.ndarray, scene_name: str,
                       frame_num: int, total: int, timestamp_us: int,
                       dual_panel: bool = True) -> np.ndarray:
    if dual_panel:
        # nuScenes: front view (top half) + rear view (bottom half)
        front = _render_persp_panel(pts_ego, pseudo_boxes, gt_boxes, R_e2g, t_e2g,
                                    PERSP_FRONT_CAM_POS, PERSP_FRONT_LOOK_AT,
                                    PERSP_PANEL_H, PERSP_FOV_DUAL, 'FRONT')
        rear  = _render_persp_panel(pts_ego, pseudo_boxes, gt_boxes, R_e2g, t_e2g,
                                    PERSP_BACK_CAM_POS, PERSP_BACK_LOOK_AT,
                                    PERSP_PANEL_H, PERSP_FOV_DUAL, 'REAR')
        front[-2:, :] = 28   # 2-px dark separator
        canvas = np.concatenate([front, rear], axis=0)
    else:
        # ECP: single front-only view, full height, ego near bottom edge
        canvas = _render_persp_panel(pts_ego, pseudo_boxes, gt_boxes, R_e2g, t_e2g,
                                     PERSP_SINGLE_CAM_POS, PERSP_SINGLE_LOOK_AT,
                                     PERSP_H, PERSP_FOV_SINGLE, 'FRONT')

    _draw_overlay(canvas, scene_name, frame_num, total,
                  len(pseudo_boxes), len(gt_boxes), timestamp_us)
    _draw_legend(canvas, PERSP_W)
    return canvas


# ── Shared overlay helpers ────────────────────────────────────────────────────

def _draw_overlay(canvas: np.ndarray, scene_name: str, frame_num: int,
                  total: int, n_pseudo: int, n_gt: int, timestamp_us: int) -> None:
    y = 46
    cv2.putText(canvas, scene_name, (14, y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, INFO_COLOR, 2)
    y += 46
    cv2.putText(canvas, f'frame {frame_num + 1:04d} / {total:04d}', (14, y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, INFO_COLOR, 2)
    y += 50
    if n_gt > 0:
        cv2.putText(canvas, f'GT  ({n_gt})', (14, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, GT_COLOR, 2)
        y += 46
    if n_pseudo > 0:
        cv2.putText(canvas, f'Pseudo  ({n_pseudo})', (14, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, PSEUDO_COLOR, 2)


def _draw_legend(canvas: np.ndarray, canvas_w: int) -> None:
    lx, ly = canvas_w - 290, 38
    for label, color in [('ground truth', GT_COLOR), ('pseudo-labels', PSEUDO_COLOR)]:
        cv2.rectangle(canvas, (lx, ly - 18), (lx + 22, ly + 6), color, -1)
        cv2.putText(canvas, label, (lx + 30, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, INFO_COLOR, 2)
        ly += 40


# ── Video I/O ─────────────────────────────────────────────────────────────────

def _open_ffmpeg(out_path: str, w: int, h: int, fps: int) -> subprocess.Popen:
    """
    Open an ffmpeg subprocess writing directly to out_path (must be under $HOME
    so snap ffmpeg's sandbox allows it). Returns a Popen whose stdin accepts
    raw BGR24 frame bytes.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo', '-pix_fmt', 'bgr24',
        '-s', f'{w}x{h}', '-r', str(fps),
        '-i', 'pipe:0',
        '-c:v', 'libx264', '-crf', '18', '-preset', 'fast',
        '-pix_fmt', 'yuv420p',
        out_path,
    ]
    env = os.environ.copy()
    env['DISPLAY'] = ''
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            env=env)
    proc._final_path = out_path
    return proc


def _close_ffmpeg(proc: subprocess.Popen) -> None:
    proc.stdin.close()
    proc.wait()
    size = os.path.getsize(proc._final_path) if os.path.exists(proc._final_path) else -1
    print(f'  ✓  saved → {proc._final_path}  ({size / 1024 / 1024:.1f} MB)')


# ── BEV side-panel helper ─────────────────────────────────────────────────────

BEV_SIDE_W = 900   # BEV side panel width (must be even)

def _attach_bev_panel(persp: np.ndarray, bev: np.ndarray) -> np.ndarray:
    """Hstack perspective with BEV: scale BEV to full panel height (square),
    then crop the centre horizontally to BEV_SIDE_W — no stretching."""
    h = persp.shape[0]
    bev_full = cv2.resize(bev, (h, h), interpolation=cv2.INTER_LINEAR)
    x0 = (h - BEV_SIDE_W) // 2
    bev_cropped = bev_full[:, x0:x0 + BEV_SIDE_W]
    # "BEV" label at top-left of the cropped panel
    (tw, _), _ = cv2.getTextSize('BEV', cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)
    cv2.putText(bev_cropped, 'BEV', (BEV_SIDE_W // 2 - tw // 2, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3, cv2.LINE_AA)
    divider = np.full((h, 4, 3), 40, dtype=np.uint8)
    return np.concatenate([persp, divider, bev_cropped], axis=1)


# ── Split-screen helper ───────────────────────────────────────────────────────

def _make_split(left: np.ndarray, right: np.ndarray,
                left_label: str = 'Ours', right_label: str = 'Compare') -> np.ndarray:
    """Hstack two same-size canvases with a 2-px white divider and panel labels."""
    divider = np.full((left.shape[0], 2, 3), 200, dtype=np.uint8)
    out = np.concatenate([left, divider, right], axis=1)
    # Labels at top-centre of each panel, with white background box
    pad = 10
    for label, x_centre in [(left_label, left.shape[1] // 2),
                             (right_label, left.shape[1] + 2 + right.shape[1] // 2)]:
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)
        tx = x_centre - tw // 2
        ty = 40
        cv2.rectangle(out,
                      (tx - pad, ty - th - pad),
                      (tx + tw + pad, ty + baseline + pad),
                      (255, 255, 255), -1)
        cv2.putText(out, label, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (30, 30, 30), 3, cv2.LINE_AA)
    return out


# ── Camera strip ─────────────────────────────────────────────────────────────

def _camera_strip_height(canvas_w: int, n_cams: int, has_divider: bool) -> int:
    """Compute camera strip height so each tile maintains 16:9 aspect ratio.
    Always returns an even number (libx264 requires even dimensions)."""
    divider_px = CAM_DIVIDER_W if has_divider else 0
    cam_w = (canvas_w - divider_px) // n_cams
    h = cam_w * 9 // 16
    return h if h % 2 == 0 else h - 1


def _build_camera_strip(nusc: NuScenes, sample: dict, cameras: list,
                        total_w: int, strip_h: int,
                        cam_divider_after: int = None) -> np.ndarray:
    """
    Load and tile camera images into a (strip_h, total_w, 3) BGR strip.
    A grey bar of width CAM_DIVIDER_W is inserted after cameras[cam_divider_after].
    """
    n = len(cameras)
    divider_px = CAM_DIVIDER_W if cam_divider_after is not None else 0
    cam_w = (total_w - divider_px) // n

    strip = np.full((strip_h, total_w, 3), 15, dtype=np.uint8)
    x = 0
    for i, cam_name in enumerate(cameras):
        # Group divider between back and front camera groups
        if cam_divider_after is not None and i == cam_divider_after + 1:
            strip[:, x:x + divider_px] = [55, 55, 55]
            x += divider_px

        if cam_name in sample['data']:
            cam_sd   = nusc.get('sample_data', sample['data'][cam_name])
            img_path = os.path.join(nusc.dataroot, cam_sd['filename'])
            img = cv2.imread(img_path)
            if img is not None:
                strip[:, x:x + cam_w] = cv2.resize(img, (cam_w, strip_h))

        # Camera label at bottom-left of each tile
        label = cam_name.replace('CAM_', '').replace('_', ' ')
        cv2.putText(strip, label, (x + 10, strip_h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (60, 60, 220), 3, cv2.LINE_AA)

        # Thin separator between tiles (skip if next tile is the group divider)
        next_is_group = (cam_divider_after is not None and i == cam_divider_after)
        if i < n - 1 and not next_is_group:
            cv2.line(strip, (x + cam_w, 0), (x + cam_w, strip_h), (40, 40, 40), 1)

        x += cam_w

    return strip


# ── Per-scene processing ──────────────────────────────────────────────────────

def collect_scene_lidar_sd(nusc: NuScenes, scene: dict) -> list:
    """
    Return all sample_data records for LIDAR_TOP in this scene, in order.

    NOTE: In ecp2nuscenes every recording frame is stored as a nuScenes
    *sample* (keyframe). There are no separate sweep records. We therefore
    iterate the sample chain and look up the LIDAR_TOP sample_data for each
    sample, rather than following the sample_data `next` chain (which would
    be the right approach for raw nuScenes sweeps).
    """
    first_sample = nusc.get('sample', scene['first_sample_token'])
    if 'LIDAR_TOP' not in first_sample['data']:
        return []

    all_sd = []
    tok = scene['first_sample_token']
    while tok:
        sample = nusc.get('sample', tok)
        if 'LIDAR_TOP' in sample['data']:
            sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
            all_sd.append(sd)
        tok = sample['next']

    return all_sd


def process_scene(
    nusc: NuScenes,
    scene: dict,
    pseudo_labels: dict,
    gt_by_sample: dict,
    out_dir: str,
    fps: int,
    view: str = 'bev',
    annotated_only: bool = False,
    writer: subprocess.Popen = None,
    compare_labels: dict = None,
    left_label: str = 'Ours',
    compare_label: str = 'Compare',
    cameras: list = None,
    cam_divider_after: int = None,
    dual_panel: bool = True,
) -> None:
    scene_name = scene['name']
    print(f'\n── {scene_name} ──')

    all_sd = collect_scene_lidar_sd(nusc, scene)
    if not all_sd:
        print('  No LIDAR_TOP data — skipped.')
        return

    if annotated_only:
        all_sd = [sd for sd in all_sd if sd['sample_token'] in pseudo_labels]
        if not all_sd:
            print('  No annotated frames — skipped.')
            return

    n_anno = sum(1 for sd in all_sd if sd['sample_token'] in pseudo_labels)
    print(f'  {len(all_sd)} frames to render  ({n_anno} with pseudo-labels)')

    # LiDAR ↔ ego calibration — constant for the whole scene
    lidar_cal = nusc.get('calibrated_sensor', all_sd[0]['calibrated_sensor_token'])
    R_l2e = Quaternion(lidar_cal['rotation']).rotation_matrix.astype(np.float64)
    t_l2e = np.array(lidar_cal['translation'], dtype=np.float64)

    total = len(all_sd)

    # Compute canvas width (perspective mode includes BEV side panel)
    base_canvas_w = (PERSP_W + 4 + BEV_SIDE_W) if view == 'perspective' else BEV_W
    full_canvas_w = base_canvas_w * 2 + 2 if compare_labels is not None else base_canvas_w
    cam_strip_h = (
        _camera_strip_height(full_canvas_w, len(cameras), cam_divider_after is not None)
        if cameras else 0
    )

    # If a shared ffmpeg pipe is passed in, use it; otherwise open a per-scene file.
    own_writer = writer is None
    if own_writer:
        os.makedirs(out_dir, exist_ok=True)
        suffix   = 'annotated_' if annotated_only else ''
        cmp_tag  = '_vs' if compare_labels is not None else ''
        out_path = os.path.join(out_dir, f'{scene_name}_{suffix}{view}{cmp_tag}.mp4')
        canvas_h = PERSP_H if view == 'perspective' else BEV_H
        writer   = _open_ffmpeg(out_path, full_canvas_w, canvas_h + cam_strip_h, fps)

    for frame_num, sd in enumerate(all_sd):
        lidar_path = os.path.join(nusc.dataroot, sd['filename'])
        if not os.path.exists(lidar_path):
            print(f'  [WARN] missing {lidar_path}', flush=True)
            continue

        try:
            raw = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)[:, :3]
        except Exception as exc:
            print(f'  [WARN] {exc}', flush=True)
            continue

        pts_ego = (R_l2e @ raw.astype(np.float64).T).T + t_l2e

        ego_pose = nusc.get('ego_pose', sd['ego_pose_token'])
        R_e2g    = Quaternion(ego_pose['rotation']).rotation_matrix.astype(np.float64)
        t_e2g    = np.array(ego_pose['translation'], dtype=np.float64)

        pseudo_boxes = pseudo_labels.get(sd['sample_token'], [])
        gt_boxes     = gt_by_sample.get(sd['sample_token'], [])

        if view == 'perspective':
            persp_fn = lambda *a, **k: render_perspective(*a, dual_panel=dual_panel, **k)
            bev_fn   = render_bev
            def render_fn(*a, **k):
                persp  = persp_fn(*a, **k)
                bev    = bev_fn(*a, overlay=False, **k)
                return _attach_bev_panel(persp, bev)
        else:
            render_fn = render_bev

        canvas = render_fn(pts_ego, pseudo_boxes, gt_boxes, R_e2g, t_e2g,
                           scene_name, frame_num, total, sd['timestamp'])

        if compare_labels is not None:
            cmp_boxes = compare_labels.get(sd['sample_token'], [])
            right = render_fn(pts_ego, cmp_boxes, gt_boxes, R_e2g, t_e2g,
                              scene_name, frame_num, total, sd['timestamp'])
            canvas = _make_split(canvas, right, left_label, compare_label)

        if cameras and cam_strip_h > 0:
            sample     = nusc.get('sample', sd['sample_token'])
            cam_strip  = _build_camera_strip(nusc, sample, cameras,
                                             canvas.shape[1], cam_strip_h,
                                             cam_divider_after)
            canvas = np.concatenate([canvas, cam_strip], axis=0)

        try:
            writer.stdin.write(canvas.tobytes())
        except BrokenPipeError:
            print(f'  [ERROR] ffmpeg pipe broke on frame {frame_num}. '
                  f'ffmpeg exit code: {writer.poll()}', flush=True)
            raise

        if (frame_num + 1) % 100 == 0 or frame_num == 0:
            print(f'  {frame_num + 1}/{total} frames rendered …', flush=True)

    if own_writer:
        _close_ffmpeg(writer)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',  default=DATA_ROOT,
                        help='ecp2nuscenes root directory')
    parser.add_argument('--version',    default=VERSION)
    parser.add_argument('--label-json', default=LABEL_JSON,
                        help='autolabel_ecp_annotated_8class.json path')
    parser.add_argument('--out-dir',    default=OUT_DIR,
                        help='output directory for MP4 files')
    parser.add_argument('--scene',      default=None,
                        help='process one scene by name (default: all with labels)')
    parser.add_argument('--all-scenes', action='store_true',
                        help='process ALL scenes (not just those with pseudo-labels)')
    parser.add_argument('--fps', type=int, default=None,
                        help='video frame rate (default: 1 for --annotated-only, 20 otherwise)')
    parser.add_argument('--view', choices=['bev', 'perspective'], default='bev',
                        help='bev: top-down flat boxes; perspective: 3D wireframe (default: bev)')
    parser.add_argument('--annotated-only', action='store_true',
                        help='only render frames that have pseudo-labels (default fps: 1)')
    parser.add_argument('--combine', action='store_true',
                        help='write all scenes into a single video file instead of one per scene')
    parser.add_argument('--compare-dir', default=None,
                        help='directory of per-scene comparison outputs (e.g. ECP VESPA) to show '
                             'in a right split-screen panel; expects '
                             '{dir}/*/#out_labels/*/ *_all_8class.json structure')
    parser.add_argument('--compare-json', default=None,
                        help='single JSON file with comparison labels (e.g. nuScenes VESPA); '
                             'alternative to --compare-dir for flat output structures')
    parser.add_argument('--left-label',    default='Ours',
                        help='label shown above the left panel in split-screen mode')
    parser.add_argument('--compare-label', default='VESPA',
                        help='label shown above the right panel in split-screen mode')
    parser.add_argument('--cameras', default=None,
                        help='comma-separated camera names to show below LiDAR '
                             '(auto-detected from --version if not set; '
                             'e.g. CAM_FRONT_LEFT,CAM_FRONT,CAM_FRONT_RIGHT)')
    parser.add_argument('--no-cameras', action='store_true',
                        help='disable the camera image strip below the LiDAR view')
    args = parser.parse_args()

    # FPS: if user didn't specify, default to 1 for annotated-only (1 sec/frame) else 20
    FPS = args.fps if args.fps is not None else (1 if args.annotated_only else 20)

    # Camera strip: auto-detect from version, allow override or disable
    is_nuscenes_mini = 'mini' in args.version
    if args.no_cameras:
        cameras, cam_divider_after = [], None
    elif args.cameras:
        cameras = [c.strip() for c in args.cameras.split(',')]
        cam_divider_after = None
    elif is_nuscenes_mini:
        cameras, cam_divider_after = NUSCENES_CAMERAS, NUSCENES_CAM_DIVIDER_AFTER
    else:  # ECP (v1.0-trainval)
        cameras, cam_divider_after = ECP_CAMERAS, None

    # Perspective mode: nuScenes mini → dual panel (front+rear), ECP → single front panel
    dual_panel = is_nuscenes_mini

    print('Loading NuScenes …')
    nusc = NuScenes(version=args.version, dataroot=args.data_root, verbose=False)
    print(f'  {len(nusc.scene)} scene(s), {len(nusc.sample)} keyframe(s)')

    print(f'Loading pseudo-labels from\n  {args.label_json}')
    with open(args.label_json) as f:
        label_data = json.load(f)
    pseudo_labels: dict = label_data['results']
    print(f'  {len(pseudo_labels)} sample token(s) with pseudo-labels')

    # Build GT lookup: sample_token → list of annotation dicts (nuScenes format)
    # Each annotation has translation, size, rotation — same convention as pseudo-labels.
    gt_by_sample: dict = {}
    for ann in nusc.sample_annotation:
        tok = ann['sample_token']
        if tok not in gt_by_sample:
            gt_by_sample[tok] = []
        gt_by_sample[tok].append(ann)
    print(f'  {len(gt_by_sample)} sample token(s) with ground-truth annotations')

    # Load comparison labels (e.g. VESPA) if requested
    compare_labels = None
    if args.compare_json:
        with open(args.compare_json) as fh:
            d = json.load(fh)
        compare_labels = d.get('results', {})
        print(f'  Comparison labels: {len(compare_labels)} token(s) from {args.compare_json}')
    elif args.compare_dir:
        import glob as _glob
        compare_labels = {}
        pattern = os.path.join(args.compare_dir, '*', '#out_labels', '*', '*_all_8class.json')
        found = _glob.glob(pattern)
        if not found:
            print(f'  [WARN] --compare-dir: no *_all_8class.json files matched under {args.compare_dir}')
        for f in found:
            with open(f) as fh:
                d = json.load(fh)
            compare_labels.update(d.get('results', {}))
        print(f'  Comparison labels: {len(compare_labels)} token(s) from {len(found)} file(s)')

    # Determine which scenes to process
    if args.scene:
        scenes = [s for s in nusc.scene if s['name'] == args.scene]
        if not scenes:
            sys.exit(f'Scene not found: {args.scene}')
    elif args.all_scenes:
        scenes = nusc.scene
    else:
        # Default: only scenes that contain at least one pseudo-labelled frame
        anno_tokens = set(pseudo_labels.keys())
        scenes = []
        for scene in nusc.scene:
            tok = scene['first_sample_token']
            while tok:
                if tok in anno_tokens:
                    scenes.append(scene)
                    break
                tok = nusc.get('sample', tok)['next']
        print(f'  {len(scenes)} scene(s) contain pseudo-labels (use --all-scenes for all)')

    # Common kwargs forwarded to every process_scene call
    scene_kwargs = dict(
        fps=FPS, view=args.view, annotated_only=args.annotated_only,
        compare_labels=compare_labels,
        left_label=args.left_label, compare_label=args.compare_label,
        cameras=cameras, cam_divider_after=cam_divider_after,
        dual_panel=dual_panel,
    )

    if args.combine:
        os.makedirs(args.out_dir, exist_ok=True)
        suffix   = 'annotated_' if args.annotated_only else ''
        cmp_tag  = '_vs' if compare_labels is not None else ''
        out_path = os.path.join(args.out_dir, f'all_scenes_{suffix}{args.view}{cmp_tag}.mp4')
        base_w   = (PERSP_W + 4 + BEV_SIDE_W) if args.view == 'perspective' else BEV_W
        canvas_w = base_w * 2 + 2 if compare_labels is not None else base_w
        canvas_h = PERSP_H if args.view == 'perspective' else BEV_H
        cam_strip_h = (
            _camera_strip_height(canvas_w, len(cameras), cam_divider_after is not None)
            if cameras else 0
        )
        shared_writer = _open_ffmpeg(out_path, canvas_w, canvas_h + cam_strip_h, FPS)
        print(f'  Combined output → {out_path}')
        for scene in scenes:
            process_scene(nusc, scene, pseudo_labels, gt_by_sample, args.out_dir,
                          writer=shared_writer, **scene_kwargs)
        _close_ffmpeg(shared_writer)
    else:
        for scene in scenes:
            process_scene(nusc, scene, pseudo_labels, gt_by_sample, args.out_dir,
                          **scene_kwargs)

    print('\nAll done.')


if __name__ == '__main__':
    main()
