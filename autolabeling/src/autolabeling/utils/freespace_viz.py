"""
Plots for the mask / free-space pilot: occupancy maps (BEV and along-the-ray side view), image overlays.

Colours: free = green, unknown = light grey, occupied = black. In a slice, a cell is occupied if ANY voxel of the slab is
occupied, else free if any is free, else unknown (occupied wins, so thin surfaces do not vanish).
"""
from typing import Iterable, Optional, Sequence

import numpy as np

from . import freespace as fs

FREE_C, UNK_C, OCC_C = (0.72, 0.90, 0.72), (0.93, 0.93, 0.93), (0.0, 0.0, 0.0)
_CMAP = np.array([FREE_C, UNK_C, OCC_C])       # index = state + 1


def box_corners(center, length, width, height, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    lx = np.array([1, 1, -1, -1, 1, 1, -1, -1]) * length / 2
    ly = np.array([1, -1, -1, 1, 1, -1, -1, 1]) * width / 2
    lz = np.array([-1, -1, -1, -1, 1, 1, 1, 1]) * height / 2
    return np.stack([c * lx - s * ly, s * lx + c * ly, lz], axis=1) + np.asarray(center, np.float64)


def slice_states(grid: fs.FreeSpaceGrid, origin, e_u, e_v, e_w, u_range, v_range, w_range, res=0.1):
    """
    2D map of the grid: plane spanned by unit vectors e_u, e_v through `origin`, collapsed over the slab w_range
    along e_w. Returns (H, W) int8 of FREE/UNKNOWN/OCC with v as rows (bottom row = v_range[0]) and the extent.
    """
    us = np.arange(u_range[0], u_range[1], res) + res / 2
    vs_ = np.arange(v_range[0], v_range[1], res) + res / 2
    ws = np.arange(w_range[0], w_range[1] + 1e-9, grid.vs)
    U, V = np.meshgrid(us, vs_)
    base = np.asarray(origin, float)[None, None, :] + U[..., None] * e_u + V[..., None] * e_v
    out = np.zeros(U.shape, np.int8)
    for w in ws:
        st = grid.state((base + w * e_w).reshape(-1, 3)).reshape(U.shape)
        out = np.where(st == fs.OCC, fs.OCC, np.where((st == fs.FREE) & (out != fs.OCC), fs.FREE, out))
    return out, (u_range[0], u_range[1], v_range[0], v_range[1])


def _show(ax, states, extent):
    ax.imshow(_CMAP[states + 1], origin='lower', extent=extent, interpolation='nearest', aspect='equal')


def _mesh_colors(grid, P, d_min=0.2):
    """
    0 red    : in free space and >= d_min in front of the nearest return (a real violation)
    1 orange : in free space but < d_min in front of a return (grazing / noise / discretisation, not counted)
    2 blue   : in unknown space (never touched by a beam, or hidden behind a return)
    3 purple : on an occupied voxel (a beam ended there)
    """
    st = grid.state(P)
    free = st == fs.FREE
    dist = np.full(len(P), np.nan)
    if free.any():
        dist[free] = grid.dist_to_occupied(P[free])
    viol = free & (np.nan_to_num(dist, nan=-1) >= d_min)
    return np.where(viol, 0, np.where(free, 1, np.where(st == fs.OCC, 3, 2)))


_MESH_CLASSES = (('red', 'mesh in free space (>= {d} m)'), ('orange', 'mesh in free space (< {d} m)'),
                 ('royalblue', 'mesh in unknown space'), ('purple', 'mesh on occupied voxel'))


def scatter_mesh(ax, x, y, col, s=2, d_min=0.2):
    for k, (c, lab) in enumerate(_MESH_CLASSES):
        m = col == k
        if m.any():
            ax.scatter(x[m], y[m], s=s, c=c, alpha=0.7, label=f'{lab.format(d=d_min)}: {m.sum()}', linewidths=0)


def bev_panel(ax, grid, center, mesh_pts, lidar_pts, boxes: Sequence[dict], half=6.0, z_band=None, d_min=0.2, title=''):
    """
    Top-down (x forward = up, y left = left). z_band = (z_lo, z_hi) collapsed for the occupancy colours, default the
    object's height range. boxes: dicts(corners=(8,3), color, label, ls).
    """
    cx, cy, cz = center
    z_lo, z_hi = z_band if z_band is not None else (cz - 1.0, cz + 1.0)
    st, ext = slice_states(grid, (0, 0, 0), np.array([0, -1.0, 0]), np.array([1.0, 0, 0]), np.array([0, 0, 1.0]),
                           (-cy - half, -cy + half), (cx - half, cx + half), (z_lo, z_hi))
    _show(ax, st, ext)
    m = (np.abs(lidar_pts[:, 0] - cx) < half) & (np.abs(lidar_pts[:, 1] - cy) < half) & \
        (lidar_pts[:, 2] > z_lo) & (lidar_pts[:, 2] < z_hi)
    ax.scatter(-lidar_pts[m, 1], lidar_pts[m, 0], s=4, c='white', edgecolors='k', linewidths=0.3, label='LiDAR')
    col = _mesh_colors(grid, mesh_pts, d_min)
    scatter_mesh(ax, -mesh_pts[:, 1], mesh_pts[:, 0], col, d_min=d_min)
    for b in boxes:
        c = b['corners']
        poly = np.vstack([c[:4], c[:1]])
        ax.plot(-poly[:, 1], poly[:, 0], color=b['color'], ls=b.get('ls', '--'), lw=1.6, label=b['label'])
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ax.set_xlabel('-y  [m]  (left is left)'); ax.set_ylabel('x forward [m]'); ax.set_title(title, fontsize=9)


def side_panel(ax, grid, center, mesh_pts, lidar_pts, boxes: Sequence[dict], half=6.0, lateral=0.6, d_min=0.2, title=''):
    """
    Side view in the RAY frame: horizontal = distance from the LiDAR along the ray to the object (camera looks this way),
    vertical = z. Slab of +-`lateral` metres across the ray. Overshoot towards the sensor shows up on the left.
    """
    o = grid.sensor
    d = np.asarray(center, float)[:2] - o[:2]
    r = np.linalg.norm(d)
    u = np.array([d[0] / r, d[1] / r, 0.0])
    n = np.array([-u[1], u[0], 0.0])
    z = np.array([0, 0, 1.0])
    st, ext = slice_states(grid, o, u, z, n, (r - half, r + half), (center[2] - o[2] - half / 2, center[2] - o[2] + half / 2),
                           (-lateral, lateral))
    _show(ax, st, ext)

    def to_uv(P):
        Q = np.asarray(P, float) - o
        return Q @ u, Q[:, 2], Q @ n

    su, sz, sn = to_uv(lidar_pts)
    m = (np.abs(sn) < lateral) & (np.abs(su - r) < half)
    ax.scatter(su[m], sz[m], s=5, c='white', edgecolors='k', linewidths=0.3, label='LiDAR (slab)')
    mu, mz, mn = to_uv(mesh_pts)
    slab = np.abs(mn) < lateral
    col = _mesh_colors(grid, mesh_pts[slab], d_min)
    scatter_mesh(ax, mu[slab], mz[slab], col, d_min=d_min)
    for b in boxes:
        bu, bz, _ = to_uv(b['corners'])
        x0, x1, z0, z1 = bu.min(), bu.max(), bz.min(), bz.max()
        ax.plot([x0, x1, x1, x0, x0], [z0, z0, z1, z1, z0], color=b['color'], ls=b.get('ls', '--'), lw=1.6, label=b['label'])
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ax.set_xlabel('distance from LiDAR along the ray [m]  (sensor is to the left)'); ax.set_ylabel('z [m]')
    ax.set_title(title, fontsize=9)


def scene_bev(ax, grid, lidar_pts, gt_corners: Sequence[np.ndarray], obj_boxes: Sequence[dict], half=40.0, z_band=(0.3, 3.0)):
    """Whole-scene occupancy map (BEV) with LiDAR, GT boxes (green dashed) and predicted OBBs (coloured, numbered)."""
    st, ext = slice_states(grid, (0, 0, 0), np.array([0, -1.0, 0]), np.array([1.0, 0, 0]), np.array([0, 0, 1.0]),
                           (-half, half), (-half / 2, half), z_band, res=0.2)
    _show(ax, st, ext)
    m = (lidar_pts[:, 2] > z_band[0]) & (lidar_pts[:, 2] < z_band[1])
    ax.scatter(-lidar_pts[m, 1], lidar_pts[m, 0], s=1, c='w', alpha=0.6, linewidths=0)
    for c in gt_corners:
        poly = np.vstack([c[:4], c[:1]])
        ax.plot(-poly[:, 1], poly[:, 0], color='green', ls='--', lw=1.0)
    for b in obj_boxes:
        c = b['corners']
        poly = np.vstack([c[:4], c[:1]])
        ax.plot(-poly[:, 1], poly[:, 0], color=b['color'], lw=1.6)
        ax.text(-c[:, 1].mean(), c[:, 0].mean(), str(b['label']), color='red', fontsize=9, fontweight='bold', ha='center')
    ax.plot(-grid.sensor[1], grid.sensor[0], 'r^', ms=8)
    ax.set_xlim(-half, half); ax.set_ylim(-half / 2, half)
    ax.set_xlabel('-y [m]'); ax.set_ylabel('x forward [m]')


def image_overlay(ax, img, mask, sil, occluders=None, title='', boxes_px: Sequence[dict] = ()):
    """Image with the SAM3 mask contour (cyan) and the mesh silhouette contour (magenta); mesh pixels outside the
    mask are tinted red (leak), mask pixels the mesh misses are tinted blue."""
    import cv2
    out = img.astype(np.float32).copy()
    leak = sil & ~mask
    if occluders is not None:
        leak &= ~occluders
    miss = mask & ~sil
    out[leak] = 0.5 * out[leak] + 0.5 * np.array([255, 0, 0])
    out[miss] = 0.5 * out[miss] + 0.5 * np.array([0, 80, 255])
    out = out.astype(np.uint8)
    for m, col in ((mask, (0, 255, 255)), (sil, (255, 0, 255))):
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, cs, -1, col, 2)
    ys, xs = np.where(mask | sil)
    pad = 40
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, img.shape[1])
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, img.shape[0])
    ax.imshow(out); ax.axis('off'); ax.set_title(title, fontsize=9)
    for b in boxes_px:
        draw_box_px(ax, b['px'], b['color'], b.get('label'))
    ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)


def scene3d(grid, center, mesh_pts, boxes: Sequence[dict], half=6.0, z_half=2.5, d_min=0.2, max_free=12000, max_occ=25000,
            title='', seed=0):
    """
    Interactive 3D view (plotly) around one object: occupied voxels (black), a random subset of the free voxels (green fog,
    they are far too many to draw all), the mesh surface (red = in free space >= d_min in front of the nearest return,
    orange = free but closer, blue = elsewhere), boxes as wireframes, and the LiDAR position. Drag to rotate.
    """
    import plotly.graph_objects as go
    rng = np.random.default_rng(seed)
    c = np.asarray(center, float)
    lo_c, hi_c = c - np.array([half, half, z_half]), c + np.array([half, half, z_half])
    i0 = np.clip(np.floor((lo_c - grid.origin) / grid.vs).astype(int), 0, np.array(grid.shape) - 1)
    i1 = np.clip(np.ceil((hi_c - grid.origin) / grid.vs).astype(int), 1, np.array(grid.shape))
    sub = grid.lo[i0[0]:i1[0], i0[1]:i1[1], i0[2]:i1[2]]

    def centres(mask, cap):
        idx = np.argwhere(mask)
        if len(idx) > cap:
            idx = idx[rng.choice(len(idx), cap, replace=False)]
        return grid.origin + (idx + i0 + 0.5) * grid.vs

    occ = centres(sub > fs.THR_OCC, max_occ)
    free = centres(sub < fs.THR_FREE, max_free)
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=free[:, 0], y=free[:, 1], z=free[:, 2], mode='markers', name='free (subset)',
                               marker=dict(size=1.5, color='rgb(120,200,120)', opacity=0.12)))
    fig.add_trace(go.Scatter3d(x=occ[:, 0], y=occ[:, 1], z=occ[:, 2], mode='markers', name='occupied',
                               marker=dict(size=2, color='black', opacity=0.9)))
    keep = np.all(np.abs(mesh_pts - c) < np.array([half, half, z_half]), axis=1)
    P = mesh_pts[keep]
    if len(P) > 15000:
        P = P[rng.choice(len(P), 15000, replace=False)]
    col = _mesh_colors(grid, P, d_min)
    for k, (cc, lab) in enumerate(_MESH_CLASSES):
        m = col == k
        if m.any():
            fig.add_trace(go.Scatter3d(x=P[m, 0], y=P[m, 1], z=P[m, 2], mode='markers', name=f'{lab.format(d=d_min)}: {int(m.sum())}',
                                       marker=dict(size=2, color=cc, opacity=0.8)))
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    for b in boxes:
        cr = b['corners']
        xs, ys, zs = [], [], []
        for a, e in edges:
            xs += [cr[a, 0], cr[e, 0], None]; ys += [cr[a, 1], cr[e, 1], None]; zs += [cr[a, 2], cr[e, 2], None]
        fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode='lines', name=b['label'], line=dict(color=b['color'], width=5)))
    s = grid.sensor
    fig.add_trace(go.Scatter3d(x=[s[0]], y=[s[1]], z=[s[2]], mode='markers', name='LiDAR', marker=dict(size=6, color='red', symbol='diamond')))
    fig.update_layout(title=title, scene=dict(aspectmode='data', xaxis_title='x forward [m]', yaxis_title='y left [m]', zaxis_title='z [m]'),
                      height=650, margin=dict(l=0, r=0, t=40, b=0), legend=dict(font=dict(size=10)))
    return fig


_BOX_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def project_corners(frame, corners_ego: np.ndarray) -> np.ndarray:
    """(8, 3) ego-frame corners -> (8, 3) with pixel u, v and camera depth z (z <= 0: behind the camera)."""
    pc = (frame.R_c2e.T @ (np.asarray(corners_ego, np.float64) - frame.t_c2e).T).T
    z = pc[:, 2]
    zs = np.where(np.abs(z) < 1e-6, 1e-6, z)
    return np.stack([pc[:, 0] / zs * frame.K[0, 0] + frame.K[0, 2], pc[:, 1] / zs * frame.K[1, 1] + frame.K[1, 2], z], axis=1)


def draw_box_px(ax, px: np.ndarray, color, label=None, lw=1.6):
    """Wireframe of a projected box; edges with an endpoint behind the camera are skipped."""
    first = True
    for a, b in _BOX_EDGES:
        if px[a, 2] <= 0.1 or px[b, 2] <= 0.1:
            continue
        ax.plot([px[a, 0], px[b, 0]], [px[a, 1], px[b, 1]], color=color, lw=lw, label=label if first else None)
        first = False


def boxes_on_image(ax, img, frame, gt_corners: Sequence[np.ndarray], obj_boxes: Sequence[dict], title=''):
    """
    Calibration check: GT boxes (green) and the OBBs of the SAM3D meshes (orange, numbered) drawn into the image with the
    same intrinsics / extrinsics as everything else. If the GT boxes do not sit on the real objects here, the
    camera-LiDAR calibration is off and mask-vs-mesh comparisons cannot be trusted.
    """
    ax.imshow(img); ax.axis('off')
    for c in gt_corners:
        draw_box_px(ax, project_corners(frame, c), 'lime', lw=1.4)
    for b in obj_boxes:
        px = project_corners(frame, b['corners'])
        draw_box_px(ax, px, 'orange', lw=1.4)
        if (px[:, 2] > 0.1).all():
            ax.text(px[:, 0].mean(), px[:, 1].min() - 6, str(b['label']), color='yellow', fontsize=11, fontweight='bold', ha='center',
                    bbox=dict(boxstyle='round,pad=0.15', fc='black', ec='none', alpha=0.6))
    ax.set_xlim(0, img.shape[1]); ax.set_ylim(img.shape[0], 0)
    ax.set_title(title, fontsize=9)
