"""
Mesh <-> 2D mask consistency (analysis of SAM3D Objects output).

render_mesh   : silhouette + depth of a camera-frame mesh (PyTorch3D rasteriser, standard camera model with K)
mask_agreement: how well the silhouette matches the SAM3 mask (recall / leak / IoU, occluder-aware leak)
depth_residuals: mesh front surface vs LiDAR returns inside the mask

Meshes are in the OpenCV camera frame (x right, y down, z forward) = what `SAM3DObjectsModel._mesh_to_r3` returns.
"""
from typing import Dict, Optional, Tuple

import numpy as np


def render_mesh(verts_cam: np.ndarray, faces: np.ndarray, K: np.ndarray, H: int, W: int,
                device: str = 'cpu', backend: str = 'auto') -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (sil (H, W) bool, depth (H, W) float32 camera-Z of the nearest surface, NaN where empty).
    Faces (partly) behind the camera (z <= 0.05) are dropped.
    backend 'pytorch3d' = exact z-buffer (fast on a GPU, hopelessly slow on a CPU for a 300k-face mesh);
    'painter' = triangles drawn far to near with OpenCV, depth = triangle mean (fine for dense meshes, a few seconds on a CPU);
    'auto' = pytorch3d if `device` is cuda, else painter.
    """
    if backend == 'auto':
        backend = 'pytorch3d' if str(device).startswith('cuda') else 'painter'
    if backend == 'painter':
        return _render_painter(verts_cam, faces, K, H, W)
    try:
        return _render_pytorch3d(verts_cam, faces, K, H, W, device)
    except RuntimeError as e:                    # e.g. a GPU the installed CUDA build has no kernels for
        if backend != 'auto_fallback' and not str(device).startswith('cuda'):
            raise
        print(f'  [warn] pytorch3d rendering on {device} failed ({str(e).splitlines()[0][:80]}); using the OpenCV painter')
        return _render_painter(verts_cam, faces, K, H, W)


def _render_painter(verts_cam, faces, K, H, W):
    import cv2
    v = np.asarray(verts_cam, np.float64)
    f = np.asarray(faces, np.int64)
    z = v[:, 2]
    f = f[(z[f] > 0.05).all(axis=1)]
    sil = np.zeros((H, W), np.uint8)
    depth = np.zeros((H, W), np.float32)
    if len(f) == 0:
        return sil.astype(bool), np.full((H, W), np.nan, np.float32)
    u = K[0, 0] * v[:, 0] / z + K[0, 2]
    w = K[1, 1] * v[:, 1] / z + K[1, 2]
    tri = np.stack([u[f], w[f]], axis=-1)                                   # (F, 3, 2)
    on = ~((tri[..., 0].max(1) < 0) | (tri[..., 0].min(1) > W) | (tri[..., 1].max(1) < 0) | (tri[..., 1].min(1) > H))
    tri, zt = tri[on], z[f][on].mean(axis=1)
    order = np.argsort(-zt)                                                  # far first, near overwrites
    P = np.clip(np.round(tri[order] * 16), -1e6, 1e6).astype(np.int32)       # 4 fractional bits
    for pts, zz in zip(P, zt[order]):
        cv2.fillConvexPoly(sil, pts, 1, lineType=cv2.LINE_8, shift=4)
        cv2.fillConvexPoly(depth, pts, float(zz), lineType=cv2.LINE_8, shift=4)
    sil = sil.astype(bool)
    return sil, np.where(sil, depth, np.nan).astype(np.float32)


def _render_pytorch3d(verts_cam, faces, K, H, W, device):
    import torch
    from pytorch3d.renderer import MeshRasterizer, PerspectiveCameras, RasterizationSettings
    from pytorch3d.structures import Meshes

    v = torch.as_tensor(np.asarray(verts_cam, np.float32), device=device)
    f = torch.as_tensor(np.asarray(faces, np.int64), device=device)
    keep = (v[f][:, :, 2] > 0.05).all(dim=1)              # faces fully in front of the camera
    f = f[keep]
    if len(f) == 0:
        return np.zeros((H, W), bool), np.full((H, W), np.nan, np.float32)
    v_p3d = torch.stack([-v[:, 0], -v[:, 1], v[:, 2]], dim=1)      # PyTorch3D camera: +x left, +y up, +z forward
    cams = PerspectiveCameras(
        focal_length=torch.tensor([[K[0, 0], K[1, 1]]], dtype=torch.float32, device=device),
        principal_point=torch.tensor([[K[0, 2], K[1, 2]]], dtype=torch.float32, device=device),
        image_size=torch.tensor([[H, W]], dtype=torch.float32, device=device),
        in_ndc=False, device=device)
    settings = RasterizationSettings(image_size=(H, W), blur_radius=0.0, faces_per_pixel=1, cull_backfaces=False,
                                     bin_size=0 if str(device) == 'cpu' else None, max_faces_per_bin=max(10000, len(f)))
    frag = MeshRasterizer(cameras=cams, raster_settings=settings)(Meshes(verts=[v_p3d], faces=[f]))
    z = frag.zbuf[0, :, :, 0].cpu().numpy()
    sil = z > 0
    return sil, np.where(sil, z, np.nan).astype(np.float32)


def mask_agreement(sil: np.ndarray, mask: np.ndarray, occluders: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    recall    : share of the mask covered by the mesh. Occlusion cannot explain a miss here (mask pixels are visible object).
    leak      : share of the mesh silhouette outside the mask.
    leak_free : the part of the leak NOT behind another detection's mask (`occluders`, bool image) = unexplained.
    iou       : |sil & mask| / |sil | mask|.
    touches_border : mask touches the image edge (truncated object: silhouette comparison is less meaningful).
    """
    sil = sil.astype(bool)
    mask = mask.astype(bool)
    inter = int((sil & mask).sum())
    out = dict(mask_px=int(mask.sum()), mesh_px=int(sil.sum()),
               recall=inter / max(int(mask.sum()), 1),
               leak=int((sil & ~mask).sum()) / max(int(sil.sum()), 1),
               iou=inter / max(int((sil | mask).sum()), 1),
               touches_border=bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()))
    occ = np.zeros_like(mask) if occluders is None else occluders.astype(bool)
    out['leak_free'] = int((sil & ~mask & ~occ).sum()) / max(int(sil.sum()), 1)
    return out


def depth_residuals(depth_mesh: np.ndarray, mask: np.ndarray, u: np.ndarray, v: np.ndarray, z: np.ndarray) -> Dict[str, float]:
    """
    Residual z_lidar - z_mesh_front at the pixels of LiDAR returns inside the mask (u, v pixel, z camera depth).
    > 0: the mesh surface is in FRONT of the return (mesh nearer to the camera than the measured surface).
    < 0: the return is in front of the mesh (mesh reaches less far towards the camera).
    Returns with no mesh at their pixel are counted in `n_uncovered`.
    """
    H, W = mask.shape
    ui, vi = np.round(u).astype(int), np.round(v).astype(int)
    ok = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    ui, vi, z = ui[ok], vi[ok], z[ok]
    inm = mask[vi, ui]
    ui, vi, z = ui[inm], vi[inm], z[inm]
    zm = depth_mesh[vi, ui]
    cov = np.isfinite(zm)
    res = (z - zm)[cov]
    out = dict(n_in_mask=int(len(z)), n_uncovered=int((~cov).sum()))
    if len(res):
        out.update(median=float(np.median(res)), p25=float(np.percentile(res, 25)), p75=float(np.percentile(res, 75)),
                   frac_within_0p3=float((np.abs(res) <= 0.3).mean()))
    else:
        out.update(median=float('nan'), p25=float('nan'), p75=float('nan'), frac_within_0p3=float('nan'))
    return out
