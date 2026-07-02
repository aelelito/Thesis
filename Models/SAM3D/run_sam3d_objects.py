"""
SAM-3D-Objects inference script for nuScenes vehicle reconstruction.

Mirrors run_sam3d_bodies.py in structure and output format.
Processes CAM_FRONT images and outputs per frame (using ECP intrinsics):

  frame{N}_mesh.jpg  — mesh vertices projected onto the image
                        (analogue of _mesh.jpg in run_sam3d_bodies.py)
  frame{N}_bbox.jpg  — axis-aligned 3D bounding box wireframe projected
                        (analogue of _bbox_orientation.jpg; orientation
                         logic is commented out and left for future work)

And per object:
  mesh.npz, bbox_3d.npy, pose.npz, metadata.json

Orientation note (TODO):
  run_sam3d_bodies.py derives forward direction from the shoulder line,
  disambiguated by the nose.  For vehicles we have no skeleton — orientation
  via PCA on the horizontal Gaussian spread is implemented below but
  commented out.  For now a simple axis-aligned bbox is used instead.

Run with:
  conda activate sam3d-objects
  python /home/lleba/Thesis/Development/SAM3D/run_sam3d_objects.py
"""

from __future__ import annotations

import os
import sys
import json
import numpy as np
import torch
from pathlib import Path
from PIL import Image
import cv2
from tqdm import tqdm

# ── Configuration ──────────────────────────────────────────────────────────────

SAM3D_REPO  = Path(__file__).parent / "sam-3d-objects"
CKPT_CONFIG = SAM3D_REPO / "checkpoints" / "hf" / "pipeline.yaml"
DATA_ROOT   = "/media/lleba/ECP_Nuscenes_01/output2/ecp2nuscenes"

NUSCENES_SCENE_IDX = 10

# Directory that holds all SAM3 .npy files for this scene/camera
SAM3_DIR          = Path("/media/lleba/ECP_Nuscenes_01/SAM3_Visualizations/Strassbourg")
SAM3_SCENE_PREFIX = f"scene_{NUSCENES_SCENE_IDX}_cam_0_sam3_outputs__"

# Vehicle categories to process — each maps to one .npy file.
# Extend this list when masks for other categories are available.
CATEGORIES = [
    "car",
]

# Per-category BGR colors (bbox wireframe and mesh overlay)
CATEGORY_COLORS = {
    "car":                  (0,  230,   0),   # green
    "truck":                (0,  120, 255),   # blue
    "bus":                  (0,   60, 200),   # dark blue
    "trailer":              (180,  0, 255),   # purple
    "motorcycle":           (0,  220, 220),   # cyan
    "bicycle":              (255, 150,  0),   # orange
    "construction_vehicle": (0,    0, 220),   # red
}

# Frame range (set to None to process the full scene)
FRAME_START = 540
FRAME_END   = 541
FRAME_SKIP  = 1

MIN_PROB    = 0.6   # minimum SAM3 detection confidence
MIN_MASK_PX = 100   # skip detections with fewer mask pixels

SAVE_PLY  = False   # save Gaussian splat .ply (large: ~24 MB each)
SAVE_MESH = True    # save mesh vertices + faces as .npz

OUTPUT_DIR = Path(
    f"/media/lleba/ECP_Nuscenes_01/SAM3D_Outputs/Strassbourg/"
    f"scene_{NUSCENES_SCENE_IDX:03d}/SAM3D_Objects"
)

# ──────────────────────────────────────────────────────────────────────────────

# 12 edges of an 8-corner box (corners 0-3 = back face, 4-7 = front face)
_BBOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


# ── nuScenes helpers ──────────────────────────────────────────────────────────

def build_image_and_calib_index(scene_idx: int):
    """
    Build frame index with image paths, intrinsics, and camera→ego extrinsics.

    Returns
    -------
    paths  : {frame_idx: image_path}
    intrs  : {frame_idx: (fx, fy, cx, cy)}
    extrs  : {frame_idx: (R_cam2ego (3,3), t_cam2ego (3,))}
             Transforms a point from camera space to ego-vehicle space:
             p_ego = R_cam2ego @ p_cam + t_cam2ego
    """
    from nuscenes.nuscenes import NuScenes
    from pyquaternion import Quaternion
    print(f"Loading nuScenes metadata (scene {scene_idx})...")
    nusc  = NuScenes(version="v1.0-trainval", dataroot=DATA_ROOT, verbose=False)
    scene = nusc.scene[scene_idx]
    print(f"  Scene: {scene['name']}")

    paths = {}
    intrs = {}
    extrs = {}
    token = nusc.get("sample", scene["first_sample_token"])["data"]["CAM_FRONT"]
    i = 0
    while token:
        rec = nusc.get("sample_data", token)
        paths[i] = nusc.get_sample_data_path(token)
        cal  = nusc.get("calibrated_sensor", rec["calibrated_sensor_token"])
        K    = np.array(cal["camera_intrinsic"])
        intrs[i] = (float(K[0, 0]), float(K[1, 1]),
                    float(K[0, 2]), float(K[1, 2]))
        # camera → ego-vehicle rigid transform
        R_cam2ego = Quaternion(cal["rotation"]).rotation_matrix          # (3, 3)
        t_cam2ego = np.array(cal["translation"], dtype=np.float64)       # (3,)
        extrs[i]  = (R_cam2ego, t_cam2ego)
        token = rec["next"]
        i += 1
    print(f"  Indexed {len(paths)} CAM_FRONT frames.\n")
    return paths, intrs, extrs


# ── SAM3 mask loading ─────────────────────────────────────────────────────────

def load_sam3_masks(categories: list[str]) -> dict[str, dict | None]:
    """
    Load SAM3 tracking output for each vehicle category.

    Returns {category: frame_dict or None}.
    frame_dict is keyed by absolute frame index and holds:
      out_obj_ids, out_probs, out_binary_masks, out_boxes_xywh
    """
    masks = {}
    for cat in categories:
        path = SAM3_DIR / f"{SAM3_SCENE_PREFIX}{cat}.npy"
        if path.exists():
            masks[cat] = np.load(str(path), allow_pickle=True).item()
            print(f"  {cat}: {len(masks[cat])} frames loaded")
        else:
            masks[cat] = None
            print(f"  {cat}: file not found — {path}")
    return masks


def get_detections_for_frame(
    sam3_masks: dict[str, dict | None],
    frame_idx: int,
    W: int,
    H: int,
    min_prob: float = 0.6,
) -> list[dict]:
    """
    Collect all detections (all categories) for one frame.

    Returns list of dicts:
      category, obj_id, prob,
      binary_mask (H×W bool), box_xywh (normalised x1y1wh), box_xyxy (pixels)
    """
    detections = []
    for cat, data in sam3_masks.items():
        if data is None or frame_idx not in data:
            continue
        frame      = data[frame_idx]
        obj_ids    = frame["out_obj_ids"]
        probs      = np.asarray(frame["out_probs"],       dtype=np.float32)
        raw_masks  = frame["out_binary_masks"]
        boxes_xywh = np.asarray(frame["out_boxes_xywh"], dtype=np.float32)

        for i, (oid, prob) in enumerate(zip(obj_ids, probs)):
            if float(prob) < min_prob:
                continue
            bm = np.asarray(raw_masks[i], dtype=bool)
            bx = boxes_xywh[i]                    # [x1, y1, w, h] normalised
            x1 = float(bx[0]) * W
            y1 = float(bx[1]) * H
            x2 = (float(bx[0]) + float(bx[2])) * W
            y2 = (float(bx[1]) + float(bx[3])) * H
            detections.append({
                "category":    cat,
                "obj_id":      int(oid),
                "prob":        float(prob),
                "binary_mask": bm,
                "box_xywh":    bx.tolist(),
                "box_xyxy":    [x1, y1, x2, y2],
            })
    return detections


# ── Coordinate transforms ─────────────────────────────────────────────────────

def _build_l2c_and_p3d_to_r3(output: dict):
    """
    Build two transforms shared by Gaussian and mesh pipelines:
      l2c         : object-local  → PyTorch3D camera space
      t_p3d_to_r3 : PyTorch3D cam → R3/OpenCV camera space (X-right, Y-down, Z-fwd)
    """
    from pytorch3d.renderer import look_at_view_transform
    from pytorch3d.transforms import Transform3d, quaternion_to_matrix
    from sam3d_objects.data.dataset.tdfy.transforms_3d import compose_transform

    def _sq(t):
        while t.dim() > 2:
            t = t.squeeze(0)
        return t

    quat  = _sq(output["rotation"].detach().cpu().float())
    trans = _sq(output["translation"].detach().cpu().float())
    scale = _sq(output["scale"].detach().cpu().float())
    if quat.dim() == 1:
        quat = quat.unsqueeze(0)

    l2c = compose_transform(
        scale=scale,
        rotation=quaternion_to_matrix(quat),
        translation=trans,
    )

    R_r3_to_p3d, _ = look_at_view_transform(
        eye=torch.tensor([[0., 0., -1.]]),
        at=torch.tensor([[0., 0.,  0.]]),
        up=torch.tensor([[0., -1., 0.]]),
    )
    t_p3d_to_r3 = Transform3d().rotate(R_r3_to_p3d).inverse()

    return l2c, t_p3d_to_r3


def gs_to_r3(output: dict) -> np.ndarray:
    """
    Gaussian means: object-local → R3 (OpenCV) camera space.
    Returns (N, 3) float32.
    """
    l2c, t_p3d_to_r3 = _build_l2c_and_p3d_to_r3(output)
    xyz_local = output["gs"].get_xyz.detach().cpu().float().unsqueeze(0)
    xyz_p3d   = l2c.transform_points(xyz_local).squeeze(0)
    xyz_r3    = (
        t_p3d_to_r3
        .transform_points(xyz_p3d.unsqueeze(0))
        .squeeze(0)
        .numpy()
    )
    return xyz_r3.astype(np.float32)


def mesh_to_r3(output: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Mesh vertices: object-local → R3 (OpenCV) camera space.
    Returns (verts_r3 (V,3) float32, faces (F,3) int32).
    """
    mesh_res    = output["mesh"][0]
    verts_local = mesh_res.vertices.float().cpu()
    faces       = mesh_res.faces.long().cpu().numpy().astype(np.int32)

    l2c, t_p3d_to_r3 = _build_l2c_and_p3d_to_r3(output)
    xyz_p3d  = l2c.transform_points(verts_local.unsqueeze(0)).squeeze(0)
    verts_r3 = (
        t_p3d_to_r3
        .transform_points(xyz_p3d.unsqueeze(0))
        .squeeze(0)
        .numpy()
        .astype(np.float32)
    )
    return verts_r3, faces


# ── Mask-guided 3D alignment ──────────────────────────────────────────────────

def align_mesh_to_mask(
    verts_r3: np.ndarray,
    binary_mask: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    """
    Correct the 3D position and scale of mesh vertices so that their
    projection onto the image matches the SAM3 mask.

    Only two corrections are applied:
      1. Translation in X, Y: the projected mesh centroid is moved to the
         mask centroid.  Z (depth) is kept from MoGe — it is our best
         monocular estimate and determines 3D accuracy.
      2. Uniform scale around the corrected centroid: the projected width
         of the mesh is rescaled to match the mask bbox width.  Using
         width avoids aspect-ratio distortion from partially-visible cars.

    The resulting 3D position is:
      - X, Y : as accurate as the 2D detection centroid (very good)
      - Z     : as accurate as MoGe's depth estimate
      - scale : derived from 2D size + Z, so jointly as accurate as Z

    Parameters
    ----------
    verts_r3    : (V, 3) mesh vertices in R3 (OpenCV) camera space
    binary_mask : (H, W) bool — SAM3 instance mask
    K           : (3, 3) camera intrinsics used for projection

    Returns
    -------
    verts_aligned : (V, 3) corrected mesh vertices in the same R3 space
    """
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    # ── 1. Depth: median Z of mesh centroid (MoGe estimate, kept as-is) ───
    ctr = verts_r3.mean(axis=0)          # (3,)
    Z   = float(ctr[2])
    if Z < 1e-3:
        return verts_r3                  # degenerate — don't touch

    # ── 2. Target X, Y from mask centroid + MoGe depth ────────────────────
    ys, xs = np.where(binary_mask)
    if len(xs) == 0:
        return verts_r3
    mask_u = float(xs.mean())
    mask_v = float(ys.mean())
    X_target = (mask_u - cx) * Z / fx
    Y_target = (mask_v - cy) * Z / fy
    ctr_target = np.array([X_target, Y_target, Z], dtype=np.float32)

    # ── 3. Scale: match projected width to mask bbox width ────────────────
    px2d    = _project(verts_r3, K)
    valid   = px2d[:, 0] > -9000
    mask_xs_minmax = (float(xs.min()), float(xs.max()))
    mask_width_px  = mask_xs_minmax[1] - mask_xs_minmax[0]
    proj_width_px  = float(px2d[valid, 0].max() - px2d[valid, 0].min()) if valid.any() else 0.
    if proj_width_px > 1.:
        scale_factor = mask_width_px / proj_width_px
        # Clamp: don't let a bad mask size cause extreme rescaling
        scale_factor = float(np.clip(scale_factor, 0.3, 3.0))
    else:
        scale_factor = 1.0

    # ── 4. Apply: scale around current centroid, then translate ───────────
    verts_aligned = (verts_r3 - ctr) * scale_factor + ctr_target
    return verts_aligned.astype(np.float32)


# ── OBB fitting ───────────────────────────────────────────────────────────────

def _iqr_filter_vertices(verts: np.ndarray, factor: float = 2.5) -> np.ndarray:
    """
    Remove outlier vertices whose coordinate in any axis falls outside
    [Q1 - factor*IQR, Q3 + factor*IQR].  Keeps the main mesh body and
    discards stray reconstructed geometry that would inflate the OBB.
    """
    q1 = np.percentile(verts, 25, axis=0)
    q3 = np.percentile(verts, 75, axis=0)
    iqr = q3 - q1
    lo = q1 - factor * iqr
    hi = q3 + factor * iqr
    keep = np.all((verts >= lo) & (verts <= hi), axis=1)
    return verts[keep]


def mesh_principal_axis(verts_r3: np.ndarray) -> np.ndarray:
    """
    Find the longitudinal axis of a vehicle from its mesh vertices via PCA on
    the horizontal (XZ) plane — analogous to the shoulder-line in bodies.

    Front/back disambiguation is left for future work (TODO); for now the
    direction is arbitrary (whichever eigenvector sign numpy returns).

    Parameters
    ----------
    verts_r3 : (V, 3) mesh vertices in R3 camera space

    Returns
    -------
    fwd : (3,) unit vector in XZ plane (fwd[1] == 0)
    """
    if len(verts_r3) < 4:
        return np.array([0., 0., 1.], dtype=np.float32)
    xz   = verts_r3[:, [0, 2]]
    xz_c = xz - xz.mean(axis=0)
    cov  = xz_c.T @ xz_c / max(len(xz_c) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, np.argmax(eigvals)]        # (2,) — largest variance axis
    fwd = np.array([principal[0], 0.0, principal[1]], dtype=np.float32)
    norm = np.linalg.norm(fwd)
    if norm < 1e-8:
        return np.array([0., 0., 1.], dtype=np.float32)
    # TODO: disambiguate front vs back (e.g. via smaller-Z heuristic or
    #       external heading prior); for now direction is arbitrary.
    return fwd / norm


def compute_oriented_bbox(verts_r3: np.ndarray, fwd: np.ndarray) -> np.ndarray:
    """
    Compute an OBB fitted to mesh vertices in R3 camera space.

    Mirrors compute_oriented_bbox() in run_sam3d_bodies.py.
    Axes: fwd (longitudinal/XZ), right = 90° CW from fwd in XZ, up = Y.

    Parameters
    ----------
    verts_r3 : (V, 3) mesh vertices in R3 camera space
    fwd      : (3,) unit forward vector in XZ plane (fwd[1] == 0)

    Returns
    -------
    corners : (8, 3) — corners 0–3 at f_min (back), 4–7 at f_max (front)
    """
    right = np.array([ fwd[2], 0.0, -fwd[0]], dtype=np.float32)
    up    = np.array([0.0, 1.0, 0.0],          dtype=np.float32)

    r_proj = verts_r3 @ right
    y_proj = verts_r3[:, 1]
    f_proj = verts_r3 @ fwd

    r_min, r_max = float(r_proj.min()), float(r_proj.max())
    y_min, y_max = float(y_proj.min()), float(y_proj.max())
    f_min, f_max = float(f_proj.min()), float(f_proj.max())

    return np.array([
        r_min*right + y_min*up + f_min*fwd,   # 0
        r_max*right + y_min*up + f_min*fwd,   # 1
        r_max*right + y_max*up + f_min*fwd,   # 2
        r_min*right + y_max*up + f_min*fwd,   # 3
        r_min*right + y_min*up + f_max*fwd,   # 4
        r_max*right + y_min*up + f_max*fwd,   # 5
        r_max*right + y_max*up + f_max*fwd,   # 6
        r_min*right + y_max*up + f_max*fwd,   # 7
    ], dtype=np.float32)


# ── ECP pointmap builder ─────────────────────────────────────────────────────

def build_ecp_pointmap(ptmap_p3d_chw: "torch.Tensor",
                       K_ecp: np.ndarray,
                       W: int, H: int) -> "torch.Tensor":
    """
    Build a (H, W, 3) float32 pointmap in **PyTorch3D camera space** using
    ECP intrinsics + MoGe depth.

    MoGe's depth (Z) is taken from `ptmap_p3d_chw` (the Z axis is the same
    in both PyTorch3D and R3/OpenCV space).  The X, Y are recomputed from
    pixel coordinates and ECP intrinsics, then converted to PyTorch3D
    convention (negate X and Y, same as `camera_to_pytorch3d_camera`).

    Passing this tensor as `pointmap=` to `Inference.__call__` makes the
    model reconstruct the object in ECP camera space, so that mesh vertices
    after `mesh_to_r3` project correctly with ECP intrinsics.
    """
    import torch

    # Z is the same in both coordinate conventions
    Z_chw = ptmap_p3d_chw[2:3]                     # (1, H_ds, W_ds)
    Z_full = torch.nn.functional.interpolate(
        Z_chw.unsqueeze(0).float(),                 # (1, 1, H_ds, W_ds)
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    ).squeeze()                                      # (H, W)

    fx, fy = float(K_ecp[0, 0]), float(K_ecp[1, 1])
    cx, cy = float(K_ecp[0, 2]), float(K_ecp[1, 2])

    v_grid, u_grid = torch.meshgrid(
        torch.arange(H, dtype=torch.float32),
        torch.arange(W, dtype=torch.float32),
        indexing="ij",
    )                                                # both (H, W)

    # R3/OpenCV 3-D coords using ECP intrinsics
    X_r3 = (u_grid - cx) * Z_full / fx
    Y_r3 = (v_grid - cy) * Z_full / fy

    # R3 → PyTorch3D: negate X and Y (same transform as camera_to_pytorch3d_camera)
    ptmap_ecp = torch.stack([-X_r3, -Y_r3, Z_full], dim=-1).float()  # (H, W, 3)
    return ptmap_ecp


# ── Shared projection helper ──────────────────────────────────────────────────

def _to_numpy(x):
    return x.detach().cpu().numpy() if hasattr(x, "cpu") else np.asarray(x, dtype=np.float32)


def _project(pts: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Pinhole projection: (N, 3) R3 camera coords → (N, 2) pixel coords.
    Uses ECP intrinsics K (same convention as run_sam3d_bodies.py).
    Points behind camera (Z ≤ 0) receive u = v = -9999.
    """
    valid = pts[:, 2] > 1e-6
    u = np.where(valid, K[0, 0] * pts[:, 0] / np.maximum(pts[:, 2], 1e-6) + K[0, 2], -9999.)
    v = np.where(valid, K[1, 1] * pts[:, 1] / np.maximum(pts[:, 2], 1e-6) + K[1, 2], -9999.)
    return np.stack([u, v], axis=1)


# ── Visualization ─────────────────────────────────────────────────────────────

def visualize_sam3_masks(img: np.ndarray, detections: list[dict]) -> np.ndarray:
    """
    Overlay SAM3 binary masks and 2D bounding boxes on the image.

    Each detection gets a semi-transparent filled mask in its category colour,
    a solid 2D bbox rectangle, and a label showing category + confidence.
    Saved as frame{N}_sam3_masks.jpg so the SAM3 input can be compared
    directly against the SAM3D-Objects output images.

    Parameters
    ----------
    img        : (H, W, 3) BGR image
    detections : list of dicts from get_detections_for_frame()
    """
    H, W  = img.shape[:2]
    out   = img.copy()
    alpha = 0.40   # mask transparency

    for det in detections:
        color = CATEGORY_COLORS.get(det["category"], (0, 230, 0))
        mask  = det["binary_mask"]                    # (H, W) bool

        # Semi-transparent filled mask
        overlay        = out.copy()
        overlay[mask]  = color
        out            = cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)

        # 2D bounding box
        x1, y1, x2, y2 = [int(round(v)) for v in det["box_xyxy"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        # Label: category + probability
        label = f"{det['category']} {det['prob']:.2f}"
        cv2.putText(out, label, (max(0, x1), max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    return out.astype(np.uint8)


def visualize_mesh(img: np.ndarray, verts_r3: np.ndarray,
                   K: np.ndarray, color: tuple = (0, 230, 0)) -> np.ndarray:
    """
    Project mesh vertices onto the image as a colored point cloud.

    Analogue of visualize_mesh() in run_sam3d_bodies.py:
    uses ECP intrinsics K instead of the model's internal focal length.

    Parameters
    ----------
    img      : (H, W, 3) BGR image
    verts_r3 : (V, 3) mesh vertices in R3 camera space
    K        : (3, 3) ECP camera intrinsics
    """
    H, W  = img.shape[:2]
    out   = img.copy()
    pts2d = _project(verts_r3, K)
    u     = np.round(pts2d[:, 0]).astype(int)
    v     = np.round(pts2d[:, 1]).astype(int)
    valid = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    out[v[valid], u[valid]] = color
    return out.astype(np.uint8)


def visualize_bbox(
    img: np.ndarray,
    corners: np.ndarray,
    category: str,
    K: np.ndarray,
    bbox_color: tuple = (0, 230, 0),
    # ── orientation axis (commented out — TODO) ───────────────────────────────
    # fwd: np.ndarray | None = None,
    # axis_color: tuple = (0, 80, 255),
) -> np.ndarray:
    """
    Project and draw the 3D bounding box wireframe onto the image.

    Analogue of visualize_bbox_and_orientation() in run_sam3d_bodies.py.
    Currently draws only the AABB wireframe; orientation axis is left for
    future work (see commented-out OBB/axis code above and below).

    Parameters
    ----------
    img     : (H, W, 3) BGR image
    corners : (8, 3) bbox corners in R3 camera space
    K       : (3, 3) ECP camera intrinsics
    """
    H, W  = img.shape[:2]
    out   = img.copy()

    pts2d = _project(corners, K).astype(int)    # (8, 2)

    # Wireframe
    for i, j in _BBOX_EDGES:
        p1, p2 = tuple(pts2d[i]), tuple(pts2d[j])
        if pts2d[i, 0] == -9999 or pts2d[j, 0] == -9999:
            continue
        if (0 <= p1[0] < W and 0 <= p1[1] < H) or \
           (0 <= p2[0] < W and 0 <= p2[1] < H):
            cv2.line(out, p1, p2, bbox_color, 2, cv2.LINE_AA)

    # ── Orientation axis — TODO (uncomment when estimate_vehicle_axis is wired up)
    # if fwd is not None:
    #     center_r3  = corners.mean(axis=0)
    #     half_depth = float(np.linalg.norm(corners[4] - corners[0])) * 0.55
    #     p_neg = _project((center_r3 - fwd * half_depth)[np.newaxis], K).astype(int)[0]
    #     p_pos = _project((center_r3 + fwd * half_depth)[np.newaxis], K).astype(int)[0]
    #     if p_neg[0] != -9999 and p_pos[0] != -9999:
    #         on_screen = lambda p: 0 <= p[0] < W and 0 <= p[1] < H
    #         if on_screen(p_neg) or on_screen(p_pos):
    #             cv2.line(out, tuple(p_neg), tuple(p_pos), axis_color, 3, cv2.LINE_AA)

    # Category label above the topmost visible corner
    valid = pts2d[(pts2d[:, 0] >= 0) & (pts2d[:, 0] < W) &
                  (pts2d[:, 1] >= 0) & (pts2d[:, 1] < H)]
    if len(valid):
        top = valid[valid[:, 1].argmin()]
        cv2.putText(out, category, (max(0, top[0] - 5), max(15, top[1] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, bbox_color, 2, cv2.LINE_AA)

    return out.astype(np.uint8)


# ── Main processing loop ──────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(SAM3D_REPO / "notebook"))

    try:
        from inference import Inference
    except ImportError as e:
        print(f"ERROR: Cannot import inference from {SAM3D_REPO}/notebook\n  {e}")
        sys.exit(1)

    if not CKPT_CONFIG.exists():
        print(f"ERROR: Checkpoint config not found: {CKPT_CONFIG}")
        sys.exit(1)

    # Load SAM3 masks
    print("\nLoading SAM3 masks...")
    sam3_masks = load_sam3_masks(CATEGORIES)

    # Build nuScenes frame index + ECP calibration
    image_paths, intrs, extrs = build_image_and_calib_index(NUSCENES_SCENE_IDX)

    # Load SAM3D-Objects model
    print(f"\nLoading SAM-3D-Objects model from {CKPT_CONFIG}...")
    inference = Inference(str(CKPT_CONFIG), compile=False)
    print("Model ready.\n")

    # Frame range
    start  = FRAME_START if FRAME_START is not None else 0
    end    = FRAME_END   if FRAME_END   is not None else len(image_paths)
    frames = list(range(start, min(end, len(image_paths)), FRAME_SKIP))

    print(f"Processing {len(frames)} frames "
          f"(indices {start}–{end-1}, every {FRAME_SKIP} frame(s))...\n")

    img_dir = OUTPUT_DIR / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    for frame_idx in tqdm(frames, desc="Frames"):
        if frame_idx not in image_paths:
            continue

        img_path            = image_paths[frame_idx]
        fx, fy, cx, cy      = intrs[frame_idx]
        K_np                = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        R_cam2ego, t_cam2ego = extrs[frame_idx]

        img = cv2.imread(img_path)           # BGR, base for both visualizations
        if img is None:
            print(f"  [WARN] Could not read {img_path}")
            continue
        H, W = img.shape[:2]

        # SAM3D-Objects inference expects RGB
        rgb = img[:, :, ::-1].copy()

        # Gather detections for this frame
        detections = get_detections_for_frame(sam3_masks, frame_idx, W, H, MIN_PROB)
        if not detections:
            continue

        print(f"\nFrame {frame_idx:05d} | {len(detections)} detection(s)")

        # ── Run MoGe once per frame; cache pointmap and K for all objects ──────
        # Injecting MoGe's own pointmap lets the pose decoder operate exactly
        # in its training distribution (correct depth scale → correct 3D size).
        # It also avoids N redundant MoGe runs (one per object) that would
        # happen if we passed pointmap=None to each inference() call.
        # We project with K_moge (MoGe's cx=W/2, cy=H/2) because the pose
        # decoder was trained to output translations in that camera space.
        K_proj    = K_np        # fallback: use ECP K if MoGe pre-compute fails
        moge_ptmap = None       # (H, W, 3) P3D — injected per object below
        try:
            rgba_full = np.concatenate(
                [rgb, np.full((H, W, 1), 255, dtype=np.uint8)], axis=-1
            )
            with inference._pipeline.device:
                ptmap_dict = inference._pipeline.compute_pointmap(rgba_full)
            norm_K = ptmap_dict["intrinsics"].cpu().numpy()  # (3, 3), normalised
            K_proj = np.array([
                [norm_K[0, 0] * W,  0.,  norm_K[0, 2] * W],   # cx = W/2
                [0.,  norm_K[1, 1] * H,  norm_K[1, 2] * H],   # cy = H/2
                [0.,  0.,  1.],
            ], dtype=np.float64)
            # ptmap_dict["pointmap"] is (3, H, W); compute_pointmap() expects (H, W, 3)
            moge_ptmap = ptmap_dict["pointmap"].cpu().permute(1, 2, 0)  # (H, W, 3)
            print(f"  MoGe fx={K_proj[0,0]:.1f}  fy={K_proj[1,1]:.1f}  "
                  f"(ECP fx={K_np[0,0]:.1f}  fy={K_np[1,1]:.1f})")
        except Exception as e:
            print(f"  [WARN] MoGe pre-compute failed ({e}); "
                  f"falling back to per-object MoGe + ECP K")

        # SAM3 mask overlay — saved immediately so it exists even if SAM3D fails
        img_masks = visualize_sam3_masks(img, detections)
        cv2.imwrite(str(img_dir / f"frame{frame_idx:05d}_sam3_masks.jpg"), img_masks)

        # Two accumulation images (BGR) — mirrors run_sam3d_bodies.py
        img_mesh = img.copy()   # mesh vertex projection  (≈ _mesh.jpg)
        img_bbox = img.copy()   # OBB wireframe           (≈ _bbox_orientation.jpg)

        n_saved = 0

        for obj_i, det in enumerate(detections):
            cat         = det["category"]
            binary_mask = det["binary_mask"]
            prob        = det["prob"]
            n_px        = int(binary_mask.sum())
            color       = CATEGORY_COLORS.get(cat, (0, 230, 0))

            print(f"  [{obj_i:02d}] {cat:<22}  prob={prob:.3f}  mask_px={n_px}")

            if n_px < MIN_MASK_PX:
                print(f"         → skip (too few pixels)")
                continue

            # ── SAM3D inference ──────────────────────────────────────────────
            try:
                output = inference(rgb, binary_mask, seed=42, pointmap=moge_ptmap)
            except Exception as e:
                print(f"         → [WARN] inference failed: {e}")
                continue

            # ── Mesh → R3 camera space ───────────────────────────────────────
            verts_r3 = np.zeros((0, 3), dtype=np.float32)
            faces    = np.zeros((0, 3), dtype=np.int32)
            try:
                verts_r3, faces = mesh_to_r3(output)
            except Exception as e:
                print(f"         → [WARN] mesh transform failed: {e}")
                continue

            # Align mesh to SAM3 mask: corrects X/Y shift and scale mismatch.
            # Z (depth) is kept from MoGe — best available monocular estimate.
            verts_r3 = align_mesh_to_mask(verts_r3, binary_mask, K_proj)

            # ── Diagnostics ──────────────────────────────────────────────────
            # Mesh centroid in camera (R3) space
            ctr_cam = verts_r3.mean(axis=0)           # (3,) in R3/camera space

            # 3D dimensions from aligned, filtered mesh
            vf = _iqr_filter_vertices(verts_r3, factor=2.5)
            if len(vf) < 4:
                vf = verts_r3
            dim_x = float(vf[:, 0].max() - vf[:, 0].min())   # width  (left-right in cam)
            dim_y = float(vf[:, 1].max() - vf[:, 1].min())   # height (up-down in cam)
            dim_z = float(vf[:, 2].max() - vf[:, 2].min())   # depth  (into scene)

            # Transform centroid to ego-vehicle frame
            # p_ego = R_cam2ego @ p_cam + t_cam2ego
            ctr_ego = R_cam2ego @ ctr_cam + t_cam2ego  # (3,) in ego frame
            # ego convention: X=forward, Y=left, Z=up (nuScenes standard)

            # Projected 2D bbox of aligned mesh
            px2d   = _project(verts_r3, K_proj)
            valid2d = px2d[:, 0] > -9000
            if valid2d.any():
                u_min, u_max = px2d[valid2d, 0].min(), px2d[valid2d, 0].max()
                v_min, v_max = px2d[valid2d, 1].min(), px2d[valid2d, 1].max()
            else:
                u_min = u_max = v_min = v_max = 0.

            # 2D bbox of SAM3 mask
            ys_m, xs_m = np.where(binary_mask)
            mx1, mx2 = float(xs_m.min()), float(xs_m.max())
            my1, my2 = float(ys_m.min()), float(ys_m.max())

            print(f"         → 3D centre (cam)  X={ctr_cam[0]:+.2f}  Y={ctr_cam[1]:+.2f}  Z={ctr_cam[2]:.2f} m")
            print(f"         → 3D centre (ego)  fwd={ctr_ego[0]:+.2f}  left={ctr_ego[1]:+.2f}  up={ctr_ego[2]:+.2f} m")
            print(f"         → 3D dims (W×H×D)  {dim_x:.2f} × {dim_y:.2f} × {dim_z:.2f} m")
            print(f"         → mesh 2D bbox  u=[{u_min:.0f},{u_max:.0f}]  v=[{v_min:.0f},{v_max:.0f}]  "
                  f"({u_max-u_min:.0f}×{v_max-v_min:.0f})px")
            print(f"         → mask 2D bbox  u=[{mx1:.0f},{mx2:.0f}]  v=[{my1:.0f},{my2:.0f}]  "
                  f"({mx2-mx1:.0f}×{my2-my1:.0f})px")
            # ─────────────────────────────────────────────────────────────────

            # ── OBB fitted to mesh vertices ──────────────────────────────────
            # Filter outlier vertices (IQR × 2.5 per axis) before OBB fitting
            # so stray reconstructed geometry doesn't inflate the box.
            verts_obb = _iqr_filter_vertices(verts_r3, factor=2.5)
            if len(verts_obb) < 4:
                verts_obb = verts_r3   # fallback if filter removes too much
            fwd     = mesh_principal_axis(verts_obb)
            bbox_3d = compute_oriented_bbox(verts_obb, fwd)    # (8, 3)

            # ── Accumulate both frame visualizations ─────────────────────────
            if len(verts_r3) > 0:
                img_mesh = visualize_mesh(img_mesh, verts_r3, K_proj, (170, 170, 170))
            img_bbox = visualize_bbox(img_bbox, bbox_3d, cat, K_proj, bbox_color=color)

            # ── Save per-object outputs ──────────────────────────────────────
            obj_dir = OUTPUT_DIR / f"frame{frame_idx:05d}_{cat}_obj{obj_i:02d}"
            obj_dir.mkdir(parents=True, exist_ok=True)

            if SAVE_MESH and len(verts_r3) > 0:
                np.savez_compressed(
                    obj_dir / "mesh.npz",
                    vertices=verts_r3,
                    faces=faces,
                )

            np.save(obj_dir / "bbox_3d.npy", bbox_3d)

            np.savez_compressed(
                obj_dir / "pose.npz",
                rotation    = _to_numpy(output["rotation"]),
                translation = _to_numpy(output["translation"]),
                scale       = _to_numpy(output["scale"]),
            )

            meta = {
                "frame_idx":   int(frame_idx),
                "obj_idx":     int(obj_i),
                "category":    cat,
                "prob":        float(prob),
                "image_path":  img_path,
                "image_size":  [W, H],
                "camera_K":    K_np.tolist(),
                "mask_pixels": n_px,
                "box_xyxy":    det["box_xyxy"],
                "n_vertices":  int(len(verts_r3)),
                "n_faces":     int(len(faces)),
                # 3D location and size
                "centre_cam":  ctr_cam.tolist(),     # (X, Y, Z) in R3 camera space [m]
                "centre_ego":  ctr_ego.tolist(),     # (fwd, left, up) in ego frame [m]
                "dims_whd":    [dim_x, dim_y, dim_z],# width, height, depth [m]
            }
            with open(obj_dir / "metadata.json", "w") as f:
                json.dump(meta, f, indent=2)

            if SAVE_PLY:
                ply_path = str(obj_dir / "splat.ply")
                output["gs"].save_ply(ply_path)
                ply_mb = os.path.getsize(ply_path) / 1e6
                print(f"         → saved .ply: {ply_path} ({ply_mb:.1f} MB)")

            n_saved += 1

        if n_saved == 0:
            continue

        # ── Save two per-frame images — mirrors run_sam3d_bodies.py naming ────
        cv2.imwrite(str(img_dir / f"frame{frame_idx:05d}_mesh.jpg"),  img_mesh)
        cv2.imwrite(str(img_dir / f"frame{frame_idx:05d}_bbox_orientation.jpg"), img_bbox)
        print(f"  Saved: _mesh.jpg  _bbox_orientation.jpg")

    print(f"\nDone. Outputs saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
