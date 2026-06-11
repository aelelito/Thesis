"""
SAM 3D Body pipeline for nuScenes pedestrian reconstruction.

Processes CAM_FRONT images from a nuScenes scene and outputs:
  - 3D body mesh (vertices + faces)
  - 3D skeleton (joint positions)
  - 3D bounding box (8 corners in camera space)
  - 2D visualizations (skeleton + mesh rendered on image)

All outputs are saved per pedestrian per frame for downstream analysis.

Run with:
  conda activate sam3d-body
  python process_nuscenes_sam3d_body.py
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

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Configuration %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# SAM 3D Body repo and checkpoint paths
SAM3D_BODY_REPO = Path(__file__).parent / "sam-3d-body"
SAM3D_BODY_CKPT = SAM3D_BODY_REPO / "checkpoints" / "sam-3d-body-dinov3" / "model.ckpt"
SAM3D_BODY_MHR  = SAM3D_BODY_REPO / "checkpoints" / "sam-3d-body-dinov3" / "assets" / "mhr_model.pt"

# nuScenes dataset
DATA_ROOT = "/media/lleba/ECP_Nuscenes_01/output2/ecp2nuscenes"
NUSCENES_SCENE_IDX = 10

# Optional: restrict to a subset of frames for testing
# Set to None to process all frames of a scene
FRAME_START = 900  # e.g., 150
FRAME_END   = 1500  # e.g., 155
FRAME_SKIP  = 50    # Process every Nth frame (1 = all frames, 10 = every 10th frame)

# Output directory
OUTPUT_DIR = Path(f"/media/lleba/ECP_Nuscenes_01/SAM3D_Outputs/Strassbourg/scene_{NUSCENES_SCENE_IDX:03d}/SAM3D_Body")



# SAM3 mask file for this scene (set to None to use ViTDet only).
# Must match NUSCENES_SCENE_IDX — provides better recall for far/occluded pedestrians.
SAM3_MASK_FILE = Path(
    f"/media/lleba/ECP_Nuscenes_01/SAM3_Visualizations/Strassbourg/"
    f"scene_{NUSCENES_SCENE_IDX}_cam_0_sam3_outputs__pedestrian.npy"
)

# ============================= Hyperparameters for detection and merging ==============================
BBOX_THRESH = 0.8             # Detection threshold for ViTDet (internal detector of SAM 3D Body)
SAM3_SCORE_THRESH     = 0.5   # minimum SAM3 confidence to accept a detection
SAM3_IOU_MERGE_THRESH = 0.4   # IoU above which a ViTDet box is considered a duplicate

SAVE_MESH_PLY = False         # Save mesh as .ply files (large, for 3D viewers like MeshLab/Blender)
# ======================================================================================================

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Setup SAM 3D Body %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def setup_sam3d_body():
    """
    Initialize SAM 3D Body estimator with detector and FOV estimator.
    
    Returns
    -------
    estimator : SAM3DBodyEstimator instance ready for inference
    """
    sys.path.insert(0, str(SAM3D_BODY_REPO))
    
    import torch
    from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
    from tools.build_detector import HumanDetector
    from tools.build_fov_estimator import FOVEstimator
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading SAM 3D Body model from {SAM3D_BODY_CKPT}...")
    model, model_cfg = load_sam_3d_body(
        str(SAM3D_BODY_CKPT),
        device=device,
        mhr_path=str(SAM3D_BODY_MHR),
    )
    
    print("Loading ViTDet human detector...")
    human_detector = HumanDetector(name="vitdet", device=device)
    
    print("Loading MoGe FOV estimator...")
    fov_estimator = FOVEstimator(name="moge2", device=device)
    
    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=human_detector,
        human_segmentor=None,  # ViTDet provides boxes; segmentation not required
        fov_estimator=fov_estimator,
    )
    
    print("SAM 3D Body ready.\n")
    return estimator
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% nuScenes helpers %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def build_image_and_calib_index(scene_idx: int):
    """
    Build index of CAM_FRONT images and camera intrinsics for a scene.
    
    Returns
    -------
    paths : dict[int, str]  — {frame_idx: image_path}
    intrs : dict[int, tuple[float, float, float, float]]  — {frame_idx: (fx, fy, cx, cy)}
    """
    from nuscenes.nuscenes import NuScenes
    
    print(f"Loading nuScenes metadata (scene {scene_idx})...")
    nusc  = NuScenes(version="v1.0-trainval", dataroot=DATA_ROOT, verbose=False)
    scene = nusc.scene[scene_idx]
    print(f"  Scene: {scene['name']}")
    
    paths = {}
    intrs = {}
    
    token = nusc.get("sample", scene["first_sample_token"])["data"]["CAM_FRONT"]
    i = 0
    while token:
        rec = nusc.get("sample_data", token)
        paths[i] = nusc.get_sample_data_path(token)
        
        cal = nusc.get("calibrated_sensor", rec["calibrated_sensor_token"])
        K   = np.array(cal["camera_intrinsic"])
        intrs[i] = (float(K[0, 0]), float(K[1, 1]),
                    float(K[0, 2]), float(K[1, 2]))
        
        token = rec["next"]
        i += 1
    
    print(f"  Indexed {len(paths)} CAM_FRONT frames.\n")
    return paths, intrs
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 3D bounding box computation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# MHR70 joint indices (joints.shape = (70, 3) and e.g. joints[0] is the 3D position of the nose)
# MHR70 is a 70-joint human body representation used by SAM 3D Body
_NOSE = 0
_L_SHOULDER = 5
_R_SHOULDER = 6

# 12 edges for an 8-corner box (same corner ordering used by both AABB and OBB below)
_BBOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),   # face at f_min (back face)
    (4, 5), (5, 6), (6, 7), (7, 4),   # face at f_max (front face)
    (0, 4), (1, 5), (2, 6), (3, 7),   # vertical pillars
]


def facing_direction(joints: np.ndarray) -> np.ndarray:
    """
    Estimate the horizontal facing direction of a pedestrian.

    Strategy: take the shoulder line, compute the perpendicular in the XZ plane
    (the ground plane), then disambiguate the sign using the nose — the nose is
    always on the front side of the body (made assumption here).

    This is more stable than a nose-to-midpoint vector because it ignores head
    tilt and is always orthogonal to the body's lateral axis.

    Parameters
    ----------
    joints : (70, 3) MHR70 keypoints in body-relative space

    Returns
    -------
    fwd : (3,) unit vector in XZ plane pointing in the direction the person faces.
          Y component is always 0.
    """
    shoulder_vec = joints[_R_SHOULDER] - joints[_L_SHOULDER]  # left → right in body space
    
    # Project onto the XZ plane and normalise
    sv_xz = np.array([shoulder_vec[0], 0.0, shoulder_vec[2]], dtype=np.float32)

    # Convert to unit vector
    sv_xz /= np.linalg.norm(sv_xz) + 1e-8 # + 1e-8 to avoid division by zero in edge cases (e.g. shoulders perfectly vertical, though unlikely)

    # The two perpendicular candidates (90° rotation in XZ):
    # facing_direction = (-sv_z, 0, sv_x)  or  (sv_z, 0, -sv_x)
    facing_direction = np.array([-sv_xz[2], 0.0, sv_xz[0]], dtype=np.float32)

    # Use nose to pick the front-facing direction
    shoulder_mid = (joints[_L_SHOULDER] + joints[_R_SHOULDER]) / 2.0
    nose_offset = joints[_NOSE] - shoulder_mid
    if np.dot(facing_direction, nose_offset) < 0:
        facing_direction = -facing_direction

    return facing_direction


def compute_oriented_bbox(vertices: np.ndarray, fwd: np.ndarray) -> np.ndarray:
    """
    Compute an oriented bounding box (OBB) whose axes are aligned with the
    pedestrian's facing direction.

    The three local axes are:
      right   = perpendicular to fwd in XZ (the body's lateral axis)
      up      = Y axis (vertical, shared with world)
      forward = fwd (the facing direction in XZ)

    Vertices are projected onto each axis, min/max are found, and the 8 corners
    are reconstructed in body-relative world space.

    Parameters
    ----------
    vertices : (N, 3) body-relative mesh vertices
    fwd      : (3,) unit facing vector in XZ plane (fwd[1] == 0)

    Returns
    -------
    corners : (8, 3) OBB corners in body-relative space.
              Corner ordering matches _BBOX_EDGES (same as a standard AABB).
    """
    # Local axes
    right = np.array([fwd[2], 0.0, -fwd[0]], dtype=np.float32)  # 90° CW from fwd in XZ
    up    = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    # Project all vertices onto local axes
    r_proj = vertices @ right       # (N,) lateral component
    y_proj = vertices[:, 1]         # (N,) vertical component (unchanged axis)
    f_proj = vertices @ fwd         # (N,) depth component along facing direction

    r_min, r_max = float(r_proj.min()), float(r_proj.max())
    y_min, y_max = float(y_proj.min()), float(y_proj.max())
    f_min, f_max = float(f_proj.min()), float(f_proj.max())

    # Reconstruct the 8 corners in body-relative space.
    # Ordering: x→right, y→up, z→fwd  (matches _BBOX_EDGES face grouping)
    corners = np.array([
        r_min*right + y_min*up + f_min*fwd,   # 0
        r_max*right + y_min*up + f_min*fwd,   # 1
        r_max*right + y_max*up + f_min*fwd,   # 2
        r_min*right + y_max*up + f_min*fwd,   # 3
        r_min*right + y_min*up + f_max*fwd,   # 4
        r_max*right + y_min*up + f_max*fwd,   # 5
        r_max*right + y_max*up + f_max*fwd,   # 6
        r_min*right + y_max*up + f_max*fwd,   # 7
    ], dtype=np.float32)
    return corners
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Visualization %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def visualize_skeleton(img: np.ndarray, outputs: list, faces,
                       K: np.ndarray) -> np.ndarray:
    """
    Draw 2D skeleton overlay for all detected persons on a single image.

    Reprojects pred_keypoints_3d using ECP intrinsics K via _project() — the
    same projection used for the OBB — instead of relying on pred_keypoints_2d
    from the model (which uses slightly different internal cx/cy assumptions).

    Persons are drawn farthest-first so nearer people render on top.

    Returns
    -------
    (H, W, 3) BGR image with skeleton overlay
    """
    sys.path.insert(0, str(SAM3D_BODY_REPO))
    from tools.vis_utils import visualizer as _vis

    result = img.copy()
    sorted_out = sorted(outputs,
                        key=lambda o: float(np.asarray(o["pred_cam_t"])[2]),
                        reverse=True)
    for person in sorted_out:
        joints_3d = _to_numpy(person["pred_keypoints_3d"])   # (N, 3) body-relative
        cam_t     = _to_numpy(person["pred_cam_t"])           # (3,)
        kp2d      = _project(joints_3d, cam_t, K)             # (N, 2) pixel coords
        kp2d_vis  = np.concatenate([kp2d, np.ones((len(kp2d), 1))], axis=-1)
        result    = _vis.draw_skeleton(result, kp2d_vis)
    return result.astype(np.uint8)


def visualize_mesh(img: np.ndarray, outputs: list, faces,
                   fx: float, cx: float, cy: float) -> np.ndarray:
    """
    Render body mesh overlay using ECP camera intrinsics.

    Renders each person individually with their correct pred_cam_t, compositing
    farthest-first so nearer people appear on top. Uses ECP focal length and
    principal point directly instead of visualize_sample_together's fake combined
    translation, which was causing meshes to appear at wrong scale/position.

    Returns
    -------
    (H, W, 3) BGR image with mesh overlay on original
    """
    sys.path.insert(0, str(SAM3D_BODY_REPO))
    from sam_3d_body.visualization.renderer import Renderer

    renderer = Renderer(focal_length=fx, faces=faces)

    # Renderer expects BGR uint8 input and returns float32 RGB [0, 1]
    canvas = img.copy()  # BGR uint8, updated after each person

    sorted_out = sorted(outputs,
                        key=lambda o: float(np.asarray(o["pred_cam_t"])[2]),
                        reverse=True)

    for person in sorted_out:
        verts = _to_numpy(person["pred_vertices"])
        cam_t = _to_numpy(person["pred_cam_t"])
        rendered = renderer(
            vertices=verts,
            cam_t=cam_t,
            image=canvas,
            full_frame=False,
            camera_center=[cx, cy],
        )
        canvas = (rendered * 255).clip(0, 255).astype(np.uint8)
        # rendered is RGB [0,1] → convert back to BGR for next iteration
        canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

    return canvas


def _project(pts: np.ndarray, cam_t: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Standard pinhole: (N,3) body-relative → (N,2) pixel coords."""
    p = pts + cam_t[np.newaxis, :]
    u = K[0, 0] * p[:, 0] / p[:, 2] + K[0, 2]
    v = K[1, 1] * p[:, 1] / p[:, 2] + K[1, 2]
    return np.stack([u, v], axis=1)


def _to_numpy(x):
    return x.detach().cpu().numpy() if hasattr(x, "cpu") else np.asarray(x, dtype=np.float32)


def visualize_bbox_and_orientation(
    img: np.ndarray,
    outputs: list,
    K: np.ndarray,
    bbox_color: tuple = (0, 255, 255),
    arrow_color: tuple = (0, 80, 255),
    arrow_len: float = 0.5,
) -> np.ndarray:
    """
    Project oriented 3D bounding boxes and facing arrows onto the image.

    For each person:
      - Computes an OBB aligned with the pedestrian's facing direction and projects
        its 8 corners using the pinhole camera model.
      - Draws a facing arrow from the shoulder midpoint.  The direction is the
        perpendicular to the shoulder line in the horizontal (XZ) plane; the nose
        is used only to resolve the front/back ambiguity.

    Parameters
    ----------
    img     : (H, W, 3) BGR image
    outputs : list of per-person dicts from estimator.process_one_image()
    K       : (3, 3) camera intrinsics (ECP calibration)

    Returns
    -------
    annotated copy of img (uint8 BGR)
    """
    H, W = img.shape[:2]
    out_img = img.copy()

    for person in outputs:
        verts  = _to_numpy(person["pred_vertices"])
        joints = _to_numpy(person["pred_keypoints_3d"])
        cam_t  = _to_numpy(person["pred_cam_t"])

        fwd = facing_direction(joints)                               # (3,) unit XZ vector

        # ── oriented 3D bbox wireframe ─────────────────────────────────────
        bbox_3d   = compute_oriented_bbox(verts, fwd)                # (8, 3)
        corners_2d = _project(bbox_3d, cam_t, K).astype(int)         # (8, 2)
        for i, j in _BBOX_EDGES:
            p1, p2 = corners_2d[i], corners_2d[j]
            if (0 <= p1[0] < W and 0 <= p1[1] < H) or (0 <= p2[0] < W and 0 <= p2[1] < H):
                cv2.line(out_img, tuple(p1), tuple(p2), bbox_color, 2, cv2.LINE_AA)

        # ── facing arrow from shoulder midpoint ────────────────────────────
        shoulder_mid = (joints[_L_SHOULDER] + joints[_R_SHOULDER]) / 2.0
        tip = shoulder_mid + fwd * arrow_len
        root_2d = _project(shoulder_mid[np.newaxis], cam_t, K)[0].astype(int)
        tip_2d  = _project(tip[np.newaxis],          cam_t, K)[0].astype(int)
        if (0 <= root_2d[0] < W and 0 <= root_2d[1] < H) or \
           (0 <= tip_2d[0]  < W and 0 <= tip_2d[1]  < H):
            cv2.arrowedLine(out_img, tuple(root_2d), tuple(tip_2d),
                            arrow_color, 3, cv2.LINE_AA, tipLength=0.25)

    return out_img.astype(np.uint8)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SAM3 detection helpers %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def get_sam3_bboxes(sam3_data: dict, frame_idx: int,
                    W: int, H: int, score_thresh: float = 0.5) -> np.ndarray:
    """
    Extract absolute xyxy bboxes from a SAM3 tracking output for one frame.

    SAM3 stores boxes as normalised (x1, y1, w, h) in [0, 1].

    Returns
    -------
    bboxes : (N, 4) float32 in absolute xyxy pixel coordinates, or (0, 4) if
             the frame has no detections above score_thresh.
    """
    if frame_idx not in sam3_data:
        return np.empty((0, 4), dtype=np.float32)
    frame = sam3_data[frame_idx]
    probs = np.asarray(frame["out_probs"],      dtype=np.float32)   # (N,)
    boxes = np.asarray(frame["out_boxes_xywh"], dtype=np.float32)   # (N, 4) norm xywh
    keep  = probs >= score_thresh
    if not keep.any():
        return np.empty((0, 4), dtype=np.float32)
    b  = boxes[keep]
    x1 = b[:, 0] * W
    y1 = b[:, 1] * H
    x2 = (b[:, 0] + b[:, 2]) * W
    y2 = (b[:, 1] + b[:, 3]) * H
    return np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)


def merge_with_vitdet(sam3_boxes: np.ndarray, vitdet_boxes: np.ndarray,
                      iou_thresh: float = 0.4) -> np.ndarray:
    """
    Return SAM3 boxes plus any ViTDet box whose max IoU with all SAM3 boxes
    is below iou_thresh (i.e. not already covered by a SAM3 track).

    SAM3 boxes always take priority; ViTDet only fills in the gaps.
    """
    if len(vitdet_boxes) == 0:
        return sam3_boxes
    if len(sam3_boxes) == 0:
        return vitdet_boxes

    # IoU matrix: rows = ViTDet, cols = SAM3
    ix1 = np.maximum(vitdet_boxes[:, 0:1], sam3_boxes[:, 0])
    iy1 = np.maximum(vitdet_boxes[:, 1:2], sam3_boxes[:, 1])
    ix2 = np.minimum(vitdet_boxes[:, 2:3], sam3_boxes[:, 2])
    iy2 = np.minimum(vitdet_boxes[:, 3:4], sam3_boxes[:, 3])
    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    area_v = ((vitdet_boxes[:, 2] - vitdet_boxes[:, 0]) *
              (vitdet_boxes[:, 3] - vitdet_boxes[:, 1]))[:, None]
    area_s = ((sam3_boxes[:, 2] - sam3_boxes[:, 0]) *
              (sam3_boxes[:, 3] - sam3_boxes[:, 1]))[None, :]
    iou = inter / (area_v + area_s - inter + 1e-8)   # (K, M)

    extra = vitdet_boxes[iou.max(axis=1) < iou_thresh]
    if len(extra) == 0:
        return sam3_boxes
    return np.vstack([sam3_boxes, extra])
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Main processing loop %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load SAM 3D Body
    estimator = setup_sam3d_body()

    # Load SAM3 tracking data (better recall for far/occluded pedestrians)
    sam3_data = None
    if SAM3_MASK_FILE and Path(SAM3_MASK_FILE).exists():
        print(f"Loading SAM3 masks from {SAM3_MASK_FILE}...")
        sam3_data = np.load(str(SAM3_MASK_FILE), allow_pickle=True).item()
        print(f"  {len(sam3_data)} frames available.\n")
    else:
        print(f"[INFO] SAM3 mask file not found — using ViTDet only.\n")

    # Index nuScenes frames
    image_paths, intrs = build_image_and_calib_index(NUSCENES_SCENE_IDX)
    
    # Determine frame range
    start = FRAME_START if FRAME_START is not None else 0
    end   = FRAME_END   if FRAME_END   is not None else len(image_paths)
    frame_indices = list(range(start, min(end, len(image_paths)), FRAME_SKIP))
    
    print(f"Processing {len(frame_indices)} frames (indices {start}–{end-1}, every {FRAME_SKIP} frame(s))...\n")
    
    # Process each frame
    for frame_idx in tqdm(frame_indices, desc="Frames"):
        img_path = image_paths[frame_idx]
        fx, fy, cx, cy = intrs[frame_idx]
        
        # Load image
        img = cv2.imread(img_path)
        H, W = img.shape[:2]
        
        # Build ECP camera intrinsics tensor for this frame (1, 3, 3)
        K_np = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        cam_int_tensor = torch.from_numpy(K_np).unsqueeze(0)  # (1, 3, 3)

        # Build merged bbox set: SAM3 tracks (primary) + supplementary ViTDet boxes
        merged_bboxes = None
        if sam3_data is not None:
            sam3_bboxes = get_sam3_bboxes(sam3_data, frame_idx, W, H, SAM3_SCORE_THRESH)
            if len(sam3_bboxes) > 0:
                # Run ViTDet separately to catch any pedestrians SAM3 missed
                vitdet_bboxes = estimator.detector.run_human_detection(
                    img, bbox_thr=BBOX_THRESH, nms_thr=0.3, default_to_full_image=False
                )
                vitdet_bboxes = np.asarray(vitdet_bboxes, dtype=np.float32).reshape(-1, 4)
                merged_bboxes = merge_with_vitdet(sam3_bboxes, vitdet_bboxes,
                                                  SAM3_IOU_MERGE_THRESH)
                print(f"  Frame {frame_idx}: {len(sam3_bboxes)} SAM3 + "
                      f"{len(vitdet_bboxes)} ViTDet → {len(merged_bboxes)} merged boxes")

        # Run SAM 3D Body with ECP intrinsics so pred_cam_t is in that space
        try:
            outputs = estimator.process_one_image(
                img_path,
                bboxes=merged_bboxes,   # None → ViTDet runs internally
                bbox_thr=BBOX_THRESH,
                use_mask=False,
                cam_int=cam_int_tensor,
            )
        except Exception as e:
            print(f"\n[WARN] Frame {frame_idx} failed: {e}")
            continue
        
        # outputs is a list of dicts, one per detected person
        n_detections = len(outputs)
        if n_detections == 0:
            continue
        
        img_dir = OUTPUT_DIR / "images"
        img_dir.mkdir(parents=True, exist_ok=True)

        # Skeleton overlay — all persons drawn on the original image
        img_skel = visualize_skeleton(img, outputs, estimator.faces, K_np)
        cv2.imwrite(str(img_dir / f"frame{frame_idx:05d}_skeleton.jpg"), img_skel)

        # Mesh overlay — all persons' meshes rendered on the original image
        img_mesh = visualize_mesh(img, outputs, estimator.faces, fx, cx, cy)
        cv2.imwrite(str(img_dir / f"frame{frame_idx:05d}_mesh.jpg"), img_mesh)

        # OBB wireframe + facing arrow (uses ECP intrinsics via K_np)
        img_bbox = visualize_bbox_and_orientation(img, outputs, K_np)
        cv2.imwrite(str(img_dir / f"frame{frame_idx:05d}_bbox_orientation.jpg"), img_bbox)
        
        # Process each detected pedestrian
        for ped_idx, out in enumerate(outputs):
            # Extract outputs (keys are pred_* not direct names)
            verts = out["pred_vertices"]
            if hasattr(verts, "cpu"):
                verts = verts.detach().cpu().numpy()
            
            faces = estimator.faces
            if hasattr(faces, "cpu"):
                faces = faces.detach().cpu().numpy()
            
            joints_3d = out["pred_keypoints_3d"]
            if hasattr(joints_3d, "cpu"):
                joints_3d = joints_3d.detach().cpu().numpy()

            # Compute oriented 3D bounding box (aligned with facing direction)
            fwd_dir = facing_direction(joints_3d)
            bbox_3d = compute_oriented_bbox(verts, fwd_dir)

            # Sanity check: print physical OBB dimensions (adults ~ 1.7m tall, 0.5m wide, 0.3m deep)
            fwd     = fwd_dir
            right   = np.array([fwd[2], 0.0, -fwd[0]], dtype=np.float32)
            r_proj  = verts @ right
            y_proj  = verts[:, 1]
            f_proj  = verts @ fwd
            h_m = float(y_proj.max() - y_proj.min())
            w_m = float(r_proj.max() - r_proj.min())
            d_m = float(f_proj.max() - f_proj.min())
            depth_m = float(np.asarray(out["pred_cam_t"], dtype=np.float32)[2])
            print(f"  [ped {ped_idx:02d}] depth={depth_m:.1f}m  "
                  f"H={h_m:.2f}m  W={w_m:.2f}m  D={d_m:.2f}m")
            
            # Save outputs
            ped_dir = OUTPUT_DIR / f"frame{frame_idx:05d}_ped{ped_idx:02d}"
            ped_dir.mkdir(parents=True, exist_ok=True)
            
            # Save mesh as .npz (compact)
            np.savez_compressed(
                ped_dir / "mesh.npz",
                vertices=verts,
                faces=faces,
            )
            
            # Save skeleton
            np.save(ped_dir / "joints_3d.npy", joints_3d)
            
            # Save 3D bbox
            np.save(ped_dir / "bbox_3d.npy", bbox_3d)
            
            # Extract camera parameters (consistent with ECP intrinsics)
            cam_t = out["pred_cam_t"]
            if hasattr(cam_t, "cpu"):
                cam_t = cam_t.detach().cpu().numpy()
            cam_t = np.asarray(cam_t, dtype=np.float32)

            kp2d = out["pred_keypoints_2d"]
            if hasattr(kp2d, "cpu"):
                kp2d = kp2d.detach().cpu().numpy()

            np.save(ped_dir / "cam_t.npy", cam_t)
            np.save(ped_dir / "keypoints_2d.npy", np.asarray(kp2d, dtype=np.float32))

            # Save metadata
            meta = {
                "frame_idx": int(frame_idx),
                "pedestrian_idx": int(ped_idx),
                "image_path": img_path,
                "image_size": [W, H],
                "camera_K": K_np.tolist(),
                "n_vertices": int(len(verts)),
                "n_joints": int(len(joints_3d)),
            }
            with open(ped_dir / "metadata.json", "w") as f:
                json.dump(meta, f, indent=2)
            
            # Optional: save .ply for 3D viewers (MeshLab, Blender)
            if SAVE_MESH_PLY:
                ply_path = ped_dir / "mesh.ply"
                with open(ply_path, "w") as f:
                    f.write("ply\n")
                    f.write("format ascii 1.0\n")
                    f.write(f"element vertex {len(verts)}\n")
                    f.write("property float x\n")
                    f.write("property float y\n")
                    f.write("property float z\n")
                    f.write(f"element face {len(faces)}\n")
                    f.write("property list uchar int vertex_indices\n")
                    f.write("end_header\n")
                    for v in verts:
                        f.write(f"{v[0]} {v[1]} {v[2]}\n")
                    for face in faces:
                        f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
    
    print(f"\nDone. Outputs saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()