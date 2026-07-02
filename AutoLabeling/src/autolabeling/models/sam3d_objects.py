"""
SAM3D Objects inference wrapper.

Pointmap modes (set lidar.pointmap_mode in config):
  'baseline'         — dense MoGe relative-depth pointmap (non-metric). Original pipeline.
  'o1_lidar'         — sparse metric LiDAR pointmap. Requires FrameRecord.lidar_path;
                       falls back to MoGe baseline if LiDAR is unavailable for a frame.
  'o2_moge_affine'   — dense MoGe pointmap calibrated to metric scale via a global affine
                       fit  Z_metric = a*Z_moge + b  (least-squares on all in-image LiDAR
                       returns). Falls back to unscaled MoGe if fewer than min_affine_pts
                       LiDAR returns are available.
  'o3_local_affine'  — dense MoGe pointmap with a per-object affine fit. For each detected
                       object, HDBSCAN isolates its in-mask LiDAR surface points in 3D ego
                       space, then fits a separate (a,b) for that object. The full-frame
                       pointmap is rebuilt per object before each inference call. Falls back
                       to global affine (O2) if too few clean in-mask points are available.
"""
import gc
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from ..utils.lidar import filter_inmask_lidar_hdbscan
from ..utils.logging_utils import suppress_output


class SAM3DObjectsModel:
    def __init__(
        self,
        repo_path: str,
        config_path: str,
        prompts: Dict[str, str],           # {text_prompt: pipeline_type}
        device: Optional[str] = None,
        pointmap_mode: str = 'baseline',   # 'baseline' | 'o1_lidar' | 'o2_moge_affine'
        min_affine_pts: int = 5,           # minimum LiDAR/MoGe pairs for a reliable affine fit
        hdbscan_params: Optional[Dict] = None,  # per-class HDBSCAN params for O3
    ):
        self.repo_path       = str(repo_path)
        self.config_path     = str(config_path)
        self.prompts         = prompts
        self.device          = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.pointmap_mode   = pointmap_mode
        self.min_affine_pts  = min_affine_pts
        self.hdbscan_params  = hdbscan_params or {}
        self._inference      = None

    def load(self) -> None:
        # SAM3D Objects uses LIDRA_SKIP_INIT to avoid importing hardware-specific
        # drivers at load time. Must be set before any sam3d_objects import.
        os.environ['LIDRA_SKIP_INIT'] = 'true'

        notebook_dir = str(Path(self.repo_path) / 'notebook')
        if notebook_dir not in sys.path:
            sys.path.insert(0, notebook_dir)

        from inference import Inference as SAM3DObjectsInference
        with suppress_output():
            self._inference = SAM3DObjectsInference(self.config_path, compile=False)
        print('  SAM3D Objects loaded.')

    def unload(self) -> None:
        del self._inference
        self._inference = None
        gc.collect()
        torch.cuda.empty_cache()
        print('  SAM3D Objects unloaded.')

    # ── Pointmap helpers ──────────────────────────────────────────────────────

    def _compute_moge_pointmap(self, img_rgb: np.ndarray, K: np.ndarray):
        """
        Run MoGe on img_rgb → dense relative-depth pointmap in PyTorch3D convention.

        Returns
        -------
        ptmap   : (H, W, 3) float32  PyTorch3D space (-X_r3, -Y_r3, Z_relative)
        Z_map   : (H, W)    float32  relative depth (MoGe units, non-metric)
        """
        H, W = img_rgb.shape[:2]
        rgba = np.concatenate([img_rgb, np.full((H, W, 1), 255, dtype=np.uint8)], axis=-1)

        with self._inference._pipeline.device, suppress_output():
            ptmap_dict = self._inference._pipeline.compute_pointmap(rgba)

        ptmap_p3d = ptmap_dict['pointmap'].cpu().permute(1, 2, 0).numpy().astype(np.float32)
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        u_grid, v_grid = np.meshgrid(
            np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32)
        )
        Z_map = ptmap_p3d[..., 2]
        X_r3  = (u_grid - cx) * Z_map / fx
        Y_r3  = (v_grid - cy) * Z_map / fy
        ptmap = np.stack([-X_r3, -Y_r3, Z_map], axis=-1).astype(np.float32)
        return ptmap, Z_map

    @staticmethod
    def _project_lidar(frame, img_rgb: np.ndarray, pts_ego: np.ndarray = None):
        """
        Project LiDAR sweep into camera frame.

        Returns u_vis, v_vis (float pixel coords), Z_vis (metric depth in metres),
        and pts_ego_vis (ego-frame XYZ) for all LiDAR points that land inside the
        image and are in front of the camera.
        Returns None, None, None, None if frame has no LiDAR.

        Parameters
        ----------
        frame    : FrameRecord
        img_rgb  : (H, W, 3) uint8 image (used only to get H, W)
        pts_ego  : optional (N, 3) pre-loaded ego-frame point cloud (e.g. aggregated).
                   If None, falls back to loading the single sweep from frame.lidar_path.
        """
        if pts_ego is None:
            if frame.lidar_path is None:
                return None, None, None, None
            pts_raw = np.fromfile(frame.lidar_path, dtype=np.float32).reshape(-1, 5)[:, :3]
            pts_ego = (frame.R_l2e @ pts_raw.astype(np.float64).T).T + frame.t_l2e

        H, W   = img_rgb.shape[:2]
        K      = frame.K
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

        pts_cam = (frame.R_c2e.T @ (pts_ego - frame.t_c2e).T).T   # ego → camera (R3/OpenCV)
        Z_cam = pts_cam[:, 2]
        front = Z_cam > 0
        u = pts_cam[front, 0] / Z_cam[front] * fx + cx
        v = pts_cam[front, 1] / Z_cam[front] * fy + cy
        in_img = (u >= -0.5) & (u < W - 0.5) & (v >= -0.5) & (v < H - 0.5)

        return (
            u[in_img].astype(np.float32),
            v[in_img].astype(np.float32),
            Z_cam[front][in_img].astype(np.float32),
            pts_ego[front][in_img].astype(np.float32),  # ego-frame coords (for HDBSCAN in O3)
        )

    def _compute_lidar_pointmap(self, frame, img_rgb: np.ndarray, pts_ego: np.ndarray = None):
        """
        O1: build a sparse per-pixel LiDAR pointmap in PyTorch3D space.
        Pixels with no LiDAR return are NaN — the model fills gaps from image priors.

        Returns (H, W, 3) float32 or falls back to MoGe if no LiDAR available.
        """
        H, W = img_rgb.shape[:2]
        u_vis, v_vis, Z_vis, _ = self._project_lidar(frame, img_rgb, pts_ego)

        if u_vis is None:
            print(f'    [warn] no LiDAR for {frame.scene_name} frame {frame.frame_idx}, '
                  f'falling back to MoGe.')
            ptmap, _ = self._compute_moge_pointmap(img_rgb, frame.K)
            return ptmap

        u_int = np.round(u_vis).astype(np.int32)
        v_int = np.round(v_vis).astype(np.int32)

        fx, fy, cx, cy = frame.K[0, 0], frame.K[1, 1], frame.K[0, 2], frame.K[1, 2]
        X_cam = (u_vis - cx) * Z_vis / fx
        Y_cam = (v_vis - cy) * Z_vis / fy

        ptmap = np.full((H, W, 3), np.nan, dtype=np.float32)
        ptmap[v_int, u_int, 0] = -X_cam   # -X_cam → PyTorch3D X
        ptmap[v_int, u_int, 1] = -Y_cam   # -Y_cam → PyTorch3D Y
        ptmap[v_int, u_int, 2] =  Z_vis   #  Z_cam → PyTorch3D Z
        return ptmap

    def _compute_moge_affine_pointmap(self, frame, img_rgb: np.ndarray, pts_ego: np.ndarray = None):
        """
        O2: dense MoGe pointmap with global affine calibration to metric scale.

        Logic
        -----
        1. Run MoGe → relative depth map Z_moge (H, W).
        2. Project LiDAR into camera → visible metric depths Z_vis at pixels (u, v).
        3. Sample Z_moge at the LiDAR pixel locations to get (Z_moge_i, Z_lidar_i) pairs.
        4. Fit Z_metric = a*Z_moge + b via least-squares over all valid pairs.
           Using all in-image LiDAR (not just in-mask) gives a better-conditioned fit
           because the points span a wider depth range (~2–60 m vs. a narrow per-object band).
        5. Apply the transform to the full Z_moge map → metric depth everywhere.
        6. Clip unphysical negatives (can appear in sky/background where MoGe extrapolates).
        7. Recompute X, Y from metric Z and camera intrinsics K.

        Falls back to unscaled MoGe if LiDAR is unavailable or too few pairs.

        Returns (H, W, 3) float32 in PyTorch3D convention (-X_r3, -Y_r3, Z_metric).
        """
        H, W = img_rgb.shape[:2]
        fx, fy, cx, cy = frame.K[0, 0], frame.K[1, 1], frame.K[0, 2], frame.K[1, 2]

        # Step 1 — run MoGe
        _, Z_moge = self._compute_moge_pointmap(img_rgb, frame.K)   # (H, W) relative

        # Step 2 — project LiDAR
        u_vis, v_vis, Z_vis, _ = self._project_lidar(frame, img_rgb, pts_ego)

        if u_vis is None or len(u_vis) == 0:
            print(f'    [warn] no LiDAR for {frame.scene_name} frame {frame.frame_idx}, '
                  f'falling back to MoGe (no affine calibration).')
            u_grid, v_grid = np.meshgrid(np.arange(W, dtype=np.float32),
                                          np.arange(H, dtype=np.float32))
            X_r3 = (u_grid - cx) * Z_moge / fx
            Y_r3 = (v_grid - cy) * Z_moge / fy
            return np.stack([-X_r3, -Y_r3, Z_moge], axis=-1).astype(np.float32)

        # Steps 3–4 — sample MoGe at LiDAR pixel locations and fit affine
        u_int = np.round(u_vis).astype(int).clip(0, W - 1)
        v_int = np.round(v_vis).astype(int).clip(0, H - 1)
        Z_moge_at_lidar = Z_moge[v_int, u_int]

        valid = (np.isfinite(Z_moge_at_lidar)
                 & (Z_vis > 0.5)    # discard near returns (ground clutter, ego reflections)
                 & (Z_vis < 80.0))  # discard far returns where LiDAR is noisy
        n_fit = int(valid.sum())

        if n_fit < self.min_affine_pts:
            print(f'    [warn] only {n_fit} valid LiDAR/MoGe pairs for '
                  f'{frame.scene_name} frame {frame.frame_idx} '
                  f'(need {self.min_affine_pts}) — using unscaled MoGe.')
            a, b = 1.0, 0.0
        else:
            Z_m = Z_moge_at_lidar[valid]
            Z_l = Z_vis[valid]
            A   = np.stack([Z_m, np.ones_like(Z_m)], axis=1)
            (a, b), _, _, _ = np.linalg.lstsq(A, Z_l, rcond=None)

        # Steps 5–7 — apply transform, clip, rebuild pointmap
        Z_metric = np.maximum(a * Z_moge + b, 0.1).astype(np.float32)
        u_grid, v_grid = np.meshgrid(np.arange(W, dtype=np.float32),
                                      np.arange(H, dtype=np.float32))
        X_r3 = (u_grid - cx) * Z_metric / fx
        Y_r3 = (v_grid - cy) * Z_metric / fy
        return np.stack([-X_r3, -Y_r3, Z_metric], axis=-1).astype(np.float32)

    # ── O3 helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _fit_affine(Z_moge_vals, Z_lidar_vals, min_pts=4, z_min=0.5, z_max=80.0):
        """Fit Z = a*Z_moge + b via least-squares. Returns (a, b) or None if too few points."""
        valid = np.isfinite(Z_moge_vals) & (Z_lidar_vals > z_min) & (Z_lidar_vals < z_max)
        if valid.sum() < min_pts:
            return None
        Z_m = Z_moge_vals[valid]
        Z_l = Z_lidar_vals[valid]
        A = np.stack([Z_m, np.ones_like(Z_m)], axis=1)
        (a, b), _, _, _ = np.linalg.lstsq(A, Z_l, rcond=None)
        return a, b

    @staticmethod
    def _build_ptmap_from_affine(Z_moge_map, K, a, b):
        """Apply Z_metric = a*Z_moge + b to full map and build PyTorch3D pointmap."""
        H, W = Z_moge_map.shape
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        Z_metric = np.maximum(a * Z_moge_map + b, 0.1).astype(np.float32)
        u_grid, v_grid = np.meshgrid(np.arange(W, dtype=np.float32),
                                      np.arange(H, dtype=np.float32))
        X_r3 = (u_grid - cx) * Z_metric / fx
        Y_r3 = (v_grid - cy) * Z_metric / fy
        return np.stack([-X_r3, -Y_r3, Z_metric], axis=-1).astype(np.float32)

    def _compute_local_affine_ptmap_for_object(
        self, frame, Z_moge_map, pts_ego_vis, u_vis, v_vis, Z_vis, binary_mask,
        min_cluster_size=3, min_samples=1, cluster_eps=0.5, min_pts=4,
        z_min=0.5, z_max=80.0,
    ):
        """
        O3: per-object MoGe + HDBSCAN local affine calibration.

        MoGe runs once per frame (Z_moge_map shared); this method fits a separate
        (a, b) for each object using only its clean in-mask LiDAR surface points.

        Returns
        -------
        ptmap : (H, W, 3) float32  metric pointmap in PyTorch3D convention
        a, b  : float  fitted affine coefficients
        mode  : str    'local' | 'global_fallback' | 'unscaled_fallback'
        """
        H, W = Z_moge_map.shape
        u_int = np.round(u_vis).astype(int).clip(0, W - 1)
        v_int = np.round(v_vis).astype(int).clip(0, H - 1)

        # Step 1 — find in-mask LiDAR points
        in_mask    = binary_mask[v_int, u_int]
        pts_inmask = pts_ego_vis[in_mask]
        Z_inmask   = Z_vis[in_mask]
        u_inmask   = u_vis[in_mask]
        v_inmask   = v_vis[in_mask]
        Z_moge_all = Z_moge_map[v_int, u_int]

        if len(pts_inmask) < min_cluster_size:
            ab = self._fit_affine(Z_moge_all, Z_vis, min_pts=min_pts,
                                  z_min=z_min, z_max=z_max)
            if ab is None:
                return self._build_ptmap_from_affine(Z_moge_map, frame.K, 1.0, 0.0), 1.0, 0.0, 'unscaled_fallback'
            return self._build_ptmap_from_affine(Z_moge_map, frame.K, *ab), ab[0], ab[1], 'global_fallback'

        # Step 2 — HDBSCAN in 3D ego space (shared utility)
        keep = filter_inmask_lidar_hdbscan(
            pts_inmask,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_eps=cluster_eps,
        )

        if keep is None:
            ab = self._fit_affine(Z_moge_all, Z_vis, min_pts=min_pts,
                                  z_min=z_min, z_max=z_max)
            if ab is None:
                return self._build_ptmap_from_affine(Z_moge_map, frame.K, 1.0, 0.0), 1.0, 0.0, 'unscaled_fallback'
            return self._build_ptmap_from_affine(Z_moge_map, frame.K, *ab), ab[0], ab[1], 'global_fallback'

        # Step 3 — fit affine on dominant cluster
        u_clean_int = np.round(u_inmask[keep]).astype(int).clip(0, W - 1)
        v_clean_int = np.round(v_inmask[keep]).astype(int).clip(0, H - 1)
        Z_moge_clean = Z_moge_map[v_clean_int, u_clean_int]
        Z_clean      = Z_inmask[keep]

        ab = self._fit_affine(Z_moge_clean, Z_clean, min_pts=min_pts,
                              z_min=z_min, z_max=z_max)
        if ab is None:
            ab = self._fit_affine(Z_moge_all, Z_vis, min_pts=min_pts,
                                  z_min=z_min, z_max=z_max)
            if ab is None:
                return self._build_ptmap_from_affine(Z_moge_map, frame.K, 1.0, 0.0), 1.0, 0.0, 'unscaled_fallback'
            return self._build_ptmap_from_affine(Z_moge_map, frame.K, *ab), ab[0], ab[1], 'global_fallback'

        # Step 4 — apply to full frame
        return self._build_ptmap_from_affine(Z_moge_map, frame.K, *ab), ab[0], ab[1], 'local'

    @staticmethod
    def _mesh_to_r3(output):
        """Convert SAM3D Objects output from object-local space to R3 camera space."""
        from pytorch3d.renderer import look_at_view_transform
        from pytorch3d.transforms import Transform3d, quaternion_to_matrix
        from sam3d_objects.data.dataset.tdfy.transforms_3d import compose_transform

        def _sq(t):
            while t.dim() > 2:
                t = t.squeeze(0)
            return t

        quat  = _sq(output['rotation'].detach().cpu().float())
        trans = _sq(output['translation'].detach().cpu().float())
        scale = _sq(output['scale'].detach().cpu().float())
        if quat.dim() == 1:
            quat = quat.unsqueeze(0)

        l2c = compose_transform(
            scale=scale, rotation=quaternion_to_matrix(quat), translation=trans
        )
        R_r3_to_p3d, _ = look_at_view_transform(
            eye=torch.tensor([[0., 0., -1.]]),
            at=torch.tensor([[0., 0., 0.]]),
            up=torch.tensor([[0., -1., 0.]]),
        )
        t_p3d_to_r3 = Transform3d().rotate(R_r3_to_p3d).inverse()
        mesh_res    = output['mesh'][0]
        verts_local = mesh_res.vertices.float().cpu()
        faces       = mesh_res.faces.long().cpu().numpy().astype(np.int32)
        xyz_p3d     = l2c.transform_points(verts_local.unsqueeze(0)).squeeze(0)
        verts_r3    = (t_p3d_to_r3.transform_points(xyz_p3d.unsqueeze(0))
                       .squeeze(0).numpy().astype(np.float32))
        return verts_r3, faces

    # ── Frame inference ───────────────────────────────────────────────────────

    def run_frame(self, frame, frame_sam3: Dict[str, list], pts_ego: np.ndarray = None) -> list:
        """
        Run pointmap computation + SAM3D Objects for all non-pedestrian detections.

        Parameters
        ----------
        frame      : FrameRecord
        frame_sam3 : SAM3 segmentation results for this frame
        pts_ego    : optional (N, 3) pre-loaded ego-frame point cloud (e.g. aggregated).
                     If None, falls back to loading the single sweep from frame.lidar_path.
        """
        img_rgb, _ = frame.load_images()

        # ── O3: MoGe runs once per frame; per-object affine computed inside the loop ──
        if self.pointmap_mode == 'o3_local_affine':
            _, Z_moge_map = self._compute_moge_pointmap(img_rgb, frame.K)
            u_vis, v_vis, Z_vis, pts_ego_vis = self._project_lidar(frame, img_rgb, pts_ego)
            if u_vis is None:
                print(f'    [warn] no LiDAR for {frame.scene_name} frame {frame.frame_idx}, '
                      f'falling back to unscaled MoGe for all objects.')
            ptmap_t = None   # built per-object below

        # ── All other modes: one pointmap for the whole frame ─────────────────────────
        else:
            if self.pointmap_mode == 'o1_lidar':
                ptmap = self._compute_lidar_pointmap(frame, img_rgb, pts_ego)
            elif self.pointmap_mode == 'o2_moge_affine':
                ptmap = self._compute_moge_affine_pointmap(frame, img_rgb, pts_ego)
            else:   # 'baseline'
                ptmap, _ = self._compute_moge_pointmap(img_rgb, frame.K)
            ptmap_t = torch.tensor(ptmap, dtype=torch.float32)

        results = []
        for prompt, pipeline_type in self.prompts.items():
            if pipeline_type != 'objects':
                continue
            dets = frame_sam3.get(prompt, [])
            if not dets:
                continue
            print(f'    [{frame.scene_name} frame {frame.frame_idx}] '
                  f'"{prompt}": {len(dets)} instance(s)...')
            for d in dets:
                try:
                    if self.pointmap_mode == 'o3_local_affine':
                        if u_vis is not None:
                            _hp = self.hdbscan_params.get(prompt, {})
                            ptmap_obj, a, b, mode = self._compute_local_affine_ptmap_for_object(
                                frame, Z_moge_map, pts_ego_vis, u_vis, v_vis, Z_vis,
                                d['binary_mask'],
                                min_cluster_size=_hp.get('min_cluster_size', 3),
                                min_samples=_hp.get('min_samples', 1),
                                cluster_eps=_hp.get('cluster_eps', 0.5),
                            )
                            print(f'      [{mode}]  Z = {a:.4f}*Z_moge + {b:.4f}')
                        else:
                            ptmap_obj, _, _ = self._compute_moge_pointmap(img_rgb, frame.K)
                        ptmap_t_use = torch.tensor(ptmap_obj, dtype=torch.float32)
                    else:
                        ptmap_t_use = ptmap_t

                    with suppress_output():
                        out = self._inference(img_rgb, d['binary_mask'], seed=42, pointmap=ptmap_t_use)
                    verts_r3, faces = self._mesh_to_r3(out)
                    results.append({
                        'vertices':    verts_r3,
                        'faces':       faces,
                        'binary_mask': d['binary_mask'],
                        'score':       d['score'],
                        'prompt':      prompt,
                    })
                except Exception as e:
                    print(f'    [WARN] SAM3D Objects failed for "{prompt}": {e}')
        return results

    def run_batch(self, frames, sam3_results: Dict[int, Dict[str, list]]) -> Dict[int, list]:
        """Run all frames. Returns {frame_idx: [object result dicts]}."""
        assert self._inference is not None, 'Call load() before run_batch()'
        results = {}
        for i, frame in enumerate(frames):
            results[i] = self.run_frame(frame, sam3_results[i])
        return results
