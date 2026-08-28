"""
SAM3D Objects inference wrapper.

Pointmap modes (set sam3d_objects.pointmap_mode in config):
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
  'o4_ground_filter'  — CompletionFormer fuses sparse LiDAR + RGB → dense metric depth
                       without MoGe. Global PseudoLabeler ground filter removes road-surface
                       returns before building the sparse depth map. One pointmap per camera
                       view, shared across all objects. Requires cformer_ckpt.
  'o5_mask_hdbscan'   — CompletionFormer as in O4, but sparse depth map is built from
                       HDBSCAN-filtered in-mask LiDAR (dominant surface cluster per object)
                       merged with raw out-of-mask points. No PseudoLabeler ground filter —
                       HDBSCAN implicitly rejects near-ground clutter within masks, and
                       out-of-mask ground returns are useful background depth anchors.
                       Requires cformer_ckpt.
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
        pointmap_mode: str = 'baseline',   # 'baseline'|'o1_lidar'|'o2_moge_affine'|'o3_local_affine'|'o4_ground_filter'|'o5_mask_hdbscan'
        min_affine_pts: int = 5,           # minimum LiDAR/MoGe pairs for a reliable affine fit
        hdbscan_params: Optional[Dict] = None,  # per-class HDBSCAN params (O3 and O5)
        cformer_ckpt: Optional[str] = None,     # path to CompletionFormer checkpoint (O4/O5)
        lidar_lines: int = 32,                  # LiDAR beam count (32 nuScenes / 64 ECP) for O4/O5
        pl_ground_inlier_thres: float = 0.10,   # O4 only: max height above ground surface to keep [m]
        ss_correction: bool = False,            # THESIS: enable mid-pipeline SS voxel correction (requires O4/O5)
        proximity_min_pts: int = 30,            # O5 proximity selection: min size of largest cluster
        proximity_ratio: float = 0.70,          # O5 proximity selection: second/largest ratio threshold
        hull_anchoring: bool = False,           # O4/O5: post-CFormer convex-hull depth clamp
        mask_erode_px: int = 0,                 # O3: erode binary mask before HDBSCAN point selection (0 = off)
        mask_erode_min_px: int = 0,             # O3: min mask area [px] to apply erosion (0 = always)
    ):
        self.repo_path              = str(repo_path)
        self.config_path            = str(config_path)
        self.prompts                = prompts
        self.device                 = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.pointmap_mode          = pointmap_mode
        self.min_affine_pts         = min_affine_pts
        self.hdbscan_params         = hdbscan_params or {}
        self.cformer_ckpt           = str(cformer_ckpt) if cformer_ckpt else None
        self.lidar_lines            = lidar_lines
        self.pl_ground_inlier_thres = pl_ground_inlier_thres
        self.ss_correction          = ss_correction   # THESIS: SS Correction flag
        self.proximity_min_pts      = proximity_min_pts
        self.proximity_ratio        = proximity_ratio
        self.hull_anchoring         = hull_anchoring
        self.mask_erode_px          = mask_erode_px
        self.mask_erode_min_px      = mask_erode_min_px
        self._inference             = None
        self._cformer               = None

    def load(self) -> None:
        # SAM3D Objects uses LIDRA_SKIP_INIT to avoid importing hardware-specific
        # drivers at load time. Must be set before any sam3d_objects import.
        os.environ['LIDRA_SKIP_INIT'] = 'true'

        for _p in [str(Path(self.repo_path) / 'notebook'), str(self.repo_path)]:
            if _p not in sys.path:
                sys.path.insert(0, _p)

        from inference import Inference as SAM3DObjectsInference
        with suppress_output():
            self._inference = SAM3DObjectsInference(self.config_path, compile=False)
        print('  SAM3D Objects loaded.')

        if self.pointmap_mode in ('o4_ground_filter', 'o5_mask_hdbscan'):
            self._load_cformer()

    def _load_cformer(self) -> None:
        import argparse
        if self.cformer_ckpt is None:
            raise ValueError(f"pointmap_mode='{self.pointmap_mode}' requires cformer_ckpt to be set in config.")
        ckpt_path = Path(self.cformer_ckpt)
        if not ckpt_path.exists():
            raise FileNotFoundError(f'CompletionFormer checkpoint not found: {ckpt_path}')

        cformer_src = str(ckpt_path.parent.parent.parent / 'src')
        cformer_dcn = str(ckpt_path.parent.parent.parent / 'src' / 'model' / 'deformconv')
        for _p in [cformer_src, cformer_dcn]:
            if _p not in sys.path:
                sys.path.insert(0, _p)

        from model.completionformer import CompletionFormer as _CompletionFormer
        args = argparse.Namespace(
            data_name='KITTIDC', prop_time=6, prop_kernel=3,
            preserve_input=True, affinity='TGASS', affinity_gamma=0.5,
            conf_prop=True, legacy=False, from_scratch=False,
            max_depth=90.0, num_sample=500, lidar_lines=self.lidar_lines,
        )
        self._cformer = _CompletionFormer(args).cuda()
        ckpt = torch.load(str(ckpt_path), map_location='cuda', weights_only=False)
        self._cformer.load_state_dict(ckpt.get('net', ckpt), strict=True)
        self._cformer.eval()
        n = sum(p.numel() for p in self._cformer.parameters())
        print(f'  CompletionFormer loaded ({n/1e6:.1f} M params, {self.lidar_lines}-beam).')

    def unload(self) -> None:
        del self._inference
        self._inference = None
        if self._cformer is not None:
            del self._cformer
            self._cformer = None
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

        # Step 1 — find in-mask LiDAR points.
        # Optionally erode the mask before HDBSCAN point selection to reduce
        # border bleed-in from adjacent objects.  The un-eroded mask is kept
        # for affine fit and pointmap reconstruction (only HDBSCAN is affected).
        if self.mask_erode_px > 0 and int(binary_mask.sum()) >= self.mask_erode_min_px:
            import cv2 as _cv2o3
            _er_k = _cv2o3.getStructuringElement(
                _cv2o3.MORPH_ELLIPSE,
                (2 * self.mask_erode_px + 1, 2 * self.mask_erode_px + 1))
            _mask_hdbscan = _cv2o3.erode(
                binary_mask.astype(np.uint8), _er_k).astype(bool)
        else:
            _mask_hdbscan = binary_mask
        in_mask    = _mask_hdbscan[v_int, u_int]
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

    def _compute_dense_completion_pointmap(
        self, img_rgb: np.ndarray, K: np.ndarray,
        u_vis: np.ndarray, v_vis: np.ndarray, Z_vis: np.ndarray,
    ):
        """
        O4/O5: CompletionFormer dense metric depth → PyTorch3D pointmap.

        Projects the pre-filtered LiDAR points into a sparse (H,W) depth map
        (min-depth per pixel), runs CompletionFormer, and back-projects the dense
        output to a (H,W,3) pointmap in PyTorch3D convention (-X_cam, -Y_cam, Z_cam).

        Returns
        -------
        ptmap      : (H,W,3) float32  PyTorch3D pointmap
        dense_depth: (H,W)   float32  raw CFormer metric depth [m]
        """
        H, W = img_rgb.shape[:2]

        # Sparse depth map — min depth per pixel
        u_int = np.clip(np.round(u_vis).astype(int), 0, W - 1)
        v_int = np.clip(np.round(v_vis).astype(int), 0, H - 1)
        _tmp = np.full((H, W), np.inf, dtype=np.float32)
        np.minimum.at(_tmp, (v_int, u_int), Z_vis.astype(np.float32))
        sparse_depth = np.where(_tmp < np.inf, _tmp, 0.0).astype(np.float32)

        # CompletionFormer inference
        rgb_t = torch.from_numpy(img_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).cuda()
        dep_t = torch.from_numpy(sparse_depth).unsqueeze(0).unsqueeze(0).cuda()
        with torch.no_grad():
            dense_depth = self._cformer({'rgb': rgb_t, 'dep': dep_t})['pred'].squeeze().cpu().numpy()

        ptmap = self._depth_to_ptmap(dense_depth, K, H, W)
        return ptmap, dense_depth

    @staticmethod
    def _depth_to_ptmap(dense_d: np.ndarray, K: np.ndarray, H: int, W: int) -> np.ndarray:
        """Back-project (H,W) depth map → (H,W,3) ptmap in PyTorch3D convention (-X,-Y,Z)."""
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        v_grid, u_grid = np.mgrid[0:H, 0:W]
        Z = dense_d.astype(np.float64)
        X = (u_grid - cx) / fx * Z
        Y = (v_grid - cy) / fy * Z
        return np.stack([-X, -Y, Z], axis=-1).astype(np.float32)

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

    # ── PseudoLabeler ground filter ───────────────────────────────────────────

    def _filter_above_ground(self, pts_ego_vis, u_vis, v_vis, Z_vis, pl_model):
        """Remove in-image LiDAR points classified as ground by PseudoLabeler.

        Used in O4 mode before building the CFormer sparse depth map so that
        ground-plane returns do not anchor incorrect depths for object pixels.

        Parameters
        ----------
        pts_ego_vis : (M, 3) ego-frame 3-D coordinates of visible LiDAR points
        u_vis, v_vis : (M,) pixel columns / rows
        Z_vis        : (M,) metric depths [m]
        pl_model     : fitted PseudoLabeler instance

        Returns
        -------
        Filtered (pts_ego_vis, u_vis, v_vis, Z_vis) with ground points removed.
        """
        pc_t = torch.tensor(pts_ego_vis, dtype=torch.float32).to(self.device)
        pl_dev = pl_model.to(self.device)
        bool_ground = pl_dev.get_ground_bool(pc_t, inlier_thres=self.pl_ground_inlier_thres)
        keep = (~bool_ground).cpu().numpy()
        n_in, n_keep = len(pts_ego_vis), int(keep.sum())
        print(f'    [PL ground filter] kept {n_keep:,}/{n_in:,} pts '
              f'(removed {n_in - n_keep:,} ground, thres={self.pl_ground_inlier_thres} m)')
        return pts_ego_vis[keep], u_vis[keep], v_vis[keep], Z_vis[keep]

    # ── O5: per-mask HDBSCAN LiDAR cleaning ──────────────────────────────────

    def _build_o5_sparse_points(self, pts_ego_vis, u_vis, v_vis, Z_vis, H, W, frame_sam3):
        """Build clean sparse depth inputs for CFormer using per-mask HDBSCAN.

        For each SAM3 segmentation mask: cluster in-mask LiDAR points in 3D
        ego space, keep only the dominant (object surface) cluster, discard the
        rest (near-ground clutter, background bleeds).  Out-of-mask points are
        kept unchanged — road/building returns outside object masks are useful
        background depth anchors for CompletionFormer.

        Failure cases (too-few pts or all-noise): in-mask pts are REMOVED so
        CFormer completes from image priors + out-of-mask anchors, rather than
        being contaminated by scattered noise or occluder anchors.

        Proximity selection: when two large clusters both qualify (both ≥
        proximity_min_pts and second ≥ proximity_ratio × largest), pick the
        CLOSER one to avoid a background cluster dominating.

        clean_sparse = (HDBSCAN-filtered in-mask pts) ∪ (all out-of-mask pts)

        Returns u_clean, v_clean, Z_clean suitable for _compute_dense_completion_pointmap.
        """
        import hdbscan as _hdbscan

        u_int = np.round(u_vis).astype(int).clip(0, W - 1)
        v_int = np.round(v_vis).astype(int).clip(0, H - 1)
        keep  = np.ones(len(u_vis), dtype=bool)

        n_masks, n_removed = 0, 0
        for prompt, pipeline_type in self.prompts.items():
            if pipeline_type != 'objects':
                continue
            hp   = self.hdbscan_params.get(prompt, {})
            mcs  = hp.get('min_cluster_size', 3)
            ms   = hp.get('min_samples', 1)
            eps  = hp.get('cluster_eps', 0.4)
            for d in frame_sam3.get(prompt, []):
                binary_mask = d['binary_mask']
                in_mask     = binary_mask[v_int, u_int]
                idx_in      = np.where(in_mask)[0]
                if len(idx_in) == 0:
                    continue
                n_masks += 1
                pts_in = pts_ego_vis[idx_in]
                if len(pts_in) < mcs:
                    # too few to cluster reliably — remove noisy anchors,
                    # let CFormer complete from image priors
                    keep[idx_in] = False
                    n_removed += len(idx_in)
                    continue
                clusterer = _hdbscan.HDBSCAN(
                    min_cluster_size=mcs, min_samples=ms,
                    metric='euclidean', cluster_selection_epsilon=eps,
                )
                labels = clusterer.fit_predict(pts_in)
                unique, counts = np.unique(labels[labels >= 0], return_counts=True)
                if len(unique) == 0:
                    # all noise — remove in-mask pts, let CFormer use image priors
                    keep[idx_in] = False
                    n_removed += len(idx_in)
                    continue
                order   = np.argsort(counts)[::-1]
                biggest = int(counts[order[0]])
                if (biggest >= self.proximity_min_pts
                        and len(order) >= 2
                        and counts[order[1]] >= self.proximity_ratio * biggest):
                    c0, c1 = unique[order[0]], unique[order[1]]
                    d0 = float(np.linalg.norm(pts_in[labels == c0].mean(axis=0)))
                    d1 = float(np.linalg.norm(pts_in[labels == c1].mean(axis=0)))
                    dominant = c0 if d0 <= d1 else c1
                else:
                    dominant = unique[order[0]]
                noise_idx = idx_in[labels != dominant]
                keep[noise_idx] = False
                n_removed += len(noise_idx)

        print(f'    [O5 HDBSCAN] {n_masks} masks, removed {n_removed:,} in-mask noise pts '
              f'({keep.sum():,}/{len(u_vis):,} kept)')
        return u_vis[keep], v_vis[keep], Z_vis[keep]

    # ── Hull anchoring (O4/O5 post-CFormer depth clamp) ──────────────────────

    def _apply_hull_anchoring(
        self,
        dense_depth: np.ndarray,
        pts_ego_vis: np.ndarray,
        u_vis: np.ndarray, v_vis: np.ndarray, Z_vis: np.ndarray,
        H: int, W: int,
        frame_sam3: dict,
    ) -> np.ndarray:
        """Per-mask convex-hull depth clamp applied after CompletionFormer.

        For each object mask, run HDBSCAN on in-mask LiDAR to find the dominant
        surface cluster, derive min_depth = min(cluster cam-depth), then clamp:

            dense_depth[hull] = max(dense_depth[hull], min_depth)

        Prevents occluder (pole/sign) LiDAR depth from bleeding into object
        interiors through CFormer interpolation.

        Parameters
        ----------
        dense_depth  : (H,W) float32 — CFormer output (modified in-place on copy)
        pts_ego_vis  : (N,3) float32 — ego-frame LiDAR visible in camera
        u_vis,v_vis,Z_vis : (N,) float32
        H, W         : int
        frame_sam3   : dict[prompt → list[det]]

        Returns
        -------
        dense_anchored : (H,W) float32
        """
        import cv2 as _cv2
        import hdbscan as _hdbscan

        dense_anchored = dense_depth.copy()
        u_int = np.round(u_vis).astype(int).clip(0, W - 1)
        v_int = np.round(v_vis).astype(int).clip(0, H - 1)
        n_anchored = 0

        for prompt, pipeline_type in self.prompts.items():
            if pipeline_type != 'objects':
                continue
            hp  = self.hdbscan_params.get(prompt, {})
            mcs = hp.get('min_cluster_size', 3)
            ms  = hp.get('min_samples', 1)
            eps = hp.get('cluster_eps', 0.4)

            for mask_idx, d in enumerate(frame_sam3.get(prompt, [])):
                binary_mask  = d['binary_mask']
                in_mask_bool = binary_mask[v_int, u_int]
                idx_in       = np.where(in_mask_bool)[0]
                _tag = f'    [Hull] "{prompt}" mask {mask_idx}'

                if len(idx_in) < mcs:
                    print(f'{_tag} — SKIP: only {len(idx_in)} in-mask pts (need ≥{mcs})')
                    continue

                pts_in = pts_ego_vis[idx_in]
                Z_in   = Z_vis[idx_in]
                print(f'{_tag} — {len(idx_in)} in-mask pts, HDBSCAN(mcs={mcs}, ms={ms}, ε={eps})')

                clusterer = _hdbscan.HDBSCAN(
                    min_cluster_size=mcs, min_samples=ms,
                    metric='euclidean', cluster_selection_epsilon=eps,
                )
                labels = clusterer.fit_predict(pts_in)
                n_noise = int((labels == -1).sum())
                unique, counts = np.unique(labels[labels >= 0], return_counts=True)

                if len(unique) == 0:
                    print(f'{_tag}   → HDBSCAN: all {len(idx_in)} pts noise — SKIP')
                    continue

                order    = np.argsort(counts)[::-1]
                _cl_info = ', '.join(f'C{unique[i]}:{counts[i]}' for i in order)
                print(f'{_tag}   → HDBSCAN: {len(unique)} cluster(s) [{_cl_info}], noise={n_noise}')

                biggest = int(counts[order[0]])
                if (biggest >= self.proximity_min_pts
                        and len(order) >= 2
                        and counts[order[1]] >= self.proximity_ratio * biggest):
                    c0, c1 = unique[order[0]], unique[order[1]]
                    d0 = float(np.linalg.norm(pts_in[labels == c0].mean(axis=0)))
                    d1 = float(np.linalg.norm(pts_in[labels == c1].mean(axis=0)))
                    dominant = c0 if d0 <= d1 else c1
                    _other   = c1 if dominant == c0 else c0
                    print(f'{_tag}   → proximity selection: C{c0} dist={d0:.1f}m, '
                          f'C{c1} dist={d1:.1f}m → chose C{dominant} (closer)')
                else:
                    dominant = unique[order[0]]
                    print(f'{_tag}   → largest cluster C{dominant} chosen (size={biggest})')

                z_dominant = Z_in[labels == dominant]
                min_depth  = float(z_dominant.min())
                print(f'{_tag}   → dominant: {int((labels==dominant).sum())} pts, '
                      f'Z=[{z_dominant.min():.2f}–{z_dominant.max():.2f}m]  min_depth={min_depth:.2f}m')

                if min_depth <= 0:
                    print(f'{_tag}   → min_depth ≤ 0 — SKIP')
                    continue

                ys, xs = np.where(binary_mask)
                if len(ys) < 3:
                    print(f'{_tag}   → mask too small for convex hull — SKIP')
                    continue
                pts_2d   = np.stack([xs, ys], axis=1).astype(np.float32)
                hull     = _cv2.convexHull(pts_2d)
                hull_img = np.zeros((H, W), dtype=np.uint8)
                _cv2.fillConvexPoly(hull_img, hull.astype(np.int32), 1)
                hull_bool = hull_img.astype(bool)

                _before = dense_anchored[hull_bool]
                dense_anchored[hull_bool] = np.maximum(_before, min_depth)
                _n_clamped = int((_before < min_depth).sum())
                print(f'{_tag}   → hull={hull_bool.sum():,} px, clamped {_n_clamped:,} px to ≥{min_depth:.2f}m')
                n_anchored += 1

        print(f'    [Hull anchoring] done: {n_anchored} mask(s) anchored')
        return dense_anchored

    # ── THESIS MODIFICATION: SS LiDAR Correction ─────────────────────────────
    # Builds the ss_correction_fn closure passed to InferencePipelinePointMap.run().
    # Called once per object crop, captures the per-camera CFormer dense depth map
    # and the LiDAR anchor mask for that frame.
    #
    # ss_correction: bool  — master switch (from config ss_correction: true/false)
    # dense_depth: (H,W)   — metric depth from CFormer (metres, float32)
    # anchor_mask: (H,W)   — bool, True where pixel is hard-anchored to LiDAR
    # K:           (3,3)   — camera intrinsics for this camera
    # binary_mask: (H,W)   — SAM3 binary mask for this specific object crop
    #
    # Returns a callable(coords, ss_return_dict) → coords, or None if disabled.
    #
    # The voxel grid uses SAM3D Objects' internal normalised object-centric space.
    # coords shape: (N, 4) — [batch_idx, z, y, x] in [0, 63] voxel indices.
    # After pose_decoder, ss_return_dict contains 'translation' (P3D camera space)
    # and 'scale' (metric). We use these to map dense_depth pixels → voxel indices
    # and suppress voxels where the surface lies clearly in front of them.

    def _build_ss_correction_fn(self, dense_depth, anchor_mask, K, binary_mask):
        """Build the SS correction callable for one object crop.

        Suppresses voxels that the SS diffusion model hallucinated behind the
        visible surface at LiDAR-anchored pixels. CFormer-interpolated pixels
        do not contribute to suppression (may be inaccurate) — future extension
        can add confirmation from those pixels.

        Args:
            dense_depth  : (H,W) float32 CFormer dense depth in metres
            anchor_mask  : (H,W) bool, True = pixel is hard-LiDAR-anchored
            K            : (3,3) camera intrinsics
            binary_mask  : (H,W) bool SAM3 object mask

        Returns:
            Callable (coords, ss_return_dict) → corrected coords tensor, or
            None if no correction can be applied (e.g. no anchors in mask).
        """
        # Find in-mask LiDAR-anchored pixels for suppression
        in_mask_anchor = binary_mask & anchor_mask  # (H,W) bool
        n_anchors = int(in_mask_anchor.sum())
        if n_anchors < 3:
            return None  # too few anchors — skip correction for this object

        # Median surface depth from LiDAR-anchored pixels inside the mask
        v_anch, u_anch = np.where(in_mask_anchor)
        Z_anch = dense_depth[v_anch, u_anch].astype(np.float64)   # metres, camera Z

        def _correction_fn(coords, ss_return_dict):
            """coords: (N,4) int tensor [batch,z,y,x] in [0,63] voxel space."""
            try:
                # Recover object pose from ss_return_dict
                import torch as _torch
                from pytorch3d.transforms import quaternion_to_matrix as _q2R

                def _sq(t):
                    while t.dim() > 2:
                        t = t.squeeze(0)
                    return t

                # .view() to known shapes — decompose_transform returns (1,4)/(1,3)/(1,3)
                quat  = ss_return_dict['rotation'].detach().cpu().float().view(1, 4)    # (1,4)
                trans = ss_return_dict['translation'].detach().cpu().float().view(3)    # (3,)
                scale = ss_return_dict['scale'].detach().cpu().float().mean().item()    # scalar

                R = _q2R(quat).squeeze(0)  # (3,3) — P3D object → P3D world (= camera here)

                # Voxel centres in normalised object space: coords[:,1:] in [0,63]
                # Model uses: voxel_norm = coords[:,1:] / 64 - 0.5 → [-0.5, 0.5]
                vox_norm = coords[:, 1:].float().cpu() / 64.0 - 0.5  # (N,3) — .cpu() avoids device mismatch with R

                # Object-local → P3D camera: p_cam = scale * R @ v_local + trans
                # (N,3) @ (3,3) → (N,3) ; + (3,) broadcasts cleanly → (N,3)
                vox_p3d = (scale * (vox_norm @ R.T) + trans)  # (N,3)

                # Convert P3D → R3 camera for depth comparison: Z is the same, X/Y flip
                vox_Z = vox_p3d[:, 2].numpy()  # depth component (same in R3 and P3D)

                # For each voxel, find the closest anchor pixel by projecting voxel
                # centres back into the image and reading the anchor depth there.
                # Approximate: use the median anchor depth as the surface reference
                # (fast; per-voxel projection is expensive and not needed for suppression).
                Z_surface = float(np.median(Z_anch))

                # Suppress voxels that are clearly BEHIND the surface at LiDAR-anchored pixels.
                # Threshold: 0.3 m — tight enough to suppress background, loose enough to keep
                # the object body (objects are not flat; some voxels legitimately span depth).
                suppress_margin = 0.3  # metres
                behind_surface = _torch.from_numpy(vox_Z > Z_surface + suppress_margin)

                n_before = len(coords)
                coords = coords[~behind_surface]
                n_after = len(coords)
                if n_before > n_after:
                    print(f'      [SS-corr] suppressed {n_before - n_after}/{n_before} '
                          f'voxels behind surface (Z_surf={Z_surface:.2f} m, '
                          f'n_anchors={n_anchors})')
                return coords

            except Exception as _e:
                print(f'      [SS-corr] skipped ({_e})')
                return coords

        return _correction_fn

    # ── END THESIS MODIFICATION ───────────────────────────────────────────────

    # ── Frame inference ───────────────────────────────────────────────────────

    def run_frame(self, frame, frame_sam3: Dict[str, list], pts_ego: np.ndarray = None,
                  pl_model=None) -> list:
        """
        Run pointmap computation + SAM3D Objects for all non-pedestrian detections.

        Parameters
        ----------
        frame      : FrameRecord
        frame_sam3 : SAM3 segmentation results for this frame
        pts_ego    : optional (N, 3) pre-loaded ego-frame point cloud (e.g. aggregated).
                     If None, falls back to loading the single sweep from frame.lidar_path.
        pl_model   : optional fitted PseudoLabeler instance (O4 only). When provided,
                     ground returns are removed from the sparse LiDAR depth map before
                     passing to CompletionFormer.
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

        # ── O4: CompletionFormer + global PseudoLabeler ground filter ────────────────
        elif self.pointmap_mode == 'o4_ground_filter':
            u_vis, v_vis, Z_vis, pts_ego_vis = self._project_lidar(frame, img_rgb, pts_ego)
            _dense_depth_ss = None   # THESIS: for SS correction
            _anchor_mask_ss = None   # THESIS: for SS correction
            if u_vis is None:
                print(f'    [warn] no LiDAR for {frame.scene_name} frame {frame.frame_idx}, '
                      f'falling back to MoGe baseline.')
                ptmap, _ = self._compute_moge_pointmap(img_rgb, frame.K)
            else:
                H, W = img_rgb.shape[:2]
                if pl_model is not None:
                    pts_ego_vis, u_vis, v_vis, Z_vis = self._filter_above_ground(
                        pts_ego_vis, u_vis, v_vis, Z_vis, pl_model)
                ptmap, dense_depth = self._compute_dense_completion_pointmap(
                    img_rgb, frame.K, u_vis, v_vis, Z_vis)
                if self.hull_anchoring:
                    dense_depth = self._apply_hull_anchoring(
                        dense_depth, pts_ego_vis, u_vis, v_vis, Z_vis, H, W, frame_sam3)
                    ptmap = self._depth_to_ptmap(dense_depth, frame.K, H, W)
                # THESIS: build anchor mask for SS correction
                if self.ss_correction:
                    _dense_depth_ss = dense_depth.copy()
                    _anchor_mask_ss = np.zeros((H, W), dtype=bool)
                    _u_int = np.clip(np.round(u_vis).astype(int), 0, W - 1)
                    _v_int = np.clip(np.round(v_vis).astype(int), 0, H - 1)
                    _anchor_mask_ss[_v_int, _u_int] = True
            ptmap_t = torch.tensor(ptmap, dtype=torch.float32)

        # ── O5: CompletionFormer + per-mask HDBSCAN anchor cleaning ──────────────────
        elif self.pointmap_mode == 'o5_mask_hdbscan':
            u_vis, v_vis, Z_vis, pts_ego_vis = self._project_lidar(frame, img_rgb, pts_ego)
            _dense_depth_ss = None   # THESIS: for SS correction
            _anchor_mask_ss = None   # THESIS: for SS correction
            if u_vis is None:
                print(f'    [warn] no LiDAR for {frame.scene_name} frame {frame.frame_idx}, '
                      f'falling back to MoGe baseline.')
                ptmap, _ = self._compute_moge_pointmap(img_rgb, frame.K)
            else:
                H, W = img_rgb.shape[:2]
                u_clean, v_clean, Z_clean = self._build_o5_sparse_points(
                    pts_ego_vis, u_vis, v_vis, Z_vis, H, W, frame_sam3)
                ptmap, dense_depth = self._compute_dense_completion_pointmap(
                    img_rgb, frame.K, u_clean, v_clean, Z_clean)
                if self.hull_anchoring:
                    dense_depth = self._apply_hull_anchoring(
                        dense_depth, pts_ego_vis, u_vis, v_vis, Z_vis, H, W, frame_sam3)
                    ptmap = self._depth_to_ptmap(dense_depth, frame.K, H, W)
                # THESIS: build anchor mask for SS correction (anchored at HDBSCAN-cleaned pixels)
                if self.ss_correction:
                    _dense_depth_ss = dense_depth.copy()
                    _anchor_mask_ss = np.zeros((H, W), dtype=bool)
                    _u_int = np.clip(np.round(u_clean).astype(int), 0, W - 1)
                    _v_int = np.clip(np.round(v_clean).astype(int), 0, H - 1)
                    _anchor_mask_ss[_v_int, _u_int] = True
                u_vis, v_vis, Z_vis = u_clean, v_clean, Z_clean
            ptmap_t = torch.tensor(ptmap, dtype=torch.float32)

        # ── All other modes: one pointmap for the whole frame ─────────────────────────
        else:
            _dense_depth_ss = None   # THESIS: SS correction not available without CFormer
            _anchor_mask_ss = None
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

                    # THESIS: build SS correction fn if enabled (O4/O5 only — needs dense_depth + anchor_mask)
                    _ss_fn = None
                    if self.ss_correction and _dense_depth_ss is not None and _anchor_mask_ss is not None:
                        _ss_fn = self._build_ss_correction_fn(
                            _dense_depth_ss, _anchor_mask_ss, frame.K, d['binary_mask'])
                    with suppress_output():
                        out = self._inference(img_rgb, d['binary_mask'], seed=42,
                                              pointmap=ptmap_t_use, ss_correction_fn=_ss_fn)
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
