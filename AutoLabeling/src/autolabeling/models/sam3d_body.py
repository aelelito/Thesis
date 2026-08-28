"""
SAM3D Body inference wrapper.

Detections are seeded from SAM3 pedestrian masks (primary) supplemented by
ViTDet boxes that don't overlap with any SAM3 mask (gap-filling). This
matches the logic from run_sam3d_bodies.py.

SAM3 loading leaves torch.autocast globally enabled (BF16), which causes
`addmm_sparse_cuda` errors in SAM3D Body. Autocast is explicitly disabled
before loading and kept disabled during inference.
"""
import gc
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from ..utils.logging_utils import suppress_output


class SAM3DBodyModel:
    def __init__(
        self,
        repo_path: str,
        checkpoint_path: str,
        mhr_path: str,
        bbox_thresh: float = 0.8,
        iou_merge_thresh: float = 0.4,
        device: Optional[str] = None,
    ):
        self.repo_path        = str(repo_path)
        self.checkpoint_path  = str(checkpoint_path)
        self.mhr_path         = str(mhr_path)
        self.bbox_thresh      = bbox_thresh
        self.iou_merge_thresh = iou_merge_thresh
        self.device           = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self._estimator       = None
        self._faces           = None

    def load(self) -> None:
        # SAM3 loading leaves autocast(BF16) globally enabled — disable it here
        # so model init and all inference run in the correct dtype.
        if torch.is_autocast_enabled():
            torch.autocast('cuda', enabled=False).__enter__()
            print('  SAM3D Body: disabled inherited BF16 autocast.')

        if self.repo_path not in sys.path:
            sys.path.insert(0, self.repo_path)

        from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
        from tools.build_detector import HumanDetector

        with suppress_output():
            model, cfg = load_sam_3d_body(
                self.checkpoint_path, device=self.device, mhr_path=self.mhr_path
            )
            self._estimator = SAM3DBodyEstimator(
                sam_3d_body_model=model,
                model_cfg=cfg,
                human_detector=HumanDetector(name='vitdet', device=self.device),
                human_segmentor=None,
                fov_estimator=None,
            )
        print('  SAM3D Body loaded.')

    def unload(self) -> None:
        del self._estimator
        self._estimator = None
        self._faces     = None
        gc.collect()
        torch.cuda.empty_cache()
        print('  SAM3D Body unloaded.')

    @staticmethod
    def _merge_boxes(sam3_boxes: np.ndarray, vitdet_boxes: np.ndarray, iou_thresh: float):
        """SAM3 boxes take priority; ViTDet fills gaps (IoU < iou_thresh with all SAM3 boxes)."""
        if len(vitdet_boxes) == 0:
            return sam3_boxes
        if len(sam3_boxes) == 0:
            return vitdet_boxes
        ix1   = np.maximum(vitdet_boxes[:, 0:1], sam3_boxes[:, 0])
        iy1   = np.maximum(vitdet_boxes[:, 1:2], sam3_boxes[:, 1])
        ix2   = np.minimum(vitdet_boxes[:, 2:3], sam3_boxes[:, 2])
        iy2   = np.minimum(vitdet_boxes[:, 3:4], sam3_boxes[:, 3])
        inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
        area_v = ((vitdet_boxes[:, 2] - vitdet_boxes[:, 0]) *
                  (vitdet_boxes[:, 3] - vitdet_boxes[:, 1]))[:, None]
        area_s = ((sam3_boxes[:, 2] - sam3_boxes[:, 0]) *
                  (sam3_boxes[:, 3] - sam3_boxes[:, 1]))[None, :]
        iou   = inter / (area_v + area_s - inter + 1e-8)
        extra = vitdet_boxes[iou.max(axis=1) < iou_thresh]
        if len(extra) == 0:
            return sam3_boxes
        return np.vstack([sam3_boxes, extra])

    def run_frame(self, frame, ped_dets: list) -> list:
        """Run SAM3D Body on a single frame. Returns list of pedestrian result dicts."""
        img_rgb, img_bgr = frame.load_images()
        K_t = torch.tensor(frame.K, dtype=torch.float32).unsqueeze(0)

        # SAM3 masks → pixel xyxy boxes
        # sam3_mask_indices[j] = index into ped_dets for the j-th SAM3 box
        # (some ped_dets entries may be skipped when the mask is empty)
        sam3_boxes = []
        sam3_mask_indices = []
        for det_idx, d in enumerate(ped_dets):
            ys, xs = np.where(d['binary_mask'])
            if len(xs) == 0:
                continue
            sam3_boxes.append([xs.min(), ys.min(), xs.max(), ys.max()])
            sam3_mask_indices.append(det_idx)
        sam3_boxes = (np.array(sam3_boxes, dtype=np.float32)
                      if sam3_boxes else np.empty((0, 4), dtype=np.float32))

        # ViTDet supplementary detections
        vitdet_boxes = self._estimator.detector.run_human_detection(
            img_bgr, bbox_thr=self.bbox_thresh, nms_thr=0.3, default_to_full_image=False
        )
        vitdet_boxes = np.asarray(vitdet_boxes, dtype=np.float32).reshape(-1, 4)

        merged = self._merge_boxes(sam3_boxes, vitdet_boxes, self.iou_merge_thresh)

        if len(merged) == 0:
            return []

        with torch.autocast('cuda', enabled=False), suppress_output():
            outputs = self._estimator.process_one_image(
                frame.img_path, bboxes=merged,  # img_path used internally by process_one_image
                bbox_thr=self.bbox_thresh, use_mask=False, cam_int=K_t,
            )

        faces = self._estimator.faces
        if hasattr(faces, 'cpu'):
            faces = faces.detach().cpu().numpy()
        faces = faces.astype(np.int32)

        results = []
        for j, o in enumerate(outputs):
            verts  = np.asarray(o['pred_vertices'],     dtype=np.float32)
            joints = np.asarray(o['pred_keypoints_3d'], dtype=np.float32)
            cam_t  = np.asarray(o['pred_cam_t'],        dtype=np.float32)
            # sam3_mask_idx: index into ped_dets for SAM3-sourced boxes;
            # None for ViTDet-only detections (appended after sam3_boxes).
            sam3_mask_idx = sam3_mask_indices[j] if j < len(sam3_mask_indices) else None
            binary_mask   = (ped_dets[sam3_mask_idx]['binary_mask']
                             if sam3_mask_idx is not None else None)
            results.append({
                'vertices':      verts + cam_t[None, :],
                'faces':         faces,
                'joints_3d':     joints,
                'cam_t':         cam_t,
                'bbox':          merged[j],
                'score':         1.0,
                'sam3_mask_idx': sam3_mask_idx,
                'binary_mask':   binary_mask,
            })
        return results

    def run_batch(self, frames, sam3_results: Dict[int, Dict[str, list]]) -> Dict[int, list]:
        """Run all frames. Returns {frame_idx: [pedestrian result dicts]}."""
        assert self._estimator is not None, 'Call load() before run_batch()'
        results = {}
        for i, frame in enumerate(frames):
            results[i] = self.run_frame(frame, sam3_results[i].get('pedestrian', []))
        return results
