"""
SAM3 text-prompted segmentation wrapper.

Runs SAM3 on a batch of frames, one text prompt at a time. Each frame is
written to a temporary directory as a single-frame 'video' (SAM3's input
format), then the session is closed after propagation.

Output per frame: dict mapping prompt → list of detection dicts:
    {
        'binary_mask': (H, W) bool,
        'score':       float,
        'prompt':      str,
    }
"""
import gc
import os
import shutil
import tempfile
from typing import Dict, Optional

import numpy as np
import torch
from PIL import Image

from ..utils.logging_utils import suppress_output


def dedup_cross_class(
    results: Dict[str, list],
    priority: Dict[str, int],
    iou_thresh: float,
) -> Dict[str, list]:
    """
    Remove lower-priority detections that overlap with higher-priority ones.

    For each pair of detections from *different* classes whose mask IoU exceeds
    `iou_thresh`, the detection belonging to the lower-priority class is dropped.
    Priority values come from the config (cross_class_dedup.priority) — all
    classes should have unique values so every pair has a definite winner.
    """
    flat = [(p, d) for p, dets in results.items() for d in dets]
    keep = [True] * len(flat)

    for i in range(len(flat)):
        for j in range(i + 1, len(flat)):
            if not keep[i] or not keep[j]:
                continue
            p_i, d_i = flat[i]
            p_j, d_j = flat[j]
            if p_i == p_j:
                continue
            a, b  = d_i['binary_mask'], d_j['binary_mask']
            inter = np.logical_and(a, b).sum()
            if inter == 0:
                continue
            iou = inter / np.logical_or(a, b).sum()
            if iou < iou_thresh:
                continue
            if priority.get(p_i, 0) >= priority.get(p_j, 0):
                keep[j] = False
            else:
                keep[i] = False

    out = {p: [] for p in results}
    for (p, d), k in zip(flat, keep):
        if k:
            out[p].append(d)

    removed = sum(1 for k in keep if not k)
    if removed:
        print(f'    cross-class dedup: removed {removed} mask(s) (IoU≥{iou_thresh})')
    return out


class SAM3Segmentor:
    def __init__(
        self,
        checkpoint_path: str,
        prompts: Dict[str, str],        # {text_prompt: pipeline_type}
        score_thresh: float = 0.5,
        min_mask_px: int = 100,
        device: Optional[str] = None,
        cross_class_iou_thresh: float = 0.5,
        cross_class_priority: Optional[Dict[str, int]] = None,
    ):
        self.checkpoint_path        = str(checkpoint_path)
        self.prompts                = prompts
        self.score_thresh           = score_thresh
        self.min_mask_px            = min_mask_px
        self.device                 = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.cross_class_iou_thresh = cross_class_iou_thresh
        self.cross_class_priority   = cross_class_priority or {}
        self._predictor             = None

    def load(self) -> None:
        from sam3.model_builder import build_sam3_video_predictor
        gpus = range(torch.cuda.device_count()) if self.device == 'cuda' else []
        with suppress_output():
            self._predictor = build_sam3_video_predictor(
                gpus_to_use=gpus, checkpoint_path=self.checkpoint_path
            )
        print('  SAM3 loaded.')

    def unload(self) -> None:
        del self._predictor
        self._predictor = None
        gc.collect()
        torch.cuda.empty_cache()
        print('  SAM3 unloaded.')

    def _run_single_frame(self, img_rgb: np.ndarray, text_prompt: str) -> list:
        """Run SAM3 for one prompt on one frame. Returns list of detection dicts."""
        tmp_dir = tempfile.mkdtemp(prefix='sam3_')
        try:
            Image.fromarray(img_rgb).save(os.path.join(tmp_dir, '00000.jpg'))
            with suppress_output():
                resp = self._predictor.handle_request(
                    dict(type='start_session', resource_path=tmp_dir)
                )
                session_id = resp['session_id']
                self._predictor.handle_request(dict(
                    type='add_prompt', session_id=session_id, frame_index=0, text=text_prompt
                ))
                frame_outputs = {}
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    for resp in self._predictor.handle_stream_request(dict(
                        type='propagate_in_video', session_id=session_id
                    )):
                        frame_outputs[resp['frame_index']] = resp['outputs']
                self._predictor.handle_request(dict(type='close_session', session_id=session_id))
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        out     = frame_outputs.get(0, {})
        obj_ids = out.get('out_obj_ids', [])
        probs   = np.asarray(out.get('out_probs', []), dtype=np.float32)
        masks   = out.get('out_binary_masks', [])

        detections = []
        for i, (_, prob) in enumerate(zip(obj_ids, probs)):
            if prob < self.score_thresh:
                continue
            bm = np.asarray(masks[i], dtype=bool)
            if bm.sum() < self.min_mask_px:
                continue
            detections.append({
                'binary_mask': bm,
                'score':       float(prob),
                'prompt':      text_prompt,
            })
        return detections

    def run_frame(self, frame) -> Dict[str, list]:
        """Run all configured prompts on a single frame. Returns {prompt: [dets]}."""
        assert self._predictor is not None, 'Call load() before run_frame()'
        img_rgb, _ = frame.load_images()
        results = {prompt: self._run_single_frame(img_rgb, prompt) for prompt in self.prompts}
        if self.cross_class_priority:
            results = dedup_cross_class(results, self.cross_class_priority, self.cross_class_iou_thresh)
        return results

    def run_batch(self, frames) -> Dict[int, Dict[str, list]]:
        """Run all frames. Returns {frame_idx: {prompt: [detection_dicts]}}."""
        return {i: self.run_frame(frame) for i, frame in enumerate(frames)}
