"""
TerraSeg ground / non-ground classifier wrapper.

TerraSeg (PTv3, trained on OmniLiDAR) is trained and evaluated on single,
independent scans, with height/range features defined relative to that scan's own
sensor origin. It must therefore be run **per sweep, in that sweep's own ego
frame, before any multi-sweep aggregation** -- see
`utils/lidar.load_lidar_pts_nonground_aggregated`.
"""
import sys
from typing import Optional

import numpy as np


class TerraSegGroundFilter:
    """
    Wraps TerraSegPredictor for per-sweep ground removal.

    Parameters
    ----------
    dev_root    : Path to <dev_root> (parent of models/)
    variant     : 'S' (default)
    ckpt_path   : explicit checkpoint path; None = auto-resolve from HF cache
    """

    def __init__(self, dev_root, variant: str = 'S', ckpt_path: Optional[str] = None):
        import torch as _torch
        _ts_lib   = str(dev_root / 'models' / 'TerraSeg' / 'terraseg_lib' / 'src')
        _ptv3_lib = str(dev_root / 'models' / 'TerraSeg' / 'ptv3' / 'src')
        for _p in [_ts_lib, _ptv3_lib]:
            if _p not in sys.path:
                sys.path.insert(0, _p)

        if ckpt_path is None:
            try:
                from huggingface_hub import hf_hub_download as _hf
                ckpt_path = _hf(
                    repo_id='TedLentsch/TerraSeg',
                    filename=f'terraseg_{variant.lower()}.pth',
                )
            except Exception as _e:
                raise RuntimeError(
                    f'Cannot locate TerraSeg checkpoint and HF download failed: {_e}'
                )

        from terraseg.predictor import TerraSegPredictor
        self._predictor = TerraSegPredictor(variant=variant, checkpoint_path=ckpt_path)
        self._torch = _torch
        print(f'TerraSeg-{variant} loaded.')

    def segment(self, pts: np.ndarray) -> np.ndarray:
        """
        Return a boolean mask: True = non-ground for each point in pts (N, 3).

        pts must be in the ego frame of the sweep they came from (z ~ up, +x forward).
        """
        if len(pts) == 0:
            return np.zeros(0, dtype=bool)
        with self._torch.no_grad():
            lbl = self._predictor.predict(
                self._torch.tensor(pts.copy(), dtype=self._torch.float32).contiguous()
            ).cpu().numpy()
        return lbl == 1   # 1 = non-ground

    def unload(self):
        del self._predictor
        self._predictor = None
        try:
            import gc
            gc.collect()
            self._torch.cuda.empty_cache()
        except Exception:
            pass
