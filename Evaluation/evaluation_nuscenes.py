#!/usr/bin/env python3
"""
Evaluate pseudo labels on the nuScenes dataset using the official nuScenes detection metrics.

Outputs (written to --out_dir):
  metrics_summary.json   mAP, NDS, mATE, mASE, mAOE, mAVE (printed to console too)
  metrics_details.json   per-class / per-threshold breakdown
  plots/                 PR and TP curves

Submission JSON format:
{
    "split": "train",          # "train" or "val"
    "mapping_name": "8class",  # one of: "1class", "3class", "8class"
    "meta": {"use_camera": false, "use_lidar": true, "use_radar": false,
             "use_map": false, "use_external": false},
    "results": {
        "<sample_token>": [
            {
                "sample_token":    "<sample_token>",
                "translation":     [x, y, z],
                "size":            [width, length, height],
                "rotation":        [w, x, y, z],
                "velocity":        [vx, vy],
                "detection_name":  "car",
                "detection_score": 0.5,
                "attribute_name":  ""
            }
        ]
    }
}

Usage:
  python evaluation_nuscenes.py --submission path/to/labels.json
  python evaluation_nuscenes.py --submission path/to/labels.json --split val
  python evaluation_nuscenes.py --submission path/to/labels.json --scenes scene-0061
  python evaluation_nuscenes.py --submission path/to/labels.json --scenes scene-0061 scene-0553
  python evaluation_nuscenes.py --submission path/to/labels.json --out_dir my_results/

Scene-wise evaluation note:
  The nuScenes evaluator requires pred_tokens == gt_tokens exactly.
  When --scenes is used, the split is patched to include only the requested
  scenes, so the submission must also cover exactly those scenes' sample tokens
  (use make_gt_submission_nuscenes.py --scenes to generate a matching test submission).
"""

import argparse
import json
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import nuscenes
import nuscenes.eval.common.loaders
import nuscenes.eval.detection.constants as _det_constants
from nuscenes.eval.detection.evaluate import DetectionEval

from class_remapping import build_detection_config, make_category_fn, VALID_MAPPINGS
from eval_patches import (
    augment_metrics_summary,
    eval_patches,
    make_patched_visualize_sample,
    make_scene_split_fn,
)

# ── Dataset root ───────────────────────────────────────────────────────────────

NUSCENES_ROOT    = Path('/media/lleba/nuScenes_mini')
NUSCENES_VERSION = 'v1.0-mini'

# ── nuScenes instance (cached) ─────────────────────────────────────────────────

@lru_cache(maxsize=None)
def _get_nusc() -> nuscenes.NuScenes:
    return nuscenes.NuScenes(NUSCENES_VERSION, str(NUSCENES_ROOT), verbose=False)

# ── Core evaluation function ───────────────────────────────────────────────────

def evaluate(
    submission_path: Path,
    out_dir: Path,
    split: Optional[str] = None,
    scenes: Optional[List[str]] = None,
    plot_examples: int = 0,
) -> None:
    """
    Run the nuScenes detection evaluation.

    Args:
        submission_path: Path to the submission JSON.
        out_dir:         Directory where all result files are written.
        split:           Override the split stored in the JSON ("train" or "val").
                         If None, the split field from the JSON is used.
        scenes:          Restrict evaluation to specific scene names.
                         The submission must contain exactly the sample tokens
                         belonging to these scenes (use make_gt_submission_nuscenes.py
                         with the same --scenes flag to prepare a matching file).
        plot_examples:   Number of random BEV example plots to render (0 = none).
    """
    with open(submission_path) as f:
        sub = json.load(f)

    mapping_name = sub['mapping_name']
    raw_split    = split if split is not None else sub['split']
    eval_set     = f'mini_{raw_split}' if NUSCENES_VERSION == 'v1.0-mini' else raw_split

    cfg  = build_detection_config(mapping_name)
    nusc = _get_nusc()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Patch (1): category_to_detection_name — maps nuScenes GT category names
    #            to the final class names for the chosen mapping scheme.
    # Patch (2): create_splits_scenes — restrict to specific scenes if requested.
    # Patch (3): DETECTION_NAMES / PRETTY_DETECTION_NAMES / DETECTION_COLORS —
    #            mutated in-place (same approach as VESPA) so that non-standard
    #            class names like "vehicle" pass the devkit's assertion in
    #            DetectionBox.deserialize and are recognised during rendering.
    # All patches span DetectionEval.__init__ AND evaluator.main() since render()
    # also looks up DETECTION_COLORS by class name.
    orig_cat_fn      = nuscenes.eval.common.loaders.category_to_detection_name
    orig_split_fn    = nuscenes.eval.common.loaders.create_splits_scenes
    orig_det_names   = list(_det_constants.DETECTION_NAMES)
    orig_pretty      = dict(_det_constants.PRETTY_DETECTION_NAMES)
    orig_colors      = dict(_det_constants.DETECTION_COLORS)

    nuscenes.eval.common.loaders.category_to_detection_name = make_category_fn(mapping_name)
    if scenes:
        nuscenes.eval.common.loaders.create_splits_scenes = make_scene_split_fn(eval_set, scenes)
    _det_constants.DETECTION_NAMES.clear()
    _det_constants.DETECTION_NAMES.extend(cfg.class_names)
    _det_constants.PRETTY_DETECTION_NAMES.clear()
    _det_constants.PRETTY_DETECTION_NAMES.update({k: k for k in cfg.class_names})
    _det_constants.DETECTION_COLORS.clear()
    _det_constants.DETECTION_COLORS.update({k: f"C{i}" for i, k in enumerate(cfg.class_names)})

    try:
        evaluator = DetectionEval(
            nusc,
            config=cfg,
            result_path=str(submission_path),
            eval_set=eval_set,
            output_dir=str(out_dir),
            verbose=True,
        )
        nuscenes.eval.common.loaders.category_to_detection_name = orig_cat_fn
        nuscenes.eval.common.loaders.create_splits_scenes       = orig_split_fn

        vis_fn = make_patched_visualize_sample()
        with eval_patches(cfg, visualize_sample_fn=vis_fn):
            evaluator.main(plot_examples=plot_examples, render_curves=True)
        augment_metrics_summary(out_dir, cfg)
    finally:
        nuscenes.eval.common.loaders.category_to_detection_name = orig_cat_fn
        nuscenes.eval.common.loaders.create_splits_scenes       = orig_split_fn
        _det_constants.DETECTION_NAMES.clear()
        _det_constants.DETECTION_NAMES.extend(orig_det_names)
        _det_constants.PRETTY_DETECTION_NAMES.clear()
        _det_constants.PRETTY_DETECTION_NAMES.update(orig_pretty)
        _det_constants.DETECTION_COLORS.clear()
        _det_constants.DETECTION_COLORS.update(orig_colors)

# ── CLI ────────────────────────────────────────────────────────────────────────

def _validate_submission(path: Path) -> dict:
    with open(path) as f:
        sub = json.load(f)
    missing = {'split', 'mapping_name', 'results'} - sub.keys()
    if missing:
        raise ValueError(f'Submission JSON missing required keys: {missing}')
    if sub['mapping_name'] not in VALID_MAPPINGS:
        raise ValueError(
            f"Unknown mapping_name '{sub['mapping_name']}'. "
            f"Expected one of: {VALID_MAPPINGS}"
        )
    if sub['split'] not in ('train', 'val'):
        raise ValueError(
            f"Unknown split '{sub['split']}' in submission. Expected 'train' or 'val'."
        )
    return sub


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Evaluate nuScenes pseudo labels using nuScenes detection metrics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--submission', type=Path, required=True,
        help='Path to the submission JSON',
    )
    parser.add_argument(
        '--split', choices=['train', 'val'], default=None,
        help='Override the split stored in the submission JSON',
    )
    parser.add_argument(
        '--scenes', nargs='+', default=None,
        help='Restrict to specific scene names (e.g. --scenes scene-0061 scene-0553)',
    )
    parser.add_argument(
        '--out_dir', type=Path, default=None,
        help='Output directory (default: eval_results/<submission_stem>/nuscenes/)',
    )
    parser.add_argument(
        '--plot_examples', type=int, default=0,
        help='Number of random BEV example plots to render (default: 0)',
    )
    args = parser.parse_args()

    submission_path = args.submission.resolve()
    if not submission_path.exists():
        raise FileNotFoundError(f'Submission not found: {submission_path}')

    sub       = _validate_submission(submission_path)
    raw_split = args.split if args.split is not None else sub['split']
    eval_set  = f'mini_{raw_split}' if NUSCENES_VERSION == 'v1.0-mini' else raw_split

    out_dir = (
        args.out_dir.resolve() if args.out_dir
        else Path(__file__).parent / 'eval_results' / submission_path.stem / 'nuscenes'
    )

    scene_info = f'  ({", ".join(args.scenes)})' if args.scenes else '  (all scenes in split)'

    print(f'Submission  : {submission_path.name}')
    print(f'Version     : {NUSCENES_VERSION}')
    print(f'Split       : {raw_split}  (eval_set: {eval_set})')
    print(f'Scenes      :{scene_info}')
    print(f'Mapping     : {sub["mapping_name"]}')
    print(f'Samples     : {len(sub["results"])}')
    print(f'Output dir  : {out_dir}\n')

    evaluate(submission_path, out_dir, split=args.split, scenes=args.scenes,
             plot_examples=args.plot_examples)


if __name__ == '__main__':
    main()
