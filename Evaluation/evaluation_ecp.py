#!/usr/bin/env python3
"""
Evaluate pseudo labels on the ECP dataset using the nuScenes detection metrics.

ECP data is converted to nuScenes format at ECP_ROOT/v1.0-trainval/.
Only manually annotated keyframes are used as GT — unannotated frames are
automatically excluded so they don't pollute precision/recall.  As more frames
are annotated and added to sample_annotation.json, they are picked up without
any changes to this script.

Outputs (written to --out_dir):
  metrics_summary.json   mAP, NDS, mATE, mASE, mAOE, mAVE (printed to console too)
  metrics_details.json   per-class / per-threshold breakdown
  plots/                 PR and TP curves

Submission JSON format (same as nuScenes):
{
    "split": "train",          # always "train" for ECP (no val split yet)
    "mapping_name": "8class",  # one of: "1class", "3class", "8class"
    "meta": {"use_camera": false, "use_lidar": true, "use_radar": false,
             "use_map": false, "use_external": false},
    "results": {
        "<sample_token>": [ { ...box... } ]
    }
}

The submission must contain entries for exactly the annotated sample tokens
(use make_gt_submission_ecp.py to generate a matching test submission).

Usage:
  python evaluation_ecp.py --submission path/to/labels.json
  python evaluation_ecp.py --submission path/to/labels.json --scenes scene-euro-...
  python evaluation_ecp.py --submission path/to/labels.json --out_dir my_results/
  python evaluation_ecp.py --submission path/to/labels.json --plot_examples 3
"""

import argparse
import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nuscenes
import nuscenes.eval.common.loaders
import nuscenes.eval.detection.evaluate
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.detection.evaluate import DetectionEval
from nuscenes.utils.data_classes import LidarPointCloud

from class_remapping import build_detection_config, make_category_fn, VALID_MAPPINGS
from eval_patches import (
    augment_metrics_summary,
    eval_patches,
    make_ecp_lidar_fn,
    make_patched_visualize_sample,
    make_scene_split_fn,
)

# ── Dataset root ───────────────────────────────────────────────────────────────

ECP_ROOT    = Path('/media/lleba/ECP_Nuscenes_01/output2/ecp2nuscenes')
ECP_VERSION = 'v1.0-trainval'

# ── NuScenes instance (cached) ────────────────────────────────────────────────

@lru_cache(maxsize=None)
def _get_nusc() -> nuscenes.NuScenes:
    return nuscenes.NuScenes(ECP_VERSION, str(ECP_ROOT), verbose=False)

# ── Auto-discovery of annotated GT ────────────────────────────────────────────

@lru_cache(maxsize=None)
def _get_annotated_info() -> Tuple[List[str], List[str]]:
    """
    Auto-discover which scenes and sample tokens have ground truth annotations.

    Reads sample_annotation.json → finds annotated sample tokens → maps to
    scene names.  Any future annotations added to the ECP nuScenes conversion
    are automatically picked up here without script changes.

    Returns:
        (sorted list of annotated scene names,
         sorted list of annotated sample tokens)
    """
    nusc = _get_nusc()

    annotated_sample_tokens = {ann['sample_token'] for ann in nusc.sample_annotation}

    annotated_scene_names = set()
    for sample in nusc.sample:
        if sample['token'] in annotated_sample_tokens:
            scene = nusc.get('scene', sample['scene_token'])
            annotated_scene_names.add(scene['name'])

    return sorted(annotated_scene_names), sorted(annotated_sample_tokens)

# ── LIDAR index for ECP ───────────────────────────────────────────────────────
#
# ECP's sample_data.json does not populate the 'channel' field, so
# sample['data'] is always empty.  We build a mapping from sample_token to
# LIDAR sample_data token by scanning filenames (built lazily, cached).

@lru_cache(maxsize=None)
def _build_lidar_index() -> Dict[str, str]:
    """Return {sample_token: lidar_sample_data_token} for all ECP key frames."""
    nusc = _get_nusc()
    return {
        sd['sample_token']: sd['token']
        for sd in nusc.sample_data
        if sd['is_key_frame'] and 'LIDAR_TOP' in sd.get('filename', '')
    }

# ── load_gt patch: annotated samples only ────────────────────────────────────

def _make_annotated_only_load_gt(original_load_gt):
    """
    Wrap load_gt to discard sample tokens that have no GT annotations.

    The nuScenes evaluator adds ALL samples in the evaluated scenes to the GT
    token set (even those with empty annotation lists).  For ECP, where only a
    small fraction of frames are manually annotated, keeping unannotated samples
    would require the submission to cover every frame in those scenes, and any
    pseudo-label predictions on unannotated frames would incorrectly count as
    false positives.

    This wrapper keeps only sample tokens that actually contain at least one GT
    box, so the submission is expected to cover exactly those tokens.
    """
    def _load_gt_annotated_only(nusc_inst, eval_split, box_cls, verbose=False):
        all_boxes = original_load_gt(nusc_inst, eval_split, box_cls, verbose)
        filtered = EvalBoxes()
        for token in all_boxes.sample_tokens:
            if len(all_boxes[token]) > 0:
                filtered.add_boxes(token, all_boxes[token])
        if verbose:
            n_orig = len(all_boxes.sample_tokens)
            n_filt = len(filtered.sample_tokens)
            print(f'ECP: GT restricted to {n_filt} annotated samples '
                  f'(dropped {n_orig - n_filt} unannotated)')
        return filtered
    return _load_gt_annotated_only

# ── load_prediction patch: restrict to annotated tokens ──────────────────────

def _make_annotated_only_load_pred(original_load_pred, annotated_tokens):
    """
    Wrap load_prediction to keep only annotated sample tokens.

    The submission may contain predictions for all frames VESPA ran on (e.g.
    641 frames across the full ECP dataset).  The evaluator requires pred_tokens
    == gt_tokens exactly.  This wrapper:
      - keeps predictions that fall in the annotated set
      - adds empty prediction lists for annotated tokens absent from the submission
        and emits a warning (missing frames = all GT boxes for that frame are FN)
      - drops all predictions for unannotated tokens
    """
    annotated_set = set(annotated_tokens)

    def _load_pred_annotated_only(result_path, max_boxes_per_sample, box_cls, verbose=False):
        all_boxes, meta = original_load_pred(result_path, max_boxes_per_sample, box_cls, verbose)
        filtered = EvalBoxes()
        missing = []
        for token in annotated_set:
            if token in all_boxes.sample_tokens:
                filtered.add_boxes(token, all_boxes[token])
            else:
                filtered.add_boxes(token, [])
                missing.append(token)
        if missing:
            import warnings
            warnings.warn(
                f'ECP: {len(missing)} annotated sample(s) have no predictions in the '
                f'submission — all GT boxes for these frames will count as FN:\n'
                + '\n'.join(f'  {t}' for t in sorted(missing)),
                stacklevel=2,
            )
        if verbose:
            n_pred = len(all_boxes.sample_tokens)
            n_kept = len(annotated_set) - len(missing)
            print(f'ECP: predictions restricted to {len(annotated_set)} annotated samples '
                  f'({n_kept} with predictions, {len(missing)} padded as empty, '
                  f'{n_pred - n_kept} unannotated dropped)')
        return filtered, meta

    return _load_pred_annotated_only


# ── Core evaluation function ───────────────────────────────────────────────────

def evaluate(
    submission_path: Path,
    out_dir: Path,
    split: Optional[str] = None,
    scenes: Optional[List[str]] = None,
    plot_examples: int = 0,
) -> None:
    """
    Run the ECP detection evaluation using nuScenes metrics.

    Args:
        submission_path: Path to the submission JSON.
        out_dir:         Directory where all result files are written.
        split:           Override the split stored in the JSON.
                         If None, the split field from the JSON is used.
        scenes:          Restrict to a subset of the annotated scenes.
                         If None, all annotated scenes are used.
                         Supplied scene names must be in the annotated set.
        plot_examples:   Number of random BEV example plots to render (0 = none).
    """
    with open(submission_path) as f:
        sub = json.load(f)

    mapping_name = sub['mapping_name']
    raw_split    = split if split is not None else sub['split']
    # ECP uses v1.0-trainval → no 'mini_' prefix
    eval_set     = raw_split

    cfg  = build_detection_config(mapping_name)
    nusc = _get_nusc()
    out_dir.mkdir(parents=True, exist_ok=True)

    annotated_scenes, _ = _get_annotated_info()

    # Optionally restrict to a subset of the annotated scenes.
    if scenes:
        unknown = set(scenes) - set(annotated_scenes)
        if unknown:
            raise ValueError(
                f'Scenes not in annotated set: {unknown}\n'
                f'Available annotated scenes: {annotated_scenes}'
            )
        active_scenes = scenes
    else:
        active_scenes = annotated_scenes

    # ── Patches active during DetectionEval.__init__ ──────────────────────────
    #
    # (1) category_to_detection_name — ECP categories are already in short form
    #     ('car', 'pedestrian', …); use mapping_ecp from the YAML.
    # (2) create_splits_scenes — ECP has no split file; return the annotated
    #     scenes as the eval set.  This is always required (not optional).
    # (3) load_gt — filter GT to only annotated sample tokens so unannotated
    #     frames are not included in the evaluation.
    # (4) load_prediction — restrict predictions to the same annotated token set;
    #     annotated tokens absent from the submission get empty predictions (warning
    #     issued); unannotated tokens are dropped to satisfy pred_tokens==gt_tokens.

    _, annotated_tokens = _get_annotated_info()
    _token_to_scene = {
        s['token']: nusc.get('scene', s['scene_token'])['name']
        for s in nusc.sample if s['token'] in set(annotated_tokens)
    }
    active_annotated_tokens = [
        t for t in annotated_tokens
        if _token_to_scene.get(t) in set(active_scenes)
    ]

    orig_cat_fn   = nuscenes.eval.common.loaders.category_to_detection_name
    orig_split_fn = nuscenes.eval.common.loaders.create_splits_scenes
    orig_load_gt  = nuscenes.eval.detection.evaluate.load_gt
    orig_load_pred = nuscenes.eval.detection.evaluate.load_prediction

    nuscenes.eval.common.loaders.category_to_detection_name = make_category_fn(
        mapping_name, dataset='ecp'
    )
    nuscenes.eval.common.loaders.create_splits_scenes = make_scene_split_fn(
        eval_set, list(active_scenes)
    )
    nuscenes.eval.detection.evaluate.load_gt = _make_annotated_only_load_gt(orig_load_gt)
    nuscenes.eval.detection.evaluate.load_prediction = _make_annotated_only_load_pred(
        orig_load_pred, active_annotated_tokens
    )

    try:
        evaluator = DetectionEval(
            nusc,
            config=cfg,
            result_path=str(submission_path),
            eval_set=eval_set,
            output_dir=str(out_dir),
            verbose=True,
        )
    finally:
        nuscenes.eval.common.loaders.category_to_detection_name = orig_cat_fn
        nuscenes.eval.common.loaders.create_splits_scenes       = orig_split_fn
        nuscenes.eval.detection.evaluate.load_gt                = orig_load_gt
        nuscenes.eval.detection.evaluate.load_prediction        = orig_load_pred

    # ECP's sample['data'] is unpopulated → use index-based LIDAR lookup.
    # Single key-frame read (no multi-sweep chain available).
    lidar_index = _build_lidar_index() if plot_examples > 0 else {}
    lidar_fn    = make_ecp_lidar_fn(lidar_index, ECP_ROOT)
    vis_fn      = make_patched_visualize_sample(lidar_fn=lidar_fn)

    with eval_patches(cfg, visualize_sample_fn=vis_fn):
        evaluator.main(plot_examples=plot_examples, render_curves=True)
    augment_metrics_summary(out_dir, cfg)

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
    return sub


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Evaluate ECP pseudo labels using nuScenes detection metrics',
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
        help='Restrict to a subset of annotated scenes (default: all annotated scenes)',
    )
    parser.add_argument(
        '--out_dir', type=Path, default=None,
        help='Output directory (default: eval_results/<submission_stem>/ecp/)',
    )
    parser.add_argument(
        '--plot_examples', type=int, default=0,
        help='Number of random BEV example plots to render (default: 0)',
    )
    args = parser.parse_args()

    submission_path = args.submission.resolve()
    if not submission_path.exists():
        raise FileNotFoundError(f'Submission not found: {submission_path}')

    sub = _validate_submission(submission_path)

    annotated_scenes, annotated_tokens = _get_annotated_info()

    out_dir = (
        args.out_dir.resolve() if args.out_dir
        else Path(__file__).parent / 'eval_results' / submission_path.stem / 'ecp'
    )

    raw_split  = args.split if args.split is not None else sub['split']
    scene_info = f'  ({", ".join(args.scenes)})' if args.scenes else f'  (all {len(annotated_scenes)} annotated scenes)'

    print(f'Submission          : {submission_path.name}')
    print(f'Version             : {ECP_VERSION}')
    print(f'Split               : {raw_split}')
    print(f'Scenes              :{scene_info}')
    print(f'Mapping             : {sub["mapping_name"]}')
    print(f'Submission samples  : {len(sub["results"])}')
    print(f'Annotated GT samples: {len(annotated_tokens)}')
    print(f'Output dir          : {out_dir}\n')

    evaluate(submission_path, out_dir, split=args.split, scenes=args.scenes,
             plot_examples=args.plot_examples)


if __name__ == '__main__':
    main()
