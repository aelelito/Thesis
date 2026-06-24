#!/usr/bin/env python3
"""
Auto-labeling pipeline — Stage 1 (image-only, single-frame).

Usage
-----
Run on all scenes, all frames:
    python run_pipeline.py --config configs/nuscenes.yaml

Run on specific scenes:
    python run_pipeline.py --config configs/ecp.yaml \
        --scenes scene-euro-citystrasbourg-... scene-euro-...

Run on a frame range (per-scene keyframe index, 0-based, inclusive start / exclusive end):
    python run_pipeline.py --config configs/nuscenes.yaml \
        --scenes scene-0001 --frame-start 0 --frame-end 50

Output
------
A single submission JSON at <output_dir>/<run_name>.json, compatible with
evaluation_nuscenes.py and evaluation_ecp.py.
"""
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import yaml


# ── Config loading ─────────────────────────────────────────────────────────────

def _dict_to_ns(d):
    """Recursively convert a dict to SimpleNamespace for attribute access."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _dict_to_ns(v) for k, v in d.items()})
    return d


def load_config(path: str) -> SimpleNamespace:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return _dict_to_ns(raw)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='Auto-labeling pipeline — Stage 1',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--config', required=True,
        help='Path to a YAML config file (e.g. configs/nuscenes.yaml)',
    )
    parser.add_argument(
        '--scenes', nargs='*', default=None,
        help='Scene names to process. Default: all scenes in the dataset.',
    )
    parser.add_argument(
        '--frame-start', type=int, default=0,
        help='First keyframe index to include (0-based, per-scene). Default: 0.',
    )
    parser.add_argument(
        '--frame-end', type=int, default=None,
        help='Last keyframe index to exclude (per-scene). Default: end of scene.',
    )
    parser.add_argument(
        '--output-dir', type=Path, default=None,
        help='Override output directory from config.',
    )
    parser.add_argument(
        '--run-name', type=str, default=None,
        help='Submission filename stem. Default: autolabel_<scene> / autolabel_<dataset>_Nscenes / autolabel_<dataset>_all.',
    )
    parser.add_argument(
        '--split', type=str, default=None,
        help='Override the split from the config (e.g. train, val, mini_train, mini_val). '
             'For nuScenes: restricts scenes to those in the split. '
             'Also sets the split field in the output submission JSON.',
    )
    parser.add_argument(
        '--annotated-only', action='store_true',
        help='(ECP only) Process only the annotated keyframes instead of full frame ranges.',
    )
    parser.add_argument(
        '--no-checkpoint', action='store_true',
        help='Disable checkpointing (no intermediate files written to disk).',
    )
    return parser.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg  = load_config(args.config)

    output_dir   = Path(args.output_dir or cfg.output_dir)
    split        = args.split or cfg.split  # CLI overrides config; always 'train' or 'val'
    dataset_name = getattr(cfg, 'name', cfg.dataset)  # e.g. 'nuscenes_mini' or 'ecp'

    if args.run_name:
        run_name = args.run_name
    elif args.annotated_only:
        run_name = f'autolabel_{dataset_name}_annotated'
    elif args.scenes and len(args.scenes) == 1:
        run_name = f'autolabel_{args.scenes[0]}'
    elif args.scenes:
        run_name = f'autolabel_{dataset_name}_{len(args.scenes)}scenes'
    else:
        run_name = f'autolabel_{dataset_name}_{split}'
    checkpoint_dir  = None if args.no_checkpoint else output_dir / 'checkpoints'
    output_dir.mkdir(parents=True, exist_ok=True)

    if checkpoint_dir:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print(f'Checkpointing enabled: {checkpoint_dir}')

    # ── Data loading ──────────────────────────────────────────────────────────
    from nuscenes.nuscenes import NuScenes

    print(f'Loading {cfg.dataset.upper()} dataset from {cfg.data_root} ({cfg.version})...')
    nusc = NuScenes(version=cfg.version, dataroot=cfg.data_root, verbose=False)

    if cfg.dataset == 'ecp':
        from src.autolabeling.data.ecp_loader import (
            collect_frames, collect_annotated_frames, get_annotated_scene_names,
        )
        if args.annotated_only:
            frames = collect_annotated_frames(nusc, cfg.camera)
        else:
            if args.scenes is None:
                print('ECP: restricting to annotated scenes only.')
                scene_names = get_annotated_scene_names(nusc)
            else:
                scene_names = args.scenes
    else:
        from src.autolabeling.data.nuscenes_loader import collect_frames
        if args.scenes is None:
            from nuscenes.utils.splits import create_splits_scenes
            all_splits  = create_splits_scenes()
            # v1.0-mini uses 'mini_train'/'mini_val' keys; full dataset uses 'train'/'val'
            split_key   = f'mini_{split}' if f'mini_{split}' in all_splits and split in ('train', 'val') and cfg.version == 'v1.0-mini' else split
            scene_names = all_splits[split_key]
            print(f'nuScenes: {split} split ({split_key}) → {len(scene_names)} scene(s).')
        else:
            scene_names = args.scenes

    if not (cfg.dataset == 'ecp' and args.annotated_only):
        print(f'Collecting frames (frame_start={args.frame_start}, frame_end={args.frame_end})...')
        frames = collect_frames(
            nusc, scene_names, cfg.camera, args.frame_start, args.frame_end
        )
    print(f'Total: {len(frames)} keyframe(s) across '
          f'{len(set(f.scene_name for f in frames))} scene(s).\n')

    if not frames:
        print('No frames to process. Exiting.')
        sys.exit(0)

    # ── Pipeline ──────────────────────────────────────────────────────────────
    from src.autolabeling.pipeline import run_pipeline
    body_results, obj_results = run_pipeline(cfg, frames, checkpoint_dir=checkpoint_dir)

    # ── Write submission ───────────────────────────────────────────────────────
    # Always build 8class first, then derive 3class and 1class by remapping
    # detection_name values using configs/class_mapping/<mapping>.yaml.
    from src.autolabeling.writers.submission import (
        build_submission, remap_submission, write_submission,
    )

    submission_8class = build_submission(
        frames=frames,
        body_results=body_results,
        obj_results=obj_results,
        split=split,
        mapping_name='8class',
    )

    print(f'\nWriting submissions to {output_dir}/')
    for mapping in ['8class', '3class', '1class']:
        sub = (submission_8class if mapping == '8class'
               else remap_submission(submission_8class, mapping))
        write_submission(sub, output_dir / f'{run_name}_{mapping}.json')

    print(f'\nDone. Run evaluation with e.g.:')
    print(f'  python Evaluation/evaluation_{cfg.dataset}.py '
          f'--submission {output_dir / f"{run_name}_8class.json"}')


if __name__ == '__main__':
    main()
