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


def load_config(path: str):
    with open(path) as f:
        raw = yaml.safe_load(f)
    return _dict_to_ns(raw), raw


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
        '--frames-per-scene', type=int, default=None,
        help='Subsample to this many keyframes PER SCENE, evenly spaced within '
             '[--frame-start, --frame-end) (gaps between picks, not a contiguous run) -- '
             'for scene diversity (many scenes, few frames each) instead of many correlated '
             'frames from one scene. With --annotated-only (ECP), spaces within each scene\'s '
             'own annotated keyframes instead.',
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
        '--pointmap-mode', default=None,
        help='Override sam3d_objects.pointmap_mode from the config (number 1-11 or name), e.g. for mode sweeps.',
    )
    parser.add_argument(
        '--cformer-drop-on-failure', default=None, choices=['true', 'false'],
        help="Override sam3d_objects.cformer_drop_on_failure (mode 4/10/11 only): whether in-mask points are "
             "kept as-is when HDBSCAN finds no dominant cluster (default false) or dropped instead ('true').",
    )
    parser.add_argument(
        '--force-drop', default=None,
        help="Token-drop probe (analysis only): comma-separated SAM3D conditioning input(s) to zero after "
             "loading, e.g. 'pointmap,rgb_pointmap'. Valid names: pointmap/rgb_pointmap (crop/full pointmap), "
             "image/rgb_image (crop/full RGB), mask/rgb_image_mask (crop/full mask). Empty/omitted = normal "
             "behaviour. See SAM3DObjectsModel.set_force_drop.",
    )
    parser.add_argument(
        '--checkpoint-dir', type=Path, default=None,
        help='Use this folder for checkpoints instead of <output-dir>/checkpoints. Runs of different pointmap modes '
             'can share it: SAM3, TerraSeg, PseudoLabeler and Body results are reused, SAM3D Objects results are '
             'kept per mode.',
    )
    parser.add_argument(
        '--prepare-only', action='store_true',
        help='Compute only the mode-independent stages (SAM3, TerraSeg, PseudoLabeler, SAM3D Body) into the '
             'checkpoints and stop: no SAM3D Objects, no submission files. Single-camera runs only.',
    )
    parser.add_argument(
        '--no-checkpoint', action='store_true',
        help='Disable checkpointing (no intermediate files written to disk).',
    )
    return parser.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg, cfg_raw = load_config(args.config)
    if args.pointmap_mode is not None:                      # validated later by the pipeline (helpful error if unknown)
        _pm = int(args.pointmap_mode) if str(args.pointmap_mode).isdigit() else args.pointmap_mode
        cfg.sam3d_objects.pointmap_mode = _pm
        cfg_raw['sam3d_objects']['pointmap_mode'] = _pm
    if args.cformer_drop_on_failure is not None:
        _cdof = args.cformer_drop_on_failure == 'true'
        cfg.sam3d_objects.cformer_drop_on_failure = _cdof
        cfg_raw['sam3d_objects']['cformer_drop_on_failure'] = _cdof
    if args.force_drop is not None:
        _fd = [s.strip() for s in args.force_drop.split(',') if s.strip()] or None
        cfg.sam3d_objects.force_drop_modalities = _fd
        cfg_raw['sam3d_objects']['force_drop_modalities'] = _fd

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
    if not args.run_name:
        # default names carry the pointmap mode so runs with different modes don't overwrite each other
        from src.autolabeling.models.sam3d_objects import POINTMAP_MODES, resolve_pointmap_mode
        _mode = resolve_pointmap_mode(getattr(getattr(cfg, 'sam3d_objects', None), 'pointmap_mode', 2))
        _mode_no = {v: k for k, v in POINTMAP_MODES.items()}[_mode]
        run_name += f'_mode{_mode_no}'
    checkpoint_dir  = None if args.no_checkpoint else (args.checkpoint_dir or output_dir / 'checkpoints')
    output_dir.mkdir(parents=True, exist_ok=True)

    if checkpoint_dir:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print(f'Checkpointing enabled: {checkpoint_dir}')

    # ── Data loading ──────────────────────────────────────────────────────────
    from nuscenes.nuscenes import NuScenes

    print(f'Loading {cfg.dataset.upper()} dataset from {cfg.data_root} ({cfg.version})...')
    nusc = NuScenes(version=cfg.version, dataroot=cfg.data_root, verbose=False)

    # ── Determine single-camera vs. multi-camera mode ─────────────────────────
    # Must be computed before dataset-specific branching so the annotated-only
    # ECP path can also route to multi-camera frame collection.
    _cameras_list = getattr(cfg, 'cameras', None)   # plural field in config
    _multi_cam    = bool(_cameras_list and len(_cameras_list) > 1)

    if cfg.dataset == 'ecp':
        from src.autolabeling.data.ecp_loader import (
            collect_frames, collect_annotated_frames,
            collect_annotated_frames_multi_cam, collect_frames_multi_cam,
            get_annotated_scene_names,
        )
        if args.annotated_only:
            if _multi_cam:
                frames_per_cam = collect_annotated_frames_multi_cam(nusc, _cameras_list, args.frames_per_scene)
                frames = frames_per_cam[_cameras_list[0]]
            else:
                _camera = _cameras_list[0] if _cameras_list else cfg.camera
                frames = collect_annotated_frames(nusc, _camera, args.frames_per_scene)
        else:
            if args.scenes is None:
                print('ECP: restricting to annotated scenes only.')
                scene_names = get_annotated_scene_names(nusc)
            else:
                scene_names = args.scenes
    else:
        from src.autolabeling.data.nuscenes_loader import (
            collect_frames, collect_frames_multi_cam,
        )
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
        print(f'Collecting frames (frame_start={args.frame_start}, frame_end={args.frame_end}, '
              f'frames_per_scene={args.frames_per_scene})...')
        if _multi_cam:
            frames_per_cam = collect_frames_multi_cam(
                nusc, scene_names, _cameras_list, args.frame_start, args.frame_end, args.frames_per_scene
            )
            frames = frames_per_cam[_cameras_list[0]]  # reference list for count / names
        else:
            _camera = (_cameras_list[0] if _cameras_list else cfg.camera)
            frames = collect_frames(nusc, scene_names, _camera, args.frame_start, args.frame_end, args.frames_per_scene)

    _n_scenes = len(set(f.scene_name for f in frames))
    if _multi_cam:
        _n_cams = len(_cameras_list)
        print(f'Total: {len(frames)} keyframe(s) × {_n_cams} cameras = '
              f'{len(frames) * _n_cams} frame–camera pairs '
              f'across {_n_scenes} scene(s).\n')
    else:
        print(f'Total: {len(frames)} keyframe(s) across {_n_scenes} scene(s).\n')

    if not frames:
        print('No frames to process. Exiting.')
        sys.exit(0)

    # ── Pipeline ──────────────────────────────────────────────────────────────
    from src.autolabeling.writers.submission import (
        build_submission, remap_submission, write_submission,
    )

    if _multi_cam and args.prepare_only:
        sys.exit('--prepare-only supports single-camera runs only')
    if _multi_cam:
        from src.autolabeling.pipeline import run_multi_camera_pipeline
        from src.autolabeling.writers.submission import build_submission_multi_cam
        body_results_all, obj_results_all = run_multi_camera_pipeline(
            cfg, frames_per_cam, checkpoint_dir=checkpoint_dir, nusc=nusc,
        )
        submission_8class = build_submission_multi_cam(
            frames_per_cam=frames_per_cam,
            body_results_all=body_results_all,
            obj_results_all=obj_results_all,
            split=split,
            mapping_name='8class',
            config=cfg_raw,
        )
    else:
        from src.autolabeling.pipeline import run_pipeline
        body_results, obj_results = run_pipeline(
            cfg, frames, checkpoint_dir=checkpoint_dir, nusc=nusc, prepare_only=args.prepare_only,
        )
        if args.prepare_only:
            print(f'\nPrepared {len(frames)} frame(s) in {checkpoint_dir}. No submission written (prepare-only).')
            return
        submission_8class = build_submission(
            frames=frames,
            body_results=body_results,
            obj_results=obj_results,
            split=split,
            mapping_name='8class',
            config=cfg_raw,
        )

    # (submission_8class kept for remapping below — same structure either way)

    print(f'\nWriting submissions to {output_dir}/')
    for mapping in ['8class', '3class', '1class']:
        sub = (submission_8class if mapping == '8class'
               else remap_submission(submission_8class, mapping))
        write_submission(sub, output_dir / f'{run_name}_{mapping}.json')

    print(f'\nDone. Run evaluation with e.g.:')
    print(f'  python evaluation/evaluation_{cfg.dataset}.py '
          f'--submission {output_dir / f"{run_name}_8class.json"}')


if __name__ == '__main__':
    main()
