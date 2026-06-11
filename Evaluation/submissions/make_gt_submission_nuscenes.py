#!/usr/bin/env python3
"""
Create a "perfect" submission JSON from nuScenes ground truth annotations.

Running evaluation on this submission should yield (near) perfect scores,
which confirms the evaluation pipeline is correctly wired up.

Note on "near perfect" vs "perfect":
  The evaluator filters out GT boxes with 0 LiDAR points (these are in GT but
  your pseudo labels won't produce them either). This causes mAP < 1.0 for
  some classes. All geometry errors (mATE, mASE, mAOE) will be exactly 0.0.
  mAAE will be 1.0 because attribute names are left empty — this is expected
  and does not affect 3D detection quality assessment.

Usage:
  python make_gt_submission_nuscenes.py --split train
  python make_gt_submission_nuscenes.py --split val
  python make_gt_submission_nuscenes.py --split train --scenes scene-0061
  python make_gt_submission_nuscenes.py --split train --mapping 1class
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import nuscenes
from nuscenes.utils.splits import create_splits_scenes

sys.path.insert(0, str(Path(__file__).parent.parent))
from class_remapping import VALID_MAPPINGS, nuscenes_category_to_detection_name

NUSCENES_ROOT    = Path('/media/lleba/nuScenes_mini')
NUSCENES_VERSION = 'v1.0-mini'


def make_gt_submission(
    split: str,
    mapping_name: str,
    scenes: Optional[List[str]],
) -> dict:
    """
    Build a submission dict from nuScenes ground truth annotations.

    Args:
        split:        "train" or "val"
        mapping_name: "1class", "3class", or "8class"
        scenes:       Optional list of scene names to restrict to.
                      If None, all scenes in the split are used.
    """
    nusc = nuscenes.NuScenes(NUSCENES_VERSION, str(NUSCENES_ROOT), verbose=False)

    all_splits   = create_splits_scenes()
    eval_set     = f'mini_{split}' if NUSCENES_VERSION == 'v1.0-mini' else split
    split_scenes = set(all_splits[eval_set])

    if scenes:
        unknown = set(scenes) - split_scenes
        if unknown:
            raise ValueError(
                f"Scene(s) not in {eval_set}: {unknown}\n"
                f"Available: {sorted(split_scenes)}"
            )
        active_scenes = set(scenes)
    else:
        active_scenes = split_scenes

    active_sample_tokens = [
        s['token'] for s in nusc.sample
        if nusc.get('scene', s['scene_token'])['name'] in active_scenes
    ]

    print(f"Scenes   : {sorted(active_scenes)}")
    print(f"Samples  : {len(active_sample_tokens)}")

    results: Dict[str, list] = {token: [] for token in active_sample_tokens}
    skipped = 0

    for sample_token in active_sample_tokens:
        for ann_token in nusc.get('sample', sample_token)['anns']:
            ann      = nusc.get('sample_annotation', ann_token)
            det_name = nuscenes_category_to_detection_name(ann['category_name'], mapping_name)
            if det_name is None:
                skipped += 1
                continue
            results[sample_token].append({
                'sample_token':    sample_token,
                'translation':     ann['translation'],
                'size':            ann['size'],
                'rotation':        ann['rotation'],
                'velocity':        nusc.box_velocity(ann_token)[:2].tolist(),
                'detection_name':  det_name,
                'detection_score': 1.0,
                'attribute_name':  '',
            })

    total = sum(len(v) for v in results.values())
    print(f"GT boxes : {total}  (skipped {skipped} unmapped categories)")

    return {
        'split':        split,
        'mapping_name': mapping_name,
        'meta': {
            'use_camera': False, 'use_lidar': True,
            'use_radar':  False, 'use_map':   False, 'use_external': False,
        },
        'results': results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Build a GT submission JSON for evaluation pipeline testing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--split',   choices=['train', 'val'], default='train')
    parser.add_argument('--mapping', choices=VALID_MAPPINGS, default='8class',
                        dest='mapping_name')
    parser.add_argument('--scenes',  nargs='+', default=None,
                        help='Restrict to specific scene names')
    parser.add_argument('--out',     type=Path, default=None,
                        help='Output path (default: gt_submission_<split>_<mapping>.json)')
    args = parser.parse_args()

    out_path = args.out or Path(
        'gt_submission_'
        + args.split + '_' + args.mapping_name
        + (('_' + '_'.join(args.scenes)) if args.scenes else '')
        + '.json'
    )

    print(f'Split    : {args.split}')
    print(f'Mapping  : {args.mapping_name}')

    submission = make_gt_submission(args.split, args.mapping_name, args.scenes)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(submission, f)
    print(f'Written  : {out_path}')

    if args.scenes:
        print()
        print('To evaluate this scene-restricted submission, pass the same --scenes flag:')
        print(f'  python evaluation_nuscenes.py --submission {out_path} --scenes {" ".join(args.scenes)}')


if __name__ == '__main__':
    main()
