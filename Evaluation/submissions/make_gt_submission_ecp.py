#!/usr/bin/env python3
"""
Create a "perfect" ECP submission JSON from manually annotated GT keyframes.

Running the ECP evaluator on this submission should yield (near) perfect scores,
confirming the evaluation pipeline is correctly wired up.

Only sample tokens that have at least one annotation in sample_annotation.json
are included — unannotated frames are automatically excluded.  As more frames
are annotated and added to the ECP NuScenes conversion, they are picked up
without any changes to this script.

ECP has no velocity ground truth; velocity is set to [0.0, 0.0] for all boxes.

Usage:
  python make_gt_submission_ecp.py
  python make_gt_submission_ecp.py --mapping 8class
  python make_gt_submission_ecp.py --scenes scene-euro-...
  python make_gt_submission_ecp.py --out my_gt_ecp.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import nuscenes

# Add parent directory so class_remapping can be imported when running from submissions/
sys.path.insert(0, str(Path(__file__).parent.parent))
from class_remapping import VALID_MAPPINGS, make_category_fn

ECP_ROOT    = Path('/media/lleba/ECP_Nuscenes_01/output2/ecp2nuscenes')
ECP_VERSION = 'v1.0-trainval'


def make_gt_submission_ecp(
    mapping_name: str,
    scenes: Optional[List[str]] = None,
    split: str = 'train',
) -> dict:
    """
    Build a submission dict from ECP manually annotated GT keyframes.

    Args:
        mapping_name: "1class", "3class", or "8class"
        scenes:       Optional list of scene names to restrict to.
                      If None, all annotated scenes are used.
    """
    nusc = nuscenes.NuScenes(ECP_VERSION, str(ECP_ROOT), verbose=False)

    # Discover annotated sample tokens (those with at least one GT box)
    annotated_tokens = {ann['sample_token'] for ann in nusc.sample_annotation}

    # Map sample token → scene name (only for annotated tokens)
    token_to_scene: Dict[str, str] = {}
    for sample in nusc.sample:
        if sample['token'] in annotated_tokens:
            token_to_scene[sample['token']] = nusc.get('scene', sample['scene_token'])['name']

    all_annotated_scenes = sorted({v for v in token_to_scene.values()})

    if scenes:
        unknown = set(scenes) - set(all_annotated_scenes)
        if unknown:
            raise ValueError(
                f"Scene(s) not in annotated set: {unknown}\n"
                f"Available: {all_annotated_scenes}"
            )
        active_scenes = set(scenes)
    else:
        active_scenes = set(all_annotated_scenes)

    active_tokens = [
        tok for tok, scene in token_to_scene.items()
        if scene in active_scenes
    ]

    print(f"Scenes   : {sorted(active_scenes)}")
    print(f"Samples  : {len(active_tokens)}")

    # ECP categories are already in short form ('car', 'pedestrian', …)
    category_fn = make_category_fn(mapping_name, dataset='ecp')

    results: Dict[str, list] = {token: [] for token in active_tokens}
    skipped = 0

    for ann in nusc.sample_annotation:
        if ann['sample_token'] not in results:
            continue
        instance      = nusc.get('instance', ann['instance_token'])
        category_name = nusc.get('category', instance['category_token'])['name']
        det_name = category_fn(category_name)
        if det_name is None:
            skipped += 1
            continue
        results[ann['sample_token']].append({
            'sample_token':    ann['sample_token'],
            'translation':     ann['translation'],
            'size':            ann['size'],
            'rotation':        ann['rotation'],
            'velocity':        [0.0, 0.0],   # ECP has no velocity GT
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
        description='Build a GT submission JSON for ECP evaluation pipeline testing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--split', default='train',
                        help='Split tag written into the submission JSON (default: train)')
    parser.add_argument('--mapping', choices=VALID_MAPPINGS, default='8class',
                        dest='mapping_name',
                        help='Class mapping scheme (default: 8class)')
    parser.add_argument('--scenes', nargs='+', default=None,
                        help='Restrict to specific annotated scene names')
    parser.add_argument('--out', type=Path, default=None,
                        help='Output path (default: gt_submission_ecp_<mapping>.json)')
    args = parser.parse_args()

    out_path = args.out or Path(
        f'gt_submission_ecp_{args.mapping_name}'
        + (('_' + '_'.join(args.scenes)) if args.scenes else '')
        + '.json'
    )

    print(f'Mapping  : {args.mapping_name}')

    submission = make_gt_submission_ecp(args.mapping_name, args.scenes, args.split)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(submission, f)
    print(f'Written  : {out_path}')
    print()
    print('To evaluate this submission:')
    print(f'  python evaluation_ecp.py --submission {out_path}')
    if args.scenes:
        print(f'  python evaluation_ecp.py --submission {out_path} --scenes {" ".join(args.scenes)}')


if __name__ == '__main__':
    main()
