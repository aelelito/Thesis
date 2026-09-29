#!/usr/bin/env python3
"""
Merge per-scene submission JSONs into one submission per class mapping.

    python merge_submissions.py --run-dir /workspace/autolabeling/output/nuscenes_mini/mini_full

Reads  <run-dir>/scenes/<scene>/<scene>_{8class,3class,1class}.json  and writes
       <run-dir>/autolabel_<name>_{8class,3class,1class}.json   (default name: nuscenes_mini).
Fails if a scene is missing or two scenes contain the same sample token (--allow-partial to skip missing ones).
"""
import argparse
import json
import sys
from pathlib import Path

MAPPINGS = ('8class', '3class', '1class')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--scenes-file', type=Path, default=Path(__file__).parent / 'configs' / 'scenes_nuscenes_mini_train.txt',
                    help='expected scenes, one per line')
    ap.add_argument('--name', default='nuscenes_mini', help='dataset part of the output file name')
    ap.add_argument('--allow-partial', action='store_true')
    args = ap.parse_args()

    scenes = [s.strip() for s in args.scenes_file.read_text().split('\n') if s.strip()]
    name = args.name
    sdir = args.run_dir / 'scenes'

    for mapping in MAPPINGS:
        merged, seen_in, missing = None, {}, []
        for scene in scenes:
            p = sdir / scene / f'{scene}_{mapping}.json'
            if not p.exists():
                missing.append(scene)
                continue
            sub = json.loads(p.read_text())
            if merged is None:
                merged = {k: v for k, v in sub.items() if k != 'results'}
                merged['results'] = {}
                merged['merged_scenes'] = []
            for key in ('split', 'mapping_name'):
                if sub[key] != merged[key]:
                    sys.exit(f'{p}: {key} {sub[key]!r} differs from {merged[key]!r}')
            dup = set(sub['results']) & set(merged['results'])
            if dup:
                sys.exit(f'{p}: {len(dup)} sample token(s) already seen in {seen_in[next(iter(dup))]}')
            for tok in sub['results']:
                seen_in[tok] = scene
            merged['results'].update(sub['results'])
            merged['merged_scenes'].append(scene)
        if missing and not args.allow_partial:
            sys.exit(f'{mapping}: missing scene(s) {missing}. Finish them or pass --allow-partial.')
        if merged is None:
            sys.exit(f'{mapping}: nothing to merge under {sdir}')
        out = args.run_dir / f'autolabel_{name}_{mapping}.json'
        out.write_text(json.dumps(merged))
        n_box = sum(len(v) for v in merged['results'].values())
        print(f'{out}  ({len(merged["merged_scenes"])} scenes, {len(merged["results"])} frames, {n_box} boxes)'
              + (f'  [missing: {missing}]' if missing else ''))


if __name__ == '__main__':
    main()
