#!/usr/bin/env python3
"""
Merge per-scene submission JSONs for a probe run, tolerating two filename conventions:
  scenes/<scene>/<scene>_<mapping>.json          (new, scene-based run-name)
  scenes/<scene>/<anything>_<mapping>.json       (old, fixed run-name across the array)
Prefers the scene-named file when both exist (it is the more recent, post-fix recompute).
"""
import argparse
import json
import sys
from pathlib import Path

MAPPINGS = ('8class', '3class', '1class')


def find_scene_file(scene_dir: Path, scene: str, mapping: str):
    p = scene_dir / f'{scene}_{mapping}.json'
    if p.exists():
        return p
    cands = sorted(scene_dir.glob(f'*_{mapping}.json'))
    return cands[0] if cands else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--scenes-file', type=Path,
                     default=Path(__file__).parent / 'configs' / 'scenes_nuscenes_mini_train.txt')
    ap.add_argument('--name', default='nuscenes_mini')
    args = ap.parse_args()

    scenes = [s.strip() for s in args.scenes_file.read_text().split('\n') if s.strip()]
    sdir = args.run_dir / 'scenes'

    for mapping in MAPPINGS:
        merged, seen_in, missing = None, {}, []
        for scene in scenes:
            sd = sdir / scene
            p = find_scene_file(sd, scene, mapping) if sd.exists() else None
            if p is None:
                missing.append(scene)
                continue
            sub = json.loads(p.read_text())
            if merged is None:
                merged = {k: v for k, v in sub.items() if k != 'results'}
                merged['results'] = {}
                merged['merged_scenes'] = []
            dup = set(sub['results']) & set(merged['results'])
            if dup:
                sys.exit(f'{p}: {len(dup)} sample token(s) already seen in {seen_in[next(iter(dup))]}')
            for tok in sub['results']:
                seen_in[tok] = scene
            merged['results'].update(sub['results'])
            merged['merged_scenes'].append(scene)
        if missing:
            sys.exit(f'{args.run_dir}  {mapping}: missing scene(s) {missing}')
        out = args.run_dir / f'autolabel_{args.name}_{mapping}.json'
        out.write_text(json.dumps(merged))
        n_box = sum(len(v) for v in merged['results'].values())
        print(f'{out}  ({len(merged["merged_scenes"])} scenes, {len(merged["results"])} frames, {n_box} boxes)')


if __name__ == '__main__':
    main()
