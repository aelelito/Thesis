#!/usr/bin/env python3
"""
Shrink existing SAM3D Objects checkpoints: keep `--mesh-points` random vertices per object, drop the faces
(bicycles/motorcycles keep all vertices). Same deterministic subsampling the pipeline applies. LOSSY and
irreversible for the meshes -- dry run unless --apply. NOTE: a checkpoint written before 'obb_raw' existed has no
full-mesh box stored, so after slimming its OBB can only be refitted on the reduced vertices (a fresh pipeline run
stores the full-mesh box automatically).

    python slim_checkpoints.py --root /workspace/autolabeling/output/nuscenes_mini/front_cam_o1            # dry run
    python slim_checkpoints.py --root /workspace/autolabeling/output/nuscenes_mini/front_cam_o1 --apply
"""
import argparse
import gzip
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from autolabeling.utils.meshes import FULL_MESH_CLASSES, slim_object_results   # noqa: E402  (numpy only, no torch)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', type=Path, required=True, help='any folder; every objects__*/*.pkl.gz below it is processed')
    ap.add_argument('--mesh-points', type=int, default=20000)
    ap.add_argument('--apply', action='store_true', help='rewrite the files (default: only report)')
    args = ap.parse_args()

    files = sorted(args.root.rglob('objects__*/*.pkl.gz'))
    if not files:
        sys.exit(f'no objects__*/*.pkl.gz under {args.root}')
    before = after = n_done = 0
    for p in files:
        size = p.stat().st_size
        before += size
        with gzip.open(p, 'rb') as f:
            objs = pickle.load(f)
        if all(r.get('faces') is None and (r.get('prompt') in FULL_MESH_CLASSES or len(r['vertices']) <= args.mesh_points) for r in objs):
            after += size                                   # already slim
            continue
        slim_object_results(objs, args.mesh_points)
        tmp = p.with_suffix('.tmp')
        with gzip.open(tmp, 'wb', compresslevel=3) as f:
            pickle.dump(objs, f, protocol=4)
        new_size = tmp.stat().st_size
        after += new_size
        n_done += 1
        if args.apply:
            os.replace(tmp, p)                              # atomic: a killed run never leaves a half-written file
        else:
            tmp.unlink()
    gb = lambda b: f'{b / 1e9:.2f} GB'
    print(f'{len(files)} checkpoint files, {n_done} to slim: {gb(before)} -> {gb(after)}'
          + ('' if args.apply else '   (dry run; add --apply to rewrite)'))


if __name__ == '__main__':
    main()
