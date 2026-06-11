#!/usr/bin/env python3
"""
Dispatcher: routes the evaluation task to the appropriate dataset-specific script based on the --dataset argument
            evaluates pseudo labels on nuScenes or ECP using nuScenes detection metrics.

Usage:
  python evaluation.py --dataset nuscenes --submission path/to/labels.json
  python evaluation.py --dataset ecp      --submission path/to/labels.json
"""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Evaluate pseudo labels (nuScenes or ECP)',
    )
    parser.add_argument('--dataset', choices=['nuscenes', 'ecp'], required=True, help='Dataset to evaluate on')

    # Pass remaining args through to the dataset-specific script
    args, remaining = parser.parse_known_args()

    if args.dataset == 'nuscenes':
        from evaluation_nuscenes import main as run
    elif args.dataset == 'ecp':
        from evaluation_ecp import main as run

    import sys
    sys.argv = [sys.argv[0]] + remaining
    
    # run dataset-specific evaluation
    run()


if __name__ == '__main__':
    main()
