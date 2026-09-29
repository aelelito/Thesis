"""
Ground truth label extraction and evaluation utility.

This script extracts ground truth annotations from dataset scenes and converts them
to submission format for evaluation purposes. The pipeline performs these steps:
1. Load scenes from the dataset
2. Extract ground truth annotations using class mappings
3. Convert annotations to submission file format
4. Write submission files for evaluation
5. Run evaluation as a sanity check
"""
import argparse
from pathlib import Path
import yaml
import logging
import numpy as np
import rerun as rr
from typing import Dict, List

from src.labeling.mapping import get_class_mapping
from src.labeling.assign_labels import get_scene_gt_annotations_by_mapping
from src.labeling.submission import write_submission_files
from src.evaluate.evaluate import evaluate_submissions
from src.data.dataset import load_dataset_scene, get_dataset_scene_names
from src.structures import Scene, Annotation
from main_pseudo_union import get_out_dir

logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
np.set_printoptions(suppress=True)


def main() -> None:
    parser = argparse.ArgumentParser(description='Output GT labels to submission files')
    parser.add_argument('--config', type=Path, required=True)
    rr.script_add_args(parser)
    args = parser.parse_args()

    assert args.config.exists(), f"Config file {args.config} does not exist"
    assert args.config.suffix == '.yaml', f"Config file {args.config} is not a YAML file"

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    assert config['type'] == 'gt', f"Config file {args.config} is not a valid config file for this script"
    config['exp_name'] = args.config.stem

    scene_names_all: list[str] = get_dataset_scene_names(config['data'])

    gt_class_mapping = get_class_mapping(config['data']['dataset'])

    all_gt_annotations_by_mapping: List[Dict[str, Dict[str, List[Annotation]]]] = []
    data_split: List[str] = []
    for idx, scene_name in enumerate(scene_names_all):
        logger.info(f"#####Running extraction for scene {scene_name}, {idx+1}/{len(scene_names_all)}#####")
        base_scene: Scene = load_dataset_scene(config['data'], scene_name)

        gt_annotations_by_mapping = get_scene_gt_annotations_by_mapping(base_scene, gt_class_mapping)
        all_gt_annotations_by_mapping.append(gt_annotations_by_mapping)
        data_split.append(base_scene.split)

    logger.info("Writing submission files")
    submissions_gt: List[Path] = write_submission_files(all_gt_annotations_by_mapping, data_split,
                                                        get_out_dir(config, '#out_labels'),
                                                        f"{config['type']}_{config['exp_name']}", config['data']['dataset'])

    logger.info("Evaluating GT labels (sanity check)")
    evaluate_submissions(submissions_gt, config['data'], get_out_dir(config, '#out_eval'))


if __name__ == "__main__":
    main()
