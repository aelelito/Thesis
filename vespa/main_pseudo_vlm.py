"""
Approach: VESPA-Direct
Vision-Language Model (VLM) based pseudo label generation pipeline.

This script generates pseudo labels for 3D object detection using a vision-language
model approach. The pipeline processes LiDAR and camera data through the following steps:
1. Ground point removal from LiDAR data
2. Object segmentation using Grounding SAM (VLM + SAM)
3. LiDAR-to-segmentation reprojection for 3D clustering
4. Object cluster denoising and refinement
5. Appearance embedding extraction from segmentation masks
6. Multi-camera object merging and tracking
7. Velocity estimation and yaw angle calculation
8. Bounding box size inflation and object filtering
9. Pseudo class label assignment and evaluation
"""
import argparse
from pathlib import Path
import yaml
import logging
import numpy as np
import rerun as rr
import multiprocessing
from typing import Dict, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

from main_pseudo_union import get_out_dir, _suppress_output
from src.data.dataset import load_dataset_scene, get_dataset_scene_names
from src.image.grounding_sam import generate_segmentation_masks
from src.image.segmentation_reprojection import reproject_lidar_to_segmentation
from src.pointcloud.cluster_denoising import denoise_object_clusters
from src.pointcloud.remove_ground import remove_ground
from src.pointcloud.bbox_heuristic import estimate_yaw_from_tracking_and_shape, inflate_bboxes_to_common_size
from src.image.appearance_embedding import appearance_embed_mask_based
from src.tracking.appearance_tracking import estimate_velocity_by_appearance_tracking
from src.image.object_merge import merge_multicamera_objects
from src.labeling.assign_labels import assign_pseudo_class_labels_from_sam_label, get_scene_pseudo_annotations_by_mapping
from src.structures import Scene, CommonObjectInfo, Annotation
from src.labeling.submission import write_submission_files
from src.labeling.filtering import filter_objects
from src.labeling.prior_info import get_common_object_infos_sam
from src.evaluate.evaluate import evaluate_submissions
from src.visualizations.modules import visualize_base_scene, visualize_camera_segmentations, visualize_lidar
from src.visualizations.modules import visualize_object_pointclouds, visualize_object_boxes
from src.visualizations.modules import visualize_object_velocities
from src.visualizations.modules import visualize_mapped_pseudo_labeled_boxes
from src.visualizations.rerun import get_rerun_blueprint

logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
np.set_printoptions(suppress=True)


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate pseudo labels')
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--scenes', type=str, nargs='+', help='List of scenes')
    parser.add_argument('--frame-start', type=int, default=None, help='First frame index to process (inclusive)')
    parser.add_argument('--frame-end', type=int, default=None, help='Last frame index to process (exclusive)')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--workers', type=int, default=1)
    rr.script_add_args(parser)
    args = parser.parse_args()

    assert args.config.exists(), f"Config file {args.config} does not exist"
    assert args.config.suffix == '.yaml', f"Config file {args.config} is not a YAML file"

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    assert config['type'] == 'vlm', f"Config file {args.config} is not a valid config file for this script"
    config['exp_name'] = args.config.stem

    scene_names_all: list[str] = get_dataset_scene_names(config['data'])
    scene_names = scene_names_all if args.scenes is None else args.scenes
    assert all([scene_name in scene_names_all for scene_name in scene_names]), \
        f"Some scenes {args.scenes} are not in the dataset"

    if args.scenes is not None and len(args.scenes) == 1:
        config['exp_name'] = f"{config['exp_name']}/{args.scenes[0]}"

    logger.info("Loading object sizes")
    obj_info: Dict[str, CommonObjectInfo] = get_common_object_infos_sam(config['grounding_sam']['class_names'])

    all_pseudo_annotations_by_mapping: List[Dict[str, Dict[str, List[Annotation]]]] = []
    data_split: List[str] = []
    if args.workers > 1:
        logger.info(f"Running generation with {args.workers} workers")
        multiprocessing.set_start_method("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_suppress_output) as executor:
            futures = {executor.submit(process_scene, scene_name, obj_info, config, args): scene_name \
                       for scene_name in scene_names}
            done_nr = 0
            for future in as_completed(futures):
                pseudo_annotations_by_mapping, split = future.result()
                all_pseudo_annotations_by_mapping.append(pseudo_annotations_by_mapping)
                data_split.append(split)
                done_nr += 1
                logger.info(f"Completed scene:{futures[future]},{done_nr}/{len(futures)}")
    else:
        logger.info("Running generation without parallelization")
        for idx, scene_name in enumerate(scene_names):
            logger.info(f"Running generation for scene {scene_name}, {idx+1}/{len(scene_names)}")
            pseudo_annotations_by_mapping, split = process_scene(scene_name, obj_info, config, args)
            all_pseudo_annotations_by_mapping.append(pseudo_annotations_by_mapping)
            data_split.append(split)

    # Compare against scene_names (requested subset) not scene_names_all,
    # so output is written even when only a subset of scenes is processed via --scenes.
    # To restore the original guard (only write output when ALL dataset scenes are processed),
    # uncomment the line below and remove the current condition:
    # if len(all_pseudo_annotations_by_mapping) == len(scene_names_all):
    if len(all_pseudo_annotations_by_mapping) == len(scene_names) and len(scene_names) > 0:
        logger.info("All scenes processed, writing submission files")
        submissions_pseudo: List[Path] = write_submission_files(all_pseudo_annotations_by_mapping, data_split,
                                                                get_out_dir(config, '#out_labels'),
                                                                f"{config['type']}_{config['exp_name']}", config['data']['dataset'])
        # Only evaluate when all dataset scenes are processed; partial runs would give misleading mAP
        # (missing scenes count as zero detections, artificially lowering recall).
        if len(scene_names) == len(scene_names_all):
            logger.info("Evaluating pseudo labels")
            evaluate_submissions(submissions_pseudo, config['data'], get_out_dir(config, '#out_eval'))
        else:
            logger.info("Skipping evaluation: only a subset of scenes was processed")
        # logger.info("Evaluating pseudo labels")
        # evaluate_submissions(submissions_pseudo, config['data'], get_out_dir(config, '#out_eval'))
    else:
        logger.info("Not all scenes processed, skipping submission files and evaluation")

    if args.visualize:
        rr.script_teardown(args)


def process_scene(
    scene_name: str,
    obj_info: dict[str, CommonObjectInfo],
    config: dict,
    args: argparse.Namespace
) -> Tuple[Dict[str, Dict[str, List[Annotation]]], List[str]]:
    logger.info("Loading base scene")
    base_scene: Scene = load_dataset_scene(config['data'], scene_name,
                                           frame_start=args.frame_start, frame_end=args.frame_end)
    logger.info(f"Loaded {len(base_scene.frames)} frames "
                f"[{args.frame_start}:{args.frame_end}]")

    logger.info("Removing ground points")
    scene_wo_ground: Scene = remove_ground(base_scene,
                                           config['remove_ground'],
                                           get_out_dir(config, '01_remove_ground'))

    logger.info("Generating segmentation masks")
    scene_w_segmentation: Scene = generate_segmentation_masks(scene_wo_ground,
                                                              config['grounding_sam'],
                                                              get_out_dir(config, '02_grounding_sam'))

    logger.info("Reprojecting lidar to segmentation")
    scene_w_segclusters = reproject_lidar_to_segmentation(scene_w_segmentation,
                                                          config['reprojection'],
                                                          get_out_dir(config, '03_reprojection'))

    logger.info("Denoising object clusters")
    scene_w_segclusters_denoised = denoise_object_clusters(scene_w_segclusters,
                                                           obj_info,
                                                           config['denoising'],
                                                           get_out_dir(config, '04_denoising'))

    logger.info("Appearance embedding based on masks")
    scene_w_segclusters_embed = appearance_embed_mask_based(scene_w_segclusters_denoised,
                                                            config['appearance_embedding'],
                                                            get_out_dir(config, '05_appearance_embedding_mask'))

    logger.info("Merging multicamera objects")
    # When running with a single camera, clear cross-camera pairs so the merge step is a no-op
    # across cameras (LiDAR-based deduplication within the single camera still runs).
    multicam_merge_config = config['multicam_object_merge']
    if config['data'].get('single_camera', None) is not None:
        multicam_merge_config = {**multicam_merge_config, 'camera_pairs_left_right': []}
    scene_w_segclusters_merged = merge_multicamera_objects(scene_w_segclusters_embed,
                                                           obj_info,
                                                           multicam_merge_config,
                                                           get_out_dir(config, '06_multicam_object_merge'))

    logger.info("Estimating velocity by appearance tracking")
    scene_w_segclusters_velo = estimate_velocity_by_appearance_tracking(scene_w_segclusters_merged,
                                                                        obj_info,
                                                                        config['velocity_by_app_tracking'],
                                                                        get_out_dir(config, '07_velocity_by_tracking'))

    logger.info("Estimating yaw from tracking and shape")
    scene_w_segclusters_yaw = estimate_yaw_from_tracking_and_shape(scene_w_segclusters_velo,
                                                                   obj_info, config['yaw_heuristic'])

    logger.info("Inflating bboxes to common size")
    scene_w_segclusters_inflated = inflate_bboxes_to_common_size(scene_w_segclusters_yaw, obj_info, config['bbox_inflation'])

    logger.info("Filtering objects")
    scene_w_segclusters_filtered = filter_objects(scene_w_segclusters_inflated, config['object_filtering'])

    logger.info("Assigning pseudo class labels")
    scene_w_pslab_segclusters = assign_pseudo_class_labels_from_sam_label(scene_w_segclusters_filtered)

    logger.info("Creating pseudo annotations")
    pseudo_annotations_by_mapping = get_scene_pseudo_annotations_by_mapping(scene_w_pslab_segclusters)

    if args.visualize:
        logger.info("Setting up rerun")
        rerun_blueprint = get_rerun_blueprint(base_scene.camera_to_image.keys())
        rr.script_setup(args, scene_name, default_blueprint=rerun_blueprint)

        logger.info("Visualizing base scene")
        visualize_base_scene(base_scene, config.get('visualization', {}))

        logger.info("Visualizing scene without ground")
        visualize_lidar(scene_wo_ground, "01_remove_ground_LIDAR")

        logger.info("Visualizing segmented scene")
        visualize_camera_segmentations(scene_w_segmentation)

        logger.info("Visualizing segmented reprojected scene")
        visualize_object_pointclouds(scene_w_segclusters, "03_reprojection_LIDAR", "sam_label")
        visualize_object_boxes(scene_w_segclusters, "03_reprojection_LIDAR", "03_reprojection_bbox", "sam_label")

        logger.info("Visualizing denoised segmented reprojected scene")
        visualize_object_pointclouds(scene_w_segclusters_denoised, "04_denoising_LIDAR", "sam_label")
        visualize_object_boxes(scene_w_segclusters_denoised, "04_denoising_LIDAR", "04_denoising_bbox", "sam_label")

        logger.info("Visualizing merged segmented reprojected scene")
        visualize_object_pointclouds(scene_w_segclusters_merged, "06_multicam_object_merge_LIDAR", "sam_label")
        visualize_object_boxes(scene_w_segclusters_merged, "06_multicam_object_merge_LIDAR", "06_multicam_merge_bbox", "sam_label")

        logger.info("Visualizing velocities after appearance tracking")
        visualize_object_velocities(scene_w_segclusters_velo, "06_multicam_object_merge_LIDAR", "07_tracking_velo")

        logger.info("Visualizing scene with adjusted yaw")
        visualize_object_boxes(scene_w_segclusters_yaw, "06_multicam_object_merge_LIDAR", "08_yaw_heuristic_bbox", "sam_label")

        logger.info("Visualizing scene with inflated bboxes")
        visualize_object_boxes(scene_w_segclusters_inflated, "06_multicam_object_merge_LIDAR", "09_bbox_inflation_bbox", "sam_label")

        logger.info("Visualizing scene after filtering")
        visualize_object_boxes(scene_w_segclusters_filtered, "06_multicam_object_merge_LIDAR", "10_object_filtering_bbox", "sam_label")

        logger.info("Visualizing scene after pseudo class labeling")
        visualize_mapped_pseudo_labeled_boxes(scene_w_pslab_segclusters, "06_multicam_object_merge_LIDAR")

    return pseudo_annotations_by_mapping, scene_w_pslab_segclusters.split


if __name__ == '__main__':
    main()
