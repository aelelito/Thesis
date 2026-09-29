"""
Approach: VESPA-Fusion with appearance-clustering-based object discovery
Enhanced combined pseudo label generation with appearance-based clustering.

This script generates pseudo labels by combining VLM-based detection, spatial clustering,
and appearance-based clustering. The pipeline runs in two passes:

First pass (per scene):
1. Ground point removal from LiDAR data
2. Spatial clustering of LiDAR points
3. Scene flow estimation for velocity calculation
4. Object segmentation using Grounding SAM
5. LiDAR-to-segmentation reprojection
6. Object cluster denoising and multi-camera merging
7. VLM-based and clustering-based object fusion
8. Appearance embedding extraction

Second pass (global clustering + labeling):
9. Cross-scene appearance clustering based on labels
10. Label assignment from appearance clusters
11. Velocity tracking and yaw estimation
12. Bounding box inflation and object filtering
13. Pseudo label assignment and evaluation
"""

import argparse
from pathlib import Path
import yaml
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import rerun as rr
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from main_pseudo_union import get_out_dir, _suppress_output
from src.data.dataset import load_dataset_scene, get_dataset_scene_names
from src.labeling.prior_info import get_common_object_infos_sam
from src.pointcloud.remove_ground import remove_ground
from src.pointcloud.spatial_clustering import get_spatial_clustering
from src.pointcloud.sceneflow import estimate_sceneflow
from src.image.grounding_sam import generate_segmentation_masks
from src.image.segmentation_reprojection import reproject_lidar_to_segmentation
from src.pointcloud.cluster_denoising import denoise_object_clusters
from src.image.object_merge import merge_multicamera_objects, merger_vlm_based_and_clustering_based_objects
from src.image.appearance_embedding import appearance_embed_lidar_based
from src.tracking.appearance_tracking import estimate_velocity_by_appearance_tracking
from src.pointcloud.bbox_heuristic import estimate_yaw_from_tracking_and_shape, inflate_bboxes_to_common_size
from src.labeling.filtering import filter_objects
from src.image.appearance_clustering import calculate_appearance_clusters_label_based, is_appreance_cluster_calculated
from src.labeling.assign_labels import assign_pseudo_class_labels_from_sam_label, get_scene_pseudo_annotations_by_mapping
from src.labeling.assign_labels import label_from_appearance_clusters
from src.evaluate.evaluate import evaluate_submissions
from src.labeling.submission import write_submission_files
from src.structures import Scene, CommonObjectInfo, Annotation
from src.visualizations.rerun import get_rerun_blueprint
from src.visualizations.modules import visualize_base_scene, visualize_lidar
from src.visualizations.modules import visualize_camera_segmentations
from src.visualizations.modules import visualize_object_velocities, visualize_mapped_pseudo_labeled_boxes
from src.visualizations.modules import visualize_object_pointclouds, visualize_object_boxes

logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
np.set_printoptions(suppress=True)


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate pseudo labels')
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--scenes', type=str, nargs='+', help='List of scenes')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--workers', type=int, default=1)
    rr.script_add_args(parser)
    args = parser.parse_args()

    assert args.config.exists(), f"Config file {args.config} does not exist"
    assert args.config.suffix == '.yaml', f"Config file {args.config} is not a YAML file"

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    assert config['type'] == 'combined2', f"Config file {args.config} is not a valid config file for this script"
    config['exp_name'] = args.config.stem

    scene_names_all: list[str] = get_dataset_scene_names(config['data'])
    scene_names = scene_names_all if args.scenes is None else args.scenes
    assert all([scene_name in scene_names_all for scene_name in scene_names]), \
        f"Some scenes {args.scenes} are not in the dataset"

    appearance_embeddings_all: List[np.ndarray] = []
    sam_labels_all: List[Optional[str]] = []
    if not is_appreance_cluster_calculated(get_out_dir(config, '10_appearance_clustering')):
        if args.workers > 1:
            logger.info(f"Running first pass with {args.workers} workers")
            multiprocessing.set_start_method("spawn")
            with ProcessPoolExecutor(max_workers=args.workers, initializer=_suppress_output) as executor:
                futures = {executor.submit(process_scene, scene_name, config, args): scene_name \
                           for scene_name in scene_names}
                done_nr = 0
                for future in as_completed(futures):
                    appearance_embeddings, sam_labels = future.result()
                    done_nr += 1
                    appearance_embeddings_all.extend(appearance_embeddings)
                    sam_labels_all.extend(sam_labels)
                    logger.info(f"Completed scene:{futures[future]},{done_nr}/{len(futures)}")
        else:
            logger.info("Running first pass without parallelization")
            for idx, scene_name in enumerate(scene_names):
                logger.info(f"Running first pass for scene {scene_name}, {idx+1}/{len(scene_names)}")
                appearance_embeddings, sam_labels = process_scene(scene_name, config, args)
                appearance_embeddings_all.extend(appearance_embeddings)
                sam_labels_all.extend(sam_labels)

    assert is_appreance_cluster_calculated(get_out_dir(config, '10_appearance_clustering')) \
        or len(scene_names) == len(scene_names_all), \
        f"To get appearance clusters, first pass has to be run for all scenes or appearance clusters have to be cached"

    clustering = calculate_appearance_clusters_label_based(appearance_embeddings_all, sam_labels_all,
                                                           config['appearance_clustering_labels'],
                                                           get_out_dir(config, '10_appearance_clustering'))

    logger.info("Running second pass: labeling")
    all_pseudo_annotations_by_mapping: List[Dict[str, Dict[str, List[Annotation]]]] = []
    data_split: List[str] = []
    obj_info: Dict[str, CommonObjectInfo] = get_common_object_infos_sam(config['grounding_sam']['class_names'])
    for idx, scene_name in enumerate(scene_names):
        logger.info(f"#####Running labeling (and visualization) for scene {scene_name}, {idx+1}/{len(scene_names)}#####")
        logger.info("Loading cached steps")
        base_scene: Scene = load_dataset_scene(config['data'], scene_name)
        scene_wo_ground: Scene = remove_ground(base_scene,
                                               config['remove_ground'],
                                               get_out_dir(config, '01_remove_ground'))
        scene_w_clusters: Scene = get_spatial_clustering(scene_wo_ground,
                                                         config['spatial_clustering'],
                                                         get_out_dir(config, '02_spatial_clustering'))
        scene_w_velo_clusters: Scene = estimate_sceneflow(scene_w_clusters,
                                                          config['sceneflow'],
                                                          get_out_dir(config, '03_sceneflow'))
        scene_w_segmentation: Scene = generate_segmentation_masks(scene_wo_ground,
                                                                  config['grounding_sam'],
                                                                  get_out_dir(config, '04_grounding_sam'))
        scene_w_segclusters = reproject_lidar_to_segmentation(scene_w_segmentation,
                                                              config['reprojection'],
                                                              get_out_dir(config, '05_reprojection'))
        scene_w_segclusters_denoised = denoise_object_clusters(scene_w_segclusters,
                                                               obj_info,
                                                               config['denoising'],
                                                               get_out_dir(config, '06_denoising'))
        scene_w_segclusters_cammerged = merge_multicamera_objects(scene_w_segclusters_denoised,
                                                                  obj_info,
                                                                  config['multicam_object_merge'],
                                                                  get_out_dir(config, '07_multicam_object_merge'))
        scene_w_clusters_merged = merger_vlm_based_and_clustering_based_objects(scene_w_segclusters_cammerged,
                                                                                scene_w_velo_clusters,
                                                                                obj_info,
                                                                                config['vlm_and_cluster_merge'],
                                                                                get_out_dir(config, '08_vlm_and_cluster_merge'))
        scene_w_embeddings: Scene = appearance_embed_lidar_based(scene_w_clusters_merged,
                                                                 config['appearance_embedding'],
                                                                 get_out_dir(config, '09_appearance_embedding'))

        logging.info("Labeling objects with clustering labels")
        scene_w_labels: Scene = label_from_appearance_clusters(scene_w_embeddings,
                                                             clustering,
                                                             config['appearance_clustering_labels'])
        
        logger.info("Estimating velocity by appearance tracking")
        scene_w_velo = estimate_velocity_by_appearance_tracking(scene_w_labels,
                                                                obj_info,
                                                                config['velocity_by_app_tracking'],
                                                                get_out_dir(config, '11_velocity_by_tracking'))

        logger.info("Estimating yaw from tracking and shape")
        scene_w_segclusters_yaw = estimate_yaw_from_tracking_and_shape(scene_w_velo,
                                                                    obj_info, config['yaw_heuristic'])

        logger.info("Inflating bboxes to common size")
        scene_w_segclusters_inflated = inflate_bboxes_to_common_size(scene_w_segclusters_yaw, obj_info, config['bbox_inflation'])

        logger.info("Filtering objects")
        scene_w_segclusters_filtered = filter_objects(scene_w_segclusters_inflated, config['object_filtering'])

        logger.info("Assigning pseudo class labels")
        scene_w_pslab_segclusters = assign_pseudo_class_labels_from_sam_label(scene_w_segclusters_filtered)

        logger.info("Creating pseudo annotations")
        pseudo_annotations_by_mapping = get_scene_pseudo_annotations_by_mapping(scene_w_pslab_segclusters)
        all_pseudo_annotations_by_mapping.append(pseudo_annotations_by_mapping)
        data_split.append(scene_w_pslab_segclusters.split)
        
        if args.visualize:
            logger.info("Setting up rerun")
            rerun_blueprint = get_rerun_blueprint(base_scene.camera_to_image.keys())
            rr.script_setup(args, scene_name, default_blueprint=rerun_blueprint)

            logger.info("Visualizing scene")
            visualize_base_scene(base_scene, config.get('visualization', {}))

            logger.info("Visualizing scene without ground")
            visualize_lidar(scene_wo_ground, "01_remove_ground_LIDAR")

            logger.info("Visualizing scene with clusters")
            visualize_object_pointclouds(scene_w_clusters, "02_clustering_LIDAR", "index")
            visualize_object_boxes(scene_w_clusters, "02_clustering_LIDAR", "02_clustering_bbox", "index")

            logger.info("Visualizing scene after scene flow")
            visualize_object_pointclouds(scene_w_velo_clusters, "03_sceneflow_LIDAR", "index")
            visualize_object_boxes(scene_w_velo_clusters, "03_sceneflow_LIDAR", "03_sceneflow_bbox", "index")
            visualize_object_velocities(scene_w_velo_clusters, "03_sceneflow_LIDAR", "03_sceneflow_velo")

            logger.info("Visualizing segmented scene")
            visualize_camera_segmentations(scene_w_segmentation)

            logger.info("Visualizing segmented reprojected scene")
            visualize_object_pointclouds(scene_w_segclusters, "04_reprojection_LIDAR", "sam_label")
            visualize_object_boxes(scene_w_segclusters, "04_reprojection_LIDAR", "04_reprojection_bbox", "sam_label")

            logger.info("Visualizing denoised segmented reprojected scene")
            visualize_object_pointclouds(scene_w_segclusters_denoised, "05_denoising_LIDAR", "sam_label")
            visualize_object_boxes(scene_w_segclusters_denoised, "05_denoising_LIDAR", "06_denoising_bbox", "sam_label")

            logger.info("Visualizing merged segmented reprojected scene")
            visualize_object_pointclouds(scene_w_segclusters_cammerged, "07_multicam_object_merge_LIDAR", "sam_label")
            visualize_object_boxes(scene_w_segclusters_cammerged, "07_multicam_object_merge_LIDAR", "07_multicam_merge_bbox", "sam_label")

            logger.info("Visualizing fused scene")
            visualize_object_pointclouds(scene_w_clusters_merged, "08_fused_LIDAR", "sam_label")
            visualize_object_boxes(scene_w_clusters_merged, "08_fused_LIDAR", "08_fused_bbox", "sam_label")
            visualize_object_velocities(scene_w_clusters_merged, "08_fused_LIDAR", "08_fused_velo")

            logger.info("Visualizing scene after appearance clustering and relabeling")
            visualize_object_pointclouds(scene_w_labels, "09_fused_relabeled_LIDAR", "sam_label")
            visualize_object_boxes(scene_w_labels, "09_fused_relabeled_LIDAR", "09_fused_relabeled_bbox", "sam_label")
            visualize_object_velocities(scene_w_labels, "09_fused_relabeled_LIDAR", "09_fused_relabeled_velo")

            logger.info("Visualizing velocities after appearance tracking")
            visualize_object_velocities(scene_w_velo, "09_fused_relabeled_LIDAR", "10_tracking_velo")

            logger.info("Visualizing scene with adjusted yaw")
            visualize_object_boxes(scene_w_segclusters_yaw, "09_fused_relabeled_LIDAR", "11_yaw_heuristic_bbox", "sam_label")

            logger.info("Visualizing scene with inflated bboxes")
            visualize_object_boxes(scene_w_segclusters_inflated, "09_fused_relabeled_LIDAR", "12_bbox_inflation_bbox", "sam_label")

            logger.info("Visualizing scene after filtering")
            visualize_object_boxes(scene_w_segclusters_filtered, "09_fused_relabeled_LIDAR", "13_object_filtering_bbox", "sam_label")

            logger.info("Visualizing scene after pseudo class labeling")
            visualize_mapped_pseudo_labeled_boxes(scene_w_pslab_segclusters, "09_fused_relabeled_LIDAR")

    if len(all_pseudo_annotations_by_mapping) == len(scene_names_all):
        logger.info("All scenes processed, writing submission files")
        submissions_pseudo: List[Path] = write_submission_files(all_pseudo_annotations_by_mapping, data_split,
                                                                get_out_dir(config, '#out_labels'), 
                                                                f"{config['type']}_{config['exp_name']}", config['data']['dataset'])
        
        logger.info("Evaluating pseudo labels")
        evaluate_submissions(submissions_pseudo, config['data'], get_out_dir(config, '#out_eval'))
    else:
        logger.info("Not all scenes processed, skipping submission files and evaluation")

    if args.visualize:
        rr.script_teardown(args)


def process_scene(scene_name: str, config: dict, args: argparse.Namespace) -> Tuple[List[np.ndarray], List[Optional[str]]]:
    logger.info(f"##### Running for scene {scene_name} #####")

    obj_info: Dict[str, CommonObjectInfo] = get_common_object_infos_sam(config['grounding_sam']['class_names'])

    logger.info("Loading base scene")
    base_scene: Scene = load_dataset_scene(config['data'], scene_name)

    logger.info("Removing ground points")
    scene_wo_ground: Scene = remove_ground(base_scene,
                                           config['remove_ground'],
                                           get_out_dir(config, '01_remove_ground'))

    logger.info("Spatial clustering")
    scene_w_clusters: Scene = get_spatial_clustering(scene_wo_ground,
                                                     config['spatial_clustering'],
                                                     get_out_dir(config, '02_spatial_clustering'))

    logger.info("Estimating scene flow")
    scene_w_velo_clusters: Scene = estimate_sceneflow(scene_w_clusters,
                                                      config['sceneflow'],
                                                      get_out_dir(config, '03_sceneflow'))

    logger.info("Generating segmentation masks")
    scene_w_segmentation: Scene = generate_segmentation_masks(scene_wo_ground,
                                                              config['grounding_sam'],
                                                              get_out_dir(config, '04_grounding_sam'))

    logger.info("Reprojecting lidar to segmentation")
    scene_w_segclusters = reproject_lidar_to_segmentation(scene_w_segmentation,
                                                          config['reprojection'],
                                                          get_out_dir(config, '05_reprojection'))

    logger.info("Denoising object clusters")
    scene_w_segclusters_denoised = denoise_object_clusters(scene_w_segclusters,
                                                           obj_info,
                                                           config['denoising'],
                                                           get_out_dir(config, '06_denoising'))

    logger.info("Merging multicamera objects")
    scene_w_segclusters_cammerged = merge_multicamera_objects(scene_w_segclusters_denoised,
                                                              obj_info,
                                                              config['multicam_object_merge'],
                                                              get_out_dir(config, '07_multicam_object_merge'))

    logger.info("Merging vlm based and clustering based objects")
    scene_w_clusters_merged = merger_vlm_based_and_clustering_based_objects(scene_w_segclusters_cammerged,
                                                                            scene_w_velo_clusters,
                                                                            obj_info,
                                                                            config['vlm_and_cluster_merge'],
                                                                            get_out_dir(config, '08_vlm_and_cluster_merge'))

    logger.info("Calculating appearance embeddings")
    scene_w_embeddings: Scene = appearance_embed_lidar_based(scene_w_clusters_merged,
                                                             config['appearance_embedding'],
                                                             get_out_dir(config, '09_appearance_embedding'))

    if scene_w_embeddings.split == 'train':
        appearance_embeddings = [obj.appearance_embedding for frame in scene_w_embeddings.frames for obj in frame.objects]
        sam_labels = [obj.sam_label for frame in scene_w_embeddings.frames for obj in frame.objects]
    else:
        appearance_embeddings = []
        sam_labels = []

    return appearance_embeddings, sam_labels


if __name__ == "__main__":
    main()
