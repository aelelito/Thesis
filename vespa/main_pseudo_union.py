"""
Approach: UNION (https://arxiv.org/abs/2405.15688)
Union-based pseudo label generation pipeline using appearance clustering.

This script generates pseudo labels by combining spatial clustering with appearance-based
clustering across multiple scenes. The pipeline runs in two passes:

First pass (per scene):
1. Ground point removal from LiDAR data
2. Spatial clustering of LiDAR points
3. Appearance embedding extraction from images
4. Scene flow estimation for velocity calculation

Second pass (global clustering + labeling):
5. Cross-scene appearance clustering using embeddings and velocities
6. Prototype embedding for class assignment
7. Cluster-to-class mapping based on prototype similarity
8. Pseudo label assignment and evaluation
"""

import argparse
from pathlib import Path
import yaml
import logging
import numpy as np
import rerun as rr
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from src.pointcloud.remove_ground import remove_ground
from src.pointcloud.spatial_clustering import get_spatial_clustering
from src.pointcloud.sceneflow import estimate_sceneflow
from src.image.appearance_embedding import appearance_embed_lidar_based, embed_prototypes
from src.image.appearance_clustering import calculate_appearance_clusters_2step_velocity_based, is_appreance_cluster_calculated
from src.labeling.mapping import get_mc_cluster_to_class_by_mappings
from src.labeling.filtering import filter_objects
from src.labeling.assign_labels import assign_cluster_labels, assign_pseudo_class_labels_from_mc_cluster
from src.labeling.assign_labels import get_scene_pseudo_annotations_by_mapping
from src.labeling.submission import write_submission_files
from src.evaluate.evaluate import evaluate_submissions
from src.data.dataset import load_dataset_scene, get_dataset_scene_names
from src.data import EXP_PSEUDO_OUT_PATH
from src.structures import Scene, Annotation
from src.visualizations.rerun import get_rerun_blueprint
from src.visualizations.modules import visualize_base_scene, visualize_lidar, visualize_object_velocities
from src.visualizations.modules import visualize_object_pointclouds, visualize_object_boxes, visualize_mapped_pseudo_labeled_boxes

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
    assert config['type'] == 'union', f"Config file {args.config} is not a valid config file for this script"
    config['exp_name'] = args.config.stem

    scene_names_all: list[str] = get_dataset_scene_names(config['data'])
    scene_names = scene_names_all if args.scenes is None else args.scenes
    assert all([scene_name in scene_names_all for scene_name in scene_names]), \
        f"Some scenes {args.scenes} are not in the dataset"

    if not is_appreance_cluster_calculated(get_out_dir(config, '05_appearance_clustering')):
        logger.info("Appearance clusters not calculated yet, have to run first pass")
        appearance_embeddings: List[np.ndarray] = []
        velocities: List[np.ndarray] = []

        if args.workers > 1:
            logger.info(f"Running first pass with {args.workers} workers")
            multiprocessing.set_start_method("spawn")
            with ProcessPoolExecutor(max_workers=args.workers, initializer=_suppress_output) as executor:
                futures = {executor.submit(process_scene_first_pass, scene_name, config): scene_name \
                           for scene_name in scene_names}
                done_nr = 0
                for future in as_completed(futures):
                    done_nr += 1
                    result = future.result()
                    if result[0] is not None:
                        appearance_embeddings.append(result[0])
                        velocities.append(result[1])
                    logger.info(f"Completed scene:{futures[future]},{done_nr}/{len(futures)}")
        else:
            logger.info("Running first pass without parallelization")
            for idx, scene_name in enumerate(scene_names):
                logger.info(f"Running first pass for scene {scene_name}, {idx+1}/{len(scene_names)}")
                result = process_scene_first_pass(scene_name, config)
                if result[0] is not None:
                    appearance_embeddings.append(result[0])
                    velocities.append(result[1])
    else:
        logger.info("Appearance clusters already cached, skipping first pass")
        appearance_embeddings = []
        velocities = []

    logger.info("Running appearance clustering")
    assert is_appreance_cluster_calculated(get_out_dir(config, '05_appearance_clustering')) or len(scene_names) == len(scene_names_all), \
        f"To get appearance clusters, first pass has to be run for all scenes or appearance clusters have to be cached"
    sd_kcluster, mc_kcluster = calculate_appearance_clusters_2step_velocity_based(appearance_embeddings,
                                                                                  velocities,
                                                                                  config['appearance_clustering'],
                                                                                  get_out_dir(config, '05_appearance_clustering'))

    logger.info("Embedding prototypes")
    prototype_embeddings: Dict[str, np.ndarray] = embed_prototypes(config['appearance_embedding'], 
                                                                   get_out_dir(config, '00_appearance_embedding_prot'))

    logger.info("Assigning mc clusters to prototypes")
    mcid_to_class_by_mappings = get_mc_cluster_to_class_by_mappings(prototype_embeddings, mc_kcluster)

    logger.info("Running second pass: labeling")
    all_pseudo_annotations_by_mapping: List[Dict[str, Dict[str, List[Annotation]]]] = []
    data_split: List[str] = []
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
        scene_w_app_clusters: Scene = appearance_embed_lidar_based(scene_w_clusters,
                                                                   config['appearance_embedding'],
                                                                   get_out_dir(config, '03_appearance_embedding'))

        logger.info("Estimating scene flow")
        scene_w_velo_clusters: Scene = estimate_sceneflow(scene_w_app_clusters,
                                                          config['sceneflow'],
                                                          get_out_dir(config, '04_sceneflow'))

        logger.info("Assigning cluster labels")
        scene_w_lab_clusters = assign_cluster_labels(scene_w_velo_clusters, sd_kcluster, mc_kcluster)

        logger.info("Assigning pseudo class labels")
        scene_w_pslab_clusters = assign_pseudo_class_labels_from_mc_cluster(scene_w_lab_clusters, mcid_to_class_by_mappings)

        logger.info("Filtering objects")
        scene_w_clusters_filtered = filter_objects(scene_w_pslab_clusters, config['object_filtering'])

        logger.info("Creating pseudo annotations")
        pseudo_annotations_by_mapping = get_scene_pseudo_annotations_by_mapping(scene_w_clusters_filtered)
        all_pseudo_annotations_by_mapping.append(pseudo_annotations_by_mapping)
        data_split.append(scene_w_clusters_filtered.split)

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

            logger.info("Visualizing scene after appearance clustering")
            visualize_object_boxes(scene_w_lab_clusters, "03_sceneflow_LIDAR", "04_cluster_bbox_sd", \
                                   lambda obj: sd_kcluster.labels[obj.sd_cluster_id])
            visualize_object_boxes(scene_w_lab_clusters, "03_sceneflow_LIDAR", "04_cluster_bbox_mc", \
                                   lambda obj: obj.mc_cluster_id if obj.mc_cluster_id is not None else -1)

            logger.info("Visualizing scene after pseudo class labeling")
            visualize_mapped_pseudo_labeled_boxes(scene_w_pslab_clusters, "03_sceneflow_LIDAR")

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


def process_scene_first_pass(scene_name: str, config: dict) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    logger.info(f"##### Running first pass for scene {scene_name} #####")

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

    logger.info("Computing appearance embeddings")
    scene_w_app_clusters: Scene = appearance_embed_lidar_based(scene_w_clusters,
                                                               config['appearance_embedding'],
                                                               get_out_dir(config, '03_appearance_embedding'))

    logger.info("Estimating scene flow")
    scene_w_velo_clusters: Scene = estimate_sceneflow(scene_w_app_clusters,
                                                      config['sceneflow'],
                                                      get_out_dir(config, '04_sceneflow'))

    if scene_w_velo_clusters.split == 'train':
        appearance_embeddings = np.stack([obj.appearance_embedding for frame in scene_w_velo_clusters.frames \
                                          for obj in frame.objects], axis=0)
        velocities = np.array([obj.velocity_magnitude if obj.velocity is not None else 0.0 for frame in scene_w_velo_clusters.frames \
                               for obj in frame.objects])
    else:
        appearance_embeddings = None
        velocities = None

    return appearance_embeddings, velocities


def get_out_dir(config: dict, modulename: str) -> Path:
    out_dir = EXP_PSEUDO_OUT_PATH / config['type'] / config['exp_name'] / modulename
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _suppress_output():
    import os
    import sys
    devnull = open(os.devnull, 'w')
    sys.stdout = devnull
    sys.stderr = devnull
    os.dup2(devnull.fileno(), 1)  # Redirects fd=1 (stdout)
    os.dup2(devnull.fileno(), 2)  # Redirects fd=2 (stderr)


if __name__ == "__main__":
    main()
