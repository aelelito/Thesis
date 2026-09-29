import argparse
from pathlib import Path
import yaml
import logging
import numpy as np
import rerun as rr
from typing import Tuple, Optional

from src.pointcloud.remove_ground import remove_ground
from src.pointcloud.spatial_clustering import get_spatial_clustering
from src.pointcloud.sceneflow import estimate_sceneflow
from src.image.appearance_embedding import appearance_embed_lidar_based
from src.data.dataset import load_dataset_scene, get_dataset_scene_names
from src.data import EXP_PSEUDO_OUT_PATH
from src.structures import Scene
from src.visualizations.rerun import get_rerun_blueprint
from src.visualizations.modules import visualize_base_scene, visualize_lidar, visualize_object_velocities
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
    assert config['type'] == 'union', f"Config file {args.config} is not a valid config file for this script"
    config['exp_name'] = args.config.stem

    scene_names_all: list[str] = get_dataset_scene_names(config['data'])
    scene_names = scene_names_all if args.scenes is None else args.scenes
    assert all([scene_name in scene_names_all for scene_name in scene_names]), \
        f"Some scenes {args.scenes} are not in the dataset"

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
