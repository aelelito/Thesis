#!/bin/bash
# Worker for the mesh/free-space/ground evaluation (contribution_ideas/phase0_mesh_freespace). Don't call directly:
# use  bash container/submit_batch_eval.sh  (it sets the env vars below). Reads existing full-mesh checkpoints only
# -- no pipeline stage (SAM3/TerraSeg/PseudoLabeler/SAM3D Body/Objects) is re-run.
#   DATASET nuscenes_mini|ecp   RUN_ROOT the prepare-shared dir (e.g. .../all_cam_shared)
#   MODE_NAME pointmap mode name (e.g. moge_affine_local)   OUT_DIR where the per-scene/dataset CSV goes
# nuscenes_mini: array task i = scene i of autolabeling/configs/scenes_nuscenes_mini_train.txt.
# ecp: single task, all annotated keyframes in one CSV.
#SBATCH --job-name=mesheval
#SBATCH --partition=me-cor,general
#SBATCH --qos=short
#SBATCH --account=me-cor
#SBATCH --time=02:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a40:1
#SBATCH --output=/tudelft.net/staff-umbrella/TeamHolgerResearch/lleba/Thesis/container/logs/mesheval_%A_%a.out
#SBATCH --error=/tudelft.net/staff-umbrella/TeamHolgerResearch/lleba/Thesis/container/logs/mesheval_%A_%a.err

set -e
THESIS=/tudelft.net/staff-umbrella/TeamHolgerResearch/lleba/Thesis
: "${DATASET:?set by submit_batch_eval.sh}" "${RUN_ROOT:?}" "${MODE_NAME:?}" "${OUT_DIR:?}"

case $DATASET in
  nuscenes_mini)
    SCENE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$THESIS/autolabeling/configs/scenes_nuscenes_mini_train.txt")
    [ -n "$SCENE" ] || { echo "no scene for array index $SLURM_ARRAY_TASK_ID"; exit 1; }
    ARGS=(--scene "$SCENE" --out "$OUT_DIR/${SCENE}.csv")
    ;;
  ecp)
    SCENE=all
    ARGS=(--out "$OUT_DIR/ecp.csv")
    ;;
  *) echo "unknown DATASET '$DATASET'"; exit 1 ;;
esac

echo "Job $SLURM_JOB_ID (array $SLURM_ARRAY_JOB_ID/$SLURM_ARRAY_TASK_ID) on $(hostname)  dataset=$DATASET  run_root=$RUN_ROOT  scene=$SCENE  mode=$MODE_NAME"
nvidia-smi
srun bash "$THESIS/container/run_in_container.sh" python -m autolabeling.batch_eval \
  --dataset "$DATASET" --run-root "$RUN_ROOT" --mode-name "$MODE_NAME" --device cuda "${ARGS[@]}"
