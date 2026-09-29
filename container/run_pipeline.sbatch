#!/bin/bash
# Worker for the auto-labeling pipeline. Don't call directly: use  bash container/submit_pipeline.sh  (it sets the env vars below).
#   DATASET  nuscenes_mini | ecp        RUN_TAG  run folder name        CONFIG  e.g. configs/nuscenes.yaml
# nuscenes_mini: array task i = scene i of autolabeling/configs/scenes_nuscenes_mini_train.txt (each scene fits the 4 h limit)
# ecp:           single task, annotated keyframes only (33 frames)
#   SHARED_RUN (optional): name of a run whose checkpoint folder is shared (SAM3/TerraSeg/PseudoLabeler/Body results are
#   reused, only SAM3D Objects is computed) -- see submit_sweep.sh.
# Every extra argument is passed on to run_pipeline.py. Progress is checkpointed per frame, so resubmitting resumes.
#SBATCH --job-name=autolabel
#SBATCH --partition=me-cor,general
#SBATCH --qos=short
#SBATCH --account=me-cor
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:a40:1
#SBATCH --output=/tudelft.net/staff-umbrella/TeamHolgerResearch/lleba/Thesis/container/logs/pipeline_%A_%a.out
#SBATCH --error=/tudelft.net/staff-umbrella/TeamHolgerResearch/lleba/Thesis/container/logs/pipeline_%A_%a.err

set -e
THESIS=/tudelft.net/staff-umbrella/TeamHolgerResearch/lleba/Thesis
: "${DATASET:?set by submit_pipeline.sh}" "${RUN_TAG:?}" "${CONFIG:?}"
OUT=/workspace/autolabeling/output/$DATASET/$RUN_TAG

case $DATASET in
  nuscenes_mini)
    SCENE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$THESIS/autolabeling/configs/scenes_nuscenes_mini_train.txt")
    [ -n "$SCENE" ] || { echo "no scene for array index $SLURM_ARRAY_TASK_ID"; exit 1; }
    ARGS=(--scenes "$SCENE" --output-dir "$OUT/scenes/$SCENE" --run-name "$SCENE")
    [ -n "$SHARED_RUN" ] && ARGS+=(--checkpoint-dir "/workspace/autolabeling/output/$DATASET/$SHARED_RUN/scenes/$SCENE/checkpoints") ;;
  ecp)
    SCENE=annotated
    ARGS=(--annotated-only --output-dir "$OUT" --run-name autolabel_ecp)
    [ -n "$SHARED_RUN" ] && ARGS+=(--checkpoint-dir "/workspace/autolabeling/output/$DATASET/$SHARED_RUN/checkpoints") ;;
  *) echo "unknown DATASET '$DATASET'"; exit 1 ;;
esac

echo "Job $SLURM_JOB_ID (array $SLURM_ARRAY_JOB_ID/$SLURM_ARRAY_TASK_ID) on $(hostname)  dataset=$DATASET  run=$RUN_TAG  scene=$SCENE  config=$CONFIG"
nvidia-smi
srun bash "$THESIS/container/run_in_container.sh" python run_pipeline.py --config "$CONFIG" "${ARGS[@]}" "$@"
