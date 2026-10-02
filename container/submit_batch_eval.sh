#!/usr/bin/env bash
# Submit the mesh / free-space / ground evaluation (contribution_ideas/phase0_mesh_freespace/PLAN.md).
# Reads the full-mesh checkpoints of an already-finished pipeline run -- no pipeline stage is re-run, this is
# pure post-hoc measurement over the meshes you already have.
#
#   bash container/submit_batch_eval.sh nuscenes_mini      # 8 array tasks, one per train scene
#   bash container/submit_batch_eval.sh ecp                # 1 task, all 33 annotated keyframes
#
# Options (env vars):
#   RUN_ROOT   prepare-shared checkpoint dir (default autolabeling/output/<dataset>/all_cam_shared)
#   MODE_NAME  pointmap mode NAME, not the number (default moge_affine_local, i.e. mode 2)
#   OUT_DIR    where the CSVs go (default contribution_ideas/phase0_mesh_freespace/results)
#   ARRAY=<idx>  nuscenes_mini only: run just these scene indices (order of scenes_nuscenes_mini_train.txt)
#   DRY_RUN=1  print the sbatch command instead of submitting
#
# Output: <OUT_DIR>/<scene>.csv per nuScenes scene, <OUT_DIR>/ecp.csv for ECP -- one row per object.
set -e
THESIS=/tudelft.net/staff-umbrella/TeamHolgerResearch/lleba/Thesis
DATASET=${1:?usage: submit_batch_eval.sh <nuscenes_mini|ecp>}
RUN_ROOT=${RUN_ROOT:-$THESIS/autolabeling/output/$DATASET/all_cam_shared}
MODE_NAME=${MODE_NAME:-moge_affine_local}
OUT_DIR=${OUT_DIR:-$THESIS/contribution_ideas/phase0_mesh_freespace/results}

case $DATASET in
  nuscenes_mini) N=$(grep -c . "$THESIS/autolabeling/configs/scenes_nuscenes_mini_train.txt")
                 ARR=${ARRAY:-0-$((N - 1))} ;;
  ecp)           ARR=0 ;;
  *) echo "dataset must be nuscenes_mini or ecp"; exit 1 ;;
esac

[ -d "$RUN_ROOT" ] || { echo "RUN_ROOT does not exist: $RUN_ROOT"; exit 1; }
mkdir -p "$OUT_DIR" "$THESIS/container/logs"

run() { if [ -n "$DRY_RUN" ]; then echo "[dry run] $*"; else "$@"; fi; }

cd "$THESIS"
if [ -n "$DRY_RUN" ]; then
  echo "[dry run] sbatch --array=$ARR --export=ALL,DATASET=$DATASET,RUN_ROOT=$RUN_ROOT,MODE_NAME=$MODE_NAME,OUT_DIR=$OUT_DIR container/batch_eval.sbatch"
else
  JOB=$(sbatch --parsable --array="$ARR" \
        --export=ALL,DATASET="$DATASET",RUN_ROOT="$RUN_ROOT",MODE_NAME="$MODE_NAME",OUT_DIR="$OUT_DIR" \
        container/batch_eval.sbatch)
  echo "submitted job $JOB  (array $ARR)   logs: container/logs/mesheval_${JOB}_<index>.out   out: $OUT_DIR"
fi
