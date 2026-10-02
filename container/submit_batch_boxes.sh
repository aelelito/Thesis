#!/usr/bin/env bash
# Submit the CHEAP box-only extraction (autolabeling.batch_boxes): box geometry, score and free-space/below-ground
# for every GT box and detection OBB, no mesh rendering -- the input to nuScenes-style center-distance re-matching
# (contribution_ideas/phase0_mesh_freespace/scripts/rematch.py). Reads existing checkpoints only.
#
#   bash container/submit_batch_boxes.sh nuscenes_mini      # 8 array tasks, one per train scene
#   bash container/submit_batch_boxes.sh ecp                # 1 task
#
# Options (env vars): RUN_ROOT, MODE_NAME, OUT_DIR (default contribution_ideas/phase0_mesh_freespace/results/boxes),
#   ARRAY=<idx>, DRY_RUN=1
set -e
THESIS=/tudelft.net/staff-umbrella/TeamHolgerResearch/lleba/Thesis
DATASET=${1:?usage: submit_batch_boxes.sh <nuscenes_mini|ecp>}
RUN_ROOT=${RUN_ROOT:-$THESIS/autolabeling/output/$DATASET/all_cam_shared}
MODE_NAME=${MODE_NAME:-moge_affine_local}
OUT_DIR=${OUT_DIR:-$THESIS/contribution_ideas/phase0_mesh_freespace/results/boxes}

case $DATASET in
  nuscenes_mini) N=$(grep -c . "$THESIS/autolabeling/configs/scenes_nuscenes_mini_train.txt")
                 ARR=${ARRAY:-0-$((N - 1))} ;;
  ecp)           ARR=0 ;;
  *) echo "dataset must be nuscenes_mini or ecp"; exit 1 ;;
esac

[ -d "$RUN_ROOT" ] || { echo "RUN_ROOT does not exist: $RUN_ROOT"; exit 1; }
mkdir -p "$OUT_DIR" "$THESIS/container/logs"

cd "$THESIS"
if [ -n "$DRY_RUN" ]; then
  echo "[dry run] sbatch --array=$ARR --export=ALL,DATASET=$DATASET,RUN_ROOT=$RUN_ROOT,MODE_NAME=$MODE_NAME,OUT_DIR=$OUT_DIR container/batch_boxes.sbatch"
else
  JOB=$(sbatch --parsable --array="$ARR" \
        --export=ALL,DATASET="$DATASET",RUN_ROOT="$RUN_ROOT",MODE_NAME="$MODE_NAME",OUT_DIR="$OUT_DIR" \
        container/batch_boxes.sbatch)
  echo "submitted job $JOB  (array $ARR)   logs: container/logs/boxeval_${JOB}_<index>.out   out: $OUT_DIR"
fi
