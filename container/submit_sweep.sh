#!/usr/bin/env bash
# Run the pipeline once per pointmap mode so the modes can be compared afterwards.
#
#   bash container/submit_sweep.sh <nuscenes_mini|ecp> <sweep_name> "<modes>" [extra run_pipeline.py args...]
#
#   ARRAY=0 bash container/submit_sweep.sh nuscenes_mini cmp1 "1 2 6 7 8"          # nuScenes scene index 0, all keyframes
#   bash container/submit_sweep.sh ecp cmp1 "1 2 6 7 8"                             # ECP, the 33 annotated keyframes
#
# Everything before SAM3D Objects (SAM3, TerraSeg, PseudoLabeler, SAM3D Body + B1/B2) does not depend on the mode, so it is
# computed ONCE by a prepare job into <sweep>_shared; the mode jobs (<sweep>_m<mode>) start when it has succeeded, reuse
# those results and only run SAM3D Objects. Each job therefore needs much less time (default 2 h; if one runs out, submit
# the same command again: finished frames are skipped).
#
# Options (env vars):
#   SWEEP_PARALLEL=3  how many modes run at the same time (default 1 = one after the other; =all everything at once)
#   TIME=02:00:00     time limit per job (default 02:00:00);  GRES=l40|a40  GPU type;  DRY_RUN=1  only print
#   NOPREP=1          no sharing: every mode job recomputes everything (old behaviour, needs more time per job)
#   ARRAY=<idx>       nuScenes only: run just these scene indices (order of configs/scenes_nuscenes_mini_train.txt)
#
# Compare afterwards (shared checkpoints -> use --shared-run + --modes; runs from NOPREP sweeps use --runs):
#   bash container/run_in_container.sh python compare_modes.py --dataset nuscenes_mini --scenes scene-0061 \
#        --shared-run cmp1_shared --modes 1 2 6 7 8 --baseline m2
# Modes: 1 sparse_lidar, 2 moge_affine_local, 3 masked, 4 completionformer, 5 raw, 6 piecewise, 7 composite, 8 regslope.
set -e
THESIS=/tudelft.net/staff-umbrella/TeamHolgerResearch/lleba/Thesis
DATASET=${1:?usage: submit_sweep.sh <nuscenes_mini|ecp> <sweep_name> "<modes>" [run_pipeline.py args]}
SWEEP=${2:?usage: submit_sweep.sh <nuscenes_mini|ecp> <sweep_name> "<modes>" [run_pipeline.py args]}
MODES=${3:?give the modes, e.g. "1 2 6 7 8"}
shift 3
export TIME=${TIME:-02:00:00}
SUB="bash $THESIS/container/submit_pipeline.sh"

PREPJOB=""
if [ -z "$NOPREP" ]; then
  echo "== prepare (SAM3, TerraSeg, PseudoLabeler, Body) -> run ${SWEEP}_shared"
  OUT=$(NO_MERGE=1 $SUB "$DATASET" "${SWEEP}_shared" --prepare-only "$@")
  echo "$OUT"
  PREPJOB=$(echo "$OUT" | sed -n 's/^submitted job \([0-9]*\).*/\1/p' | head -1)
fi

NPAR=${SWEEP_PARALLEL:-1}
[ "$NPAR" = all ] && NPAR=999
IDS=()
K=0
for M in $MODES; do
  PREV=""
  if [ -z "$DRY_RUN" ] && [ "$K" -ge "$NPAR" ]; then PREV=${IDS[$((K - NPAR))]}; fi
  echo "== mode $M -> run ${SWEEP}_m${M}${PREPJOB:+   (after prepare job $PREPJOB)}${PREV:+   (and after job $PREV)}"
  OUT=$(AFTEROK=$PREPJOB AFTER=$PREV SHARED=${PREPJOB:+${SWEEP}_shared} $SUB "$DATASET" "${SWEEP}_m${M}" --pointmap-mode "$M" "$@")
  echo "$OUT"
  IDS+=("$(echo "$OUT" | sed -n 's/^submitted job \([0-9]*\).*/\1/p' | head -1)")
  K=$((K + 1))
done
