#!/usr/bin/env bash
# Submit an auto-labeling run.
#
#   bash container/submit_pipeline.sh <nuscenes_mini|ecp> <run_name> [extra run_pipeline.py args...]
#
#   bash container/submit_pipeline.sh nuscenes_mini front_cam_o1      # 10 scene jobs (max 4 at once) + a merge job that
#                                                                      # writes ONE json per mapping when they are all done
#   bash container/submit_pipeline.sh ecp ecp_test                     # one job, annotated keyframes only
#
# Results: autolabeling/output/<dataset>/<run_name>/autolabel_<dataset>_{8class,3class,1class}.json  (nuscenes_mini also keeps scenes/)
# Options (env vars):  ARRAY=2  or  ARRAY=0,3  run only those scenes (no merge);  DRY_RUN=1  print instead of submitting;
#   AFTER=<jobid>  start after that job has ended;  AFTEROK=<jobid>  start only if it succeeded;
#   SHARED=<run>  reuse that run's checkpoint folder (SAM3/TerraSeg/PseudoLabeler/Body cached there); NO_MERGE=1  no merge job;  GRES=l40|v100|a40  GPU type (default a40, see run_pipeline.sbatch);
#   TIME=36:00:00 QOS=medium  raise the per-job limit (needed with all 6 cameras; qos short = 4 h, medium = 36 h, long = 7 days).
# Resubmitting the same run name resumes from checkpoints. Settings (pointmap mode, aggregation, ...) come from
# autolabeling/configs/{nuscenes,ecp}.yaml; a smoke test: `ARRAY=1 bash container/submit_pipeline.sh nuscenes_mini smoke --frame-end 1`.
set -e
THESIS=/tudelft.net/staff-umbrella/TeamHolgerResearch/lleba/Thesis
DATASET=${1:?usage: submit_pipeline.sh <nuscenes_mini|ecp> <run_name> [run_pipeline.py args]}
RUN=${2:?usage: submit_pipeline.sh <nuscenes_mini|ecp> <run_name> [run_pipeline.py args]}
shift 2

case $DATASET in
  nuscenes_mini) CONFIG=configs/nuscenes.yaml
                 N=$(grep -c . "$THESIS/autolabeling/configs/scenes_nuscenes_mini_train.txt")
                 ARR=${ARRAY:-0-$((N - 1))%4} ;;
  ecp)           CONFIG=configs/ecp.yaml; ARR=0 ;;
  *) echo "dataset must be nuscenes_mini or ecp"; exit 1 ;;
esac
RUNDIR=autolabeling/output/$DATASET/$RUN
[ -d "$THESIS/$RUNDIR" ] && echo "note: $RUNDIR exists -> resuming / extending it"
mkdir -p "$THESIS/container/logs"

run() { if [ -n "$DRY_RUN" ]; then echo "[dry run] $*"; else "$@"; fi; }

# dependencies: AFTEROK=<job> (must have succeeded), AFTER=<job> (must have ended, whatever the outcome)
DEP=""
[ -n "$AFTEROK" ] && DEP="afterok:$AFTEROK"
[ -n "$AFTER" ] && DEP="${DEP:+$DEP,}afterany:$AFTER"

cd "$THESIS"
if [ -n "$DRY_RUN" ]; then
  echo "[dry run] sbatch --array=$ARR ${QOS:+--qos=$QOS} ${TIME:+--time=$TIME} ${DEP:+--dependency=$DEP} ${GRES:+--gres=gpu:$GRES:1} --export=ALL,DATASET=$DATASET,RUN_TAG=$RUN,CONFIG=$CONFIG,SHARED_RUN=$SHARED container/run_pipeline.sbatch $*"
  JOB=DRYRUN
else
  JOB=$(sbatch --parsable --array="$ARR" ${QOS:+--qos=$QOS} ${TIME:+--time=$TIME} ${DEP:+--dependency=$DEP} ${GRES:+--gres=gpu:$GRES:1} \
        --export=ALL,DATASET="$DATASET",RUN_TAG="$RUN",CONFIG="$CONFIG",SHARED_RUN="$SHARED" container/run_pipeline.sbatch "$@")
  echo "submitted job $JOB  (array $ARR)   logs: container/logs/pipeline_${JOB}_<index>.out"
fi

if [ "$DATASET" = nuscenes_mini ] && [ -z "$ARRAY" ] && [ -z "$NO_MERGE" ]; then
  run sbatch --dependency="afterok:$JOB" --kill-on-invalid-dep=yes --job-name=autolabel-merge \
        --partition=me-cor,general --qos=short --account=me-cor --time=00:20:00 --mem=8G --cpus-per-task=1 \
        --output="$THESIS/container/logs/merge_${JOB}.out" --error="$THESIS/container/logs/merge_${JOB}.err" \
        --wrap "bash $THESIS/container/run_in_container.sh python merge_submissions.py --run-dir /workspace/autolabeling/output/$DATASET/$RUN"
  echo "merge job queued: runs after all scenes succeed -> $RUNDIR/autolabel_nuscenes_mini_{8class,3class,1class}.json"
elif [ "$DATASET" = nuscenes_mini ] && [ -z "$NO_MERGE" ]; then
  echo "partial run (ARRAY=$ARRAY): no merge queued; merge later with: bash container/run_in_container.sh python merge_submissions.py --run-dir /workspace/autolabeling/output/$DATASET/$RUN --allow-partial"
fi
