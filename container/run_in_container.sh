#!/usr/bin/env bash
# Run any command inside the AutoLabeling container with the same environment as start_jupyter.sh.
# Usage:  bash container/run_in_container.sh <command...>
#   e.g.  bash container/run_in_container.sh python run_pipeline.py --config configs/nuscenes.yaml \
#             --scenes scene-0103 --frame-start 0 --frame-end 3 --run-name smoke_test
# Working directory inside the container is /workspace/autolabeling (override with WORKDIR=...).
# Keep PYTHONPATH_PROJECT in sync with start_jupyter.sh.
set -e

THESIS=/tudelft.net/staff-umbrella/TeamHolgerResearch/lleba/Thesis
SIF=$THESIS/container/autolabeling.sif

PYTHONPATH_PROJECT="/workspace/models/SAM3D/sam-3d-objects:/workspace/models/SAM3D/sam-3d-objects/notebook:/workspace/autolabeling/src:/workspace/models/TerraSeg/PseudoLabeler_scripts:/workspace/models/TerraSeg/terraseg_lib/src:/workspace/models/TerraSeg/ptv3/src"

mkdir -p "$THESIS/cache/huggingface" "$THESIS/cache/torch" "$THESIS/cache/warp" "$THESIS/autolabeling/output"

exec apptainer exec --nv \
  --bind "$THESIS:/workspace" \
  --pwd "${WORKDIR:-/workspace/autolabeling}" \
  --env PYTHONPATH="$PYTHONPATH_PROJECT" \
  --env LIDRA_SKIP_INIT=true \
  --env CONDA_PREFIX=/opt/conda \
  --env HF_HOME=/workspace/cache/huggingface \
  --env TORCH_HOME=/workspace/cache/torch \
  --env WARP_CACHE_PATH=/workspace/cache/warp \
  --env FVCORE_CACHE=/workspace/cache/fvcore \
  --env SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
  --env SSL_CERT_DIR=/etc/ssl/certs \
  "$SIF" \
  "$@"
