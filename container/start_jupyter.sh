#!/usr/bin/env bash
set -e

THESIS=/tudelft.net/staff-umbrella/TeamHolgerResearch/lleba/Thesis
SIF=$THESIS/container/autolabeling.sif

PYTHONPATH_PROJECT="/workspace/models/SAM3D/sam-3d-objects:/workspace/models/SAM3D/sam-3d-objects/notebook:/workspace/autolabeling/src:/workspace/models/TerraSeg/PseudoLabeler_scripts:/workspace/models/TerraSeg/terraseg_lib/src:/workspace/models/TerraSeg/ptv3/src"

mkdir -p \
  "$THESIS/cache/huggingface" \
  "$THESIS/cache/torch" \
  "$THESIS/cache/warp"

apptainer exec --nv \
  --bind "$THESIS:/workspace" \
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
  jupyter lab \
    --no-browser \
    --ip=0.0.0.0 \
    --port=8888 \
    --ServerApp.root_dir=/workspace \
    --ServerApp.disable_check_xsrf=True \
    --ServerApp.allow_origin='*'
