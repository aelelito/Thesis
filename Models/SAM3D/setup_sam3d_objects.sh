#!/usr/bin/env bash
# Setup script for SAM 3D Objects
#
# Prerequisites:
#   1. Request HF access at: https://huggingface.co/facebook/sam-3d-objects
#   2. huggingface-cli login (already done as lebaleander)
#   3. Run this script once access is approved
#
# Usage:
#   bash setup_sam3d_objects.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
SAM3D_DIR="$REPO_DIR/sam-3d-objects"

# ── 1. Clone the GitHub repo ────────────────────────────────────────────────
echo "==> Cloning facebookresearch/sam-3d-objects..."
if [ -d "$SAM3D_DIR" ]; then
    echo "    Already cloned at $SAM3D_DIR, skipping."
else
    git clone https://github.com/facebookresearch/sam-3d-objects.git "$SAM3D_DIR"
fi

# ── 2. Run Meta's own setup.sh (creates conda env 'sam3d') ──────────────────
echo "==> Running sam-3d-objects/setup.sh..."
cd "$SAM3D_DIR"
bash setup.sh

# ── 3. Download checkpoints from HuggingFace ────────────────────────────────
echo "==> Downloading checkpoints from facebook/sam-3d-objects..."
conda run -n sam3d python - <<'PYEOF'
from huggingface_hub import snapshot_download
import os

local_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "hf")
os.makedirs(local_dir, exist_ok=True)

snapshot_download(
    repo_id="facebook/sam-3d-objects",
    local_dir=local_dir,
    ignore_patterns=["*.md", "*.png", "*.gif", "doc/*"],
)
print(f"Checkpoints saved to: {local_dir}")
PYEOF

echo ""
echo "==> Setup complete!"
echo "    Conda env: sam3d"
echo "    Repo:      $SAM3D_DIR"
echo "    Ckpts:     $SAM3D_DIR/checkpoints/hf/"
