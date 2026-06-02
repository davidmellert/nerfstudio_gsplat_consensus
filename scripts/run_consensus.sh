#!/bin/bash
set -e

# ============================================================
# Consensus fine-tuning (4 views, cosine filtering)
# Run with: nohup bash scripts/run_consensus.sh > consensus.log 2>&1 &
# ============================================================

DATA=/mnt/hdd/data/bicycle
RENDER_PATH=docs/assets/circle_path.json
EDITED_IMAGES=images_snow
DOWNSCALE=8
FINETUNE_ITERS=17000

# Symlink
if [ ! -e "$DATA/images_snow_8" ]; then
  ln -s images_8_snow "$DATA/images_snow_8"
fi

# Find baseline checkpoint
BASELINE_DIR=$(ls -td outputs/bicycle/splatfacto-colmap/*/ 2>/dev/null | head -1)
if [ -z "$BASELINE_DIR" ]; then
  echo "ERROR: No baseline found"; exit 1
fi
CKPT_DIR="${BASELINE_DIR}nerfstudio_models"
echo "Using baseline: $BASELINE_DIR"

echo ""
echo ">>> Consensus fine-tuning (4 views, cosine filtering)..."
ns-train splatfacto-gaussian-consensus-colmap \
  --data "$DATA" \
  --load-dir "$CKPT_DIR" \
  --max-num-iterations "$FINETUNE_ITERS" \
  --pipeline.datamanager.cache-images cpu \
  --pipeline.model.gaussian-consensus-num-views 4 \
  --pipeline.model.gaussian-consensus-max-views-per-gaussian 4 \
  --vis tensorboard \
  colmap --images-path "$EDITED_IMAGES" \
  --downscale-factor "$DOWNSCALE" \
  --skip-missing-images True \
  --auto-downscale-missing-images False \
  --eval-mode all

CONFIG=$(ls -t outputs/bicycle/splatfacto-gaussian-consensus-colmap/*/config.yml 2>/dev/null | head -1)
echo ">>> Rendering consensus fine-tuning..."
mkdir -p renders
ns-render camera-path \
  --load-config "$CONFIG" \
  --camera-path-filename "$RENDER_PATH" \
  --output-path renders/bicycle_consensus_finetune.mp4

echo ""
echo "Done at $(date)"
echo "Render: renders/bicycle_consensus_finetune.mp4"
