#!/bin/bash
set -e

# ============================================================
# Fine-tuning only: skip baseline, run Steps 2 & 3
# Assumes baseline already trained under outputs/bicycle/splatfacto-colmap/
# Run with: nohup bash scripts/run_finetune_only.sh > finetune.log 2>&1 &
# ============================================================

DATA=/mnt/hdd/data/bicycle
RENDER_PATH=docs/assets/circle_path.json
EDITED_IMAGES=images_snow       # base name; dataparser appends _8 -> images_snow_8
DOWNSCALE=8
FINETUNE_ITERS=17000

# ----------------------------------------------------------
# Symlink: the colmap dataparser with --downscale-factor 8
# rewrites images_path "images_snow" -> "images_snow_8".
# Our edited images live in images_8_snow, so we symlink:
# ----------------------------------------------------------
if [ ! -e "$DATA/images_snow_8" ]; then
  echo "Creating symlink: $DATA/images_snow_8 -> images_8_snow"
  ln -s images_8_snow "$DATA/images_snow_8"
fi

# Find the latest baseline checkpoint
BASELINE_DIR=$(ls -td outputs/bicycle/splatfacto-colmap/*/ 2>/dev/null | head -1)
if [ -z "$BASELINE_DIR" ]; then
  echo "ERROR: No baseline found in outputs/bicycle/splatfacto-colmap/"
  echo "Run the full experiment script first to train a baseline."
  exit 1
fi
CKPT_DIR="${BASELINE_DIR}nerfstudio_models"
BASELINE_CONFIG="${BASELINE_DIR}config.yml"
echo "Using baseline: $BASELINE_DIR"
echo "Checkpoint dir: $CKPT_DIR"

echo ""
echo "============================================"
echo "Starting fine-tuning experiments at $(date)"
echo "============================================"

# ----------------------------------------------------------
# Step 2: Normal single-view fine-tuning on edited images
# ----------------------------------------------------------
echo ""
echo ">>> Step 2: Normal fine-tuning (1 view, no DC reg)..."
ns-train splatfacto-gaussian-batch-colmap \
  --data "$DATA" \
  --load-dir "$CKPT_DIR" \
  --max-num-iterations "$FINETUNE_ITERS" \
  --pipeline.datamanager.cache-images cpu \
  --pipeline.model.gaussian-consensus-num-views 1 \
  --vis tensorboard \
  colmap --images-path "$EDITED_IMAGES" \
  --downscale-factor "$DOWNSCALE" \
  --skip-missing-images True \
  --auto-downscale-missing-images False \
  --eval-mode all

NORMAL_CONFIG=$(ls -t outputs/bicycle/splatfacto-gaussian-batch-colmap/*/config.yml 2>/dev/null | head -1)
echo ">>> Rendering normal fine-tuning..."
mkdir -p renders
ns-render camera-path \
  --load-config "$NORMAL_CONFIG" \
  --camera-path-filename "$RENDER_PATH" \
  --output-path renders/bicycle_normal_finetune.mp4

# ----------------------------------------------------------
# Step 3: Multi-view batch fine-tuning + DC regularization
# ----------------------------------------------------------
echo ""
echo ">>> Step 3: Fine-tuning with DC regularization (4 views)..."
ns-train splatfacto-gaussian-batch-colmap \
  --data "$DATA" \
  --load-dir "$CKPT_DIR" \
  --max-num-iterations "$FINETUNE_ITERS" \
  --pipeline.datamanager.cache-images cpu \
  --pipeline.model.gaussian-consensus-num-views 4 \
  --pipeline.model.dc-regularization-enabled True \
  --pipeline.model.dc-regularization-weight 0.01 \
  --vis tensorboard \
  colmap --images-path "$EDITED_IMAGES" \
  --downscale-factor "$DOWNSCALE" \
  --skip-missing-images True \
  --auto-downscale-missing-images False \
  --eval-mode all

DC_REG_CONFIG=$(ls -t outputs/bicycle/splatfacto-gaussian-batch-colmap/*/config.yml 2>/dev/null | head -1)
echo ">>> Rendering DC regularization fine-tuning..."
ns-render camera-path \
  --load-config "$DC_REG_CONFIG" \
  --camera-path-filename "$RENDER_PATH" \
  --output-path renders/bicycle_dc_reg_finetune.mp4

# ----------------------------------------------------------
# Done
# ----------------------------------------------------------
echo ""
echo "============================================"
echo "Fine-tuning experiments completed at $(date)"
echo "============================================"
echo ""
echo "Renders saved in renders/:"
ls -la renders/bicycle_*.mp4 2>/dev/null
