#!/bin/bash
set -e

# ============================================================
# Consensus fine-tuning with agreement score heatmaps
# Run with: nohup bash scripts/run_consensus_heatmap.sh > consensus_heatmap.log 2>&1 &
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

# ----------------------------------------------------------
# Train consensus with agreement score accumulation
# ----------------------------------------------------------
echo ""
echo ">>> Consensus fine-tuning (4 views, cosine filtering, agreement scores)..."
ns-train splatfacto-gaussian-consensus-colmap \
  --data "$DATA" \
  --load-dir "$CKPT_DIR" \
  --max-num-iterations "$FINETUNE_ITERS" \
  --pipeline.datamanager.cache-images cpu \
  --pipeline.model.gaussian-consensus-num-views 4 \
  --pipeline.model.gaussian-consensus-max-views-per-gaussian 4 \
  --pipeline.model.store-agreement-scores True \
  --vis tensorboard \
  colmap --images-path "$EDITED_IMAGES" \
  --downscale-factor "$DOWNSCALE" \
  --skip-missing-images True \
  --auto-downscale-missing-images False \
  --eval-mode all

CONFIG=$(ls -t outputs/bicycle/splatfacto-gaussian-consensus-colmap/*/config.yml 2>/dev/null | head -1)
if [ -z "$CONFIG" ]; then
  echo "ERROR: No config found after training"; exit 1
fi
echo "Using config: $CONFIG"

# ----------------------------------------------------------
# Render normal video
# ----------------------------------------------------------
echo ""
echo ">>> Rendering normal consensus video..."
mkdir -p renders
ns-render camera-path \
  --load-config "$CONFIG" \
  --camera-path-filename "$RENDER_PATH" \
  --output-path renders/bicycle_consensus_finetune.mp4

# ----------------------------------------------------------
# Render agreement heatmaps
# ----------------------------------------------------------
echo ""
echo ">>> Rendering agreement heatmap: features_dc..."
python scripts/render_agreement_heatmap.py \
  --load-config "$CONFIG" \
  --camera-path-filename "$RENDER_PATH" \
  --param-group features_dc \
  --output-path renders/bicycle_heatmap_features_dc.mp4

echo ""
echo ">>> Rendering agreement heatmap: features_rest..."
python scripts/render_agreement_heatmap.py \
  --load-config "$CONFIG" \
  --camera-path-filename "$RENDER_PATH" \
  --param-group features_rest \
  --output-path renders/bicycle_heatmap_features_rest.mp4

echo ""
echo ">>> Rendering agreement heatmap: geometry..."
python scripts/render_agreement_heatmap.py \
  --load-config "$CONFIG" \
  --camera-path-filename "$RENDER_PATH" \
  --param-group geometry \
  --output-path renders/bicycle_heatmap_geometry.mp4

# ----------------------------------------------------------
# Done
# ----------------------------------------------------------
echo ""
echo "============================================"
echo "Done at $(date)"
echo "============================================"
echo ""
echo "Renders:"
ls -la renders/bicycle_consensus_finetune.mp4 renders/bicycle_heatmap_*.mp4 2>/dev/null
