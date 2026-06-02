# Experiment Configurations

This document describes the four fine-tuning strategies for updating a pre-trained 3D Gaussian Splatting scene with edited training images. All experiments start from the same baseline checkpoint and fine-tune on the same set of edited images, so the only variable is the training strategy.

## Prerequisites

Train a baseline scene first and set the variables:

```bash
DATA=/path/to/your/data
CKPT_DIR=outputs/<your_scene>/splatfacto-colmap/<timestamp>/nerfstudio_models
```

See [setup_and_training.md](setup_and_training.md) for full setup instructions.

## Overview

| Experiment | Method | Views/Step | Densification | Consensus Filtering | What It Tests |
|------------|--------|------------|---------------|---------------------|---------------|
| 1. Single-View Batch | `splatfacto-gaussian-batch-colmap` | 1 | No | No | Baseline: standard single-view fine-tuning |
| 2. Multi-View Batch | `splatfacto-gaussian-batch-colmap` | 4 | No | No | Does seeing more views per step help? |
| 3. Consensus | `splatfacto-gaussian-consensus-colmap` | 4 | No | Yes (cosine) | Does per-Gaussian agreement filtering help? |
| 4. Batch + Densification | `splatfacto-gaussian-batch-densify-colmap` | 4 | Yes | No | Does adding/removing Gaussians help? |

## Experiment 1: Single-View Batch (Baseline)

Standard fine-tuning. One edited view per step, no multi-view consensus, no densification. This is the simplest approach and serves as the baseline to compare against.

**How it works:** Each step picks one edited image, renders the scene from that camera, computes the loss, and updates all trainable Gaussian parameters.

```bash
ns-train splatfacto-gaussian-batch-colmap \
  --data "$DATA" \
  --downscale-factor 8 \
  --load-dir "$CKPT_DIR" \
  --max-num-iterations 17000 \
  --pipeline.datamanager.cache-images cpu \
  --pipeline.model.gaussian-consensus-num-views 1 \
  colmap --images-path images_edited \
  --skip-missing-images True \
  --auto-downscale-missing-images False \
  --eval-mode all
```

**Expected behavior:** Fast per-step, but each update is biased by a single viewpoint. Inconsistencies between edited views can cause flickering or color artifacts when the scene is rendered from novel viewpoints.

## Experiment 2: Multi-View Batch (4 Views)

Renders 4 views per step and averages their gradients. No per-Gaussian filtering — just a straightforward mini-batch over views.

**How it works:** Each step picks an anchor view plus 3 nearby cameras (pose-neighborhood sampling). All 4 views are rendered sequentially, their losses are averaged, and one optimizer step is taken. Every Gaussian gets the averaged gradient regardless of visibility.

```bash
ns-train splatfacto-gaussian-batch-colmap \
  --data "$DATA" \
  --downscale-factor 8 \
  --load-dir "$CKPT_DIR" \
  --max-num-iterations 17000 \
  --pipeline.datamanager.cache-images cpu \
  --pipeline.model.gaussian-consensus-num-views 4 \
  colmap --images-path images_edited \
  --skip-missing-images True \
  --auto-downscale-missing-images False \
  --eval-mode all
```

**Expected behavior:** Smoother updates than single-view because the gradient signal is less noisy. 4x slower per step since 4 views are rendered. Conflicting views still partially cancel each other in the gradient, which may lead to muted or compromised color updates.

## Experiment 3: Consensus (4 Views)

The main contribution. Renders 4 views per step, but instead of naively averaging, applies per-Gaussian soft consensus: only updates a Gaussian when the views that can see it agree on the gradient direction.

**How it works:**
1. Render 4 nearby views and compute per-view gradients for each Gaussian
2. Filter by visibility: only consider views where the Gaussian has a nonzero gradient (proxy for "this view can see this Gaussian")
3. Compute the mean gradient across visible views
4. For each view, compute cosine similarity between its gradient and the mean
5. Weight each view's contribution by its cosine similarity (views that disagree get zero weight)
6. Apply the weighted consensus gradient

```bash
ns-train splatfacto-gaussian-consensus-colmap \
  --data "$DATA" \
  --downscale-factor 8 \
  --load-dir "$CKPT_DIR" \
  --max-num-iterations 17000 \
  --pipeline.datamanager.cache-images cpu \
  --pipeline.model.gaussian-consensus-num-views 4 \
  --pipeline.model.gaussian-consensus-max-views-per-gaussian 4 \
  colmap --images-path images_edited \
  --skip-missing-images True \
  --auto-downscale-missing-images False \
  --eval-mode all
```

**Expected behavior:** Cleaner updates than plain batching because conflicting views are filtered out per Gaussian. May be more conservative (slower to converge) since some updates get suppressed. Best improvement expected for Gaussians at view boundaries where edits are inconsistent.

**Variant — 8 views:** To test whether more views improve consensus quality:

```bash
ns-train splatfacto-gaussian-consensus-colmap \
  --data "$DATA" \
  --downscale-factor 8 \
  --load-dir "$CKPT_DIR" \
  --max-num-iterations 16000 \
  --pipeline.datamanager.cache-images cpu \
  --pipeline.model.gaussian-consensus-num-views 8 \
  --pipeline.model.gaussian-consensus-max-views-per-gaussian 8 \
  colmap --images-path images_edited \
  --skip-missing-images True \
  --auto-downscale-missing-images False \
  --eval-mode all
```

## Experiment 4: Batch + Densification

Multi-view batch training (no consensus filtering) but with Gaussian densification enabled — Gaussians can be split, cloned, or pruned during fine-tuning.

**How it works:** Same as Experiment 2 (averaged multi-view gradients), but the gsplat densification strategy is active. Gaussians with large gradients get split into smaller ones, low-opacity Gaussians get pruned, and the total Gaussian count can grow. This allows the scene to add geometric detail to better represent the edit.

```bash
ns-train splatfacto-gaussian-batch-densify-colmap \
  --data "$DATA" \
  --downscale-factor 8 \
  --load-dir "$CKPT_DIR" \
  --max-num-iterations 17000 \
  --pipeline.datamanager.cache-images cpu \
  --pipeline.model.gaussian-consensus-num-views 4 \
  --pipeline.model.stop-split-at 20000 \
  colmap --images-path images_edited \
  --skip-missing-images True \
  --auto-downscale-missing-images False \
  --eval-mode all
```

**Expected behavior:** Can represent edits that require new geometry (e.g., adding objects, changing shapes) since new Gaussians can be created. Risk of overfitting to individual views since densification responds to per-view gradient magnitude without consensus. May increase total Gaussian count significantly.

## Rendering Results

After each experiment, render a video for comparison:

```bash
CONFIG=outputs/<your_scene>/<method>/<timestamp>/config.yml
CAMERA_PATH=docs/assets/circle_path.json

ns-render camera-path \
  --load-config "$CONFIG" \
  --camera-path-filename "$CAMERA_PATH" \
  --output-path renders/my_experiment.mp4
```

A default `circle_path.json` is included in the repo at `docs/assets/`. You can also create your own camera paths in the nerfstudio web viewer. Use the same camera path for all experiments so the videos are directly comparable.

## What to Look For

When comparing the rendered videos:

- **Color consistency**: Does the edit look the same from all angles? (Consensus should win here)
- **Artifacts/flickering**: Are there visible inconsistencies when the camera moves? (Single-view baseline likely worst)
- **Edit fidelity**: How closely does the render match the intended edit? (Batch + densification may be best for structural edits)
- **Over-smoothing**: Does the consensus filter suppress too much, making the edit look washed out?

## Diagnostic Report

For consensus experiments, generate a diagnostic report to check if the consensus filtering is actually doing something useful:

```bash
python scripts/consensus_metrics_report.py \
  outputs/<your_scene>/splatfacto-gaussian-consensus-colmap/<timestamp>/
```

This produces plots showing visibility, active fraction, and update fraction. See the script docstring for interpretation.
