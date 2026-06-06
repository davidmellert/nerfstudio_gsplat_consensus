# Agreement Heatmap Visualization

Render a flyaround video where each Gaussian is colored by how much views agreed on its gradient direction during consensus training. This shows **where** in the scene the edited views are inconsistent.

## Overview

- During consensus training, accumulate a per-Gaussian running average of cosine agreement scores
- After training, replace Gaussian colors with a red-green heatmap (red = views disagreed, green = views agreed)
- Render heatmap videos using the same camera path as normal renders for side-by-side comparison
- Produce separate heatmaps for all 6 parameter groups: `features_dc`, `features_rest`, `means`, `scales`, `quats`, `opacities`

## What is the agreement score?

In the consensus step (`trainer.py:592-594`), each Gaussian gets a cosine similarity between each view's gradient and the mean gradient:

```
cos(g_view, g_mean) ∈ [-1, 1]
```

Values below `min_alignment` (default 0.0) are clamped to zero. The **agreement score** for a Gaussian at one training step is the average cosine similarity across its visible views. Over training, we accumulate a running mean of this per-step score.

- Score near **1.0**: all views consistently push this Gaussian in the same direction (green)
- Score near **0.0**: views frequently disagree or cancel out (red)
- Score **in between**: partial agreement (yellow)

## Implementation Plan

### Step 1: Accumulate agreement scores during training

#### Diagnostic gradients for non-consensus groups

Currently, consensus only trains `features_dc` and `features_rest`. The other groups (`means`, `scales`, `quats`, `opacities`) have gradients disabled via `_only_consensus_groups_require_grad`. To compute agreement scores for these groups **without actually training them**, we:

1. Temporarily enable `requires_grad` for all parameter groups during the multi-view forward passes
2. Collect per-view gradients for all groups (not just the consensus-trained ones)
3. Compute cosine agreement for all groups
4. **Only apply** the consensus gradient update to `features_dc` and `features_rest` (as before)
5. Discard the gradients for the non-trained groups — they're only used for diagnostics

This answers the question: "if we trained geometry with consensus, how much would views disagree?" — without actually changing the geometry.

#### Agreement score accumulation

**File: `nerfstudio/engine/trainer.py`**

In `_aggregate_gaussian_consensus_group()`, after computing the cosine `alignment` tensor (line 592-594):

1. Compute per-Gaussian mean alignment across visible views:
   ```python
   # alignment shape: [num_views, num_gaussians_in_chunk]
   # mean_weights shape: [num_views, num_gaussians_in_chunk] (visibility mask)
   per_gaussian_agreement = (alignment * mean_weights).sum(dim=0) / visible_counts.clamp_min(1.0)
   ```

2. Store in running accumulators on the model:
   ```python
   model._agreement_sum[group_name][start:end] += per_gaussian_agreement.cpu()
   model._agreement_count[group_name][start:end] += (visible_counts > 0).float().cpu()
   ```

In `gaussian_consensus_train_iteration()`:

3. Collect diagnostic gradients for non-trained groups:
   ```python
   # All groups for agreement diagnostics
   all_groups = ["means", "scales", "quats", "opacities", "features_dc", "features_rest"]
   diagnostic_only = [g for g in all_groups if g not in trainable_groups]

   # Enable grads for all groups during forward passes
   # Collect view_grads for all groups
   # After consensus computation:
   #   - Apply consensus grad only for trainable_groups (as before)
   #   - Compute agreement scores for all groups (including diagnostic_only)
   #   - Discard grads for diagnostic_only groups
   ```

**File: `nerfstudio/models/splatfacto.py`**

Add config field:
```python
store_agreement_scores: bool = False
"""Accumulate per-Gaussian consensus agreement scores for heatmap visualization."""
```

In `populate_modules()`, initialize accumulators:
```python
if self.config.store_agreement_scores:
    self._agreement_sum = {}
    self._agreement_count = {}
    # Initialized after gauss_params are created, keyed by param group name
    # Includes all 6 groups, not just the consensus-trained ones
```

### Step 2: Save agreement scores with checkpoint

**File: `nerfstudio/engine/trainer.py`**

After training completes, save the agreement maps alongside the checkpoint:
```python
if hasattr(model, "_agreement_sum"):
    agreement_data = {}
    for group_name in model._agreement_sum:
        count = model._agreement_count[group_name].clamp_min(1.0)
        agreement_data[group_name] = model._agreement_sum[group_name] / count
    torch.save(agreement_data, ckpt_dir / "agreement_scores.pt")
```

This produces a dict like:
```python
{
    "features_dc": tensor([0.82, 0.15, 0.93, ...]),    # per-Gaussian, shape [N]
    "features_rest": tensor([0.71, 0.03, 0.88, ...]),   # per-Gaussian, shape [N]
    "geometry": tensor([0.83, 0.62, 0.51, ...]),        # per-Gaussian, shape [N] — avg of means, scales, quats, opacities
}
```

### Step 3: Render heatmap video

**New file: `scripts/render_agreement_heatmap.py`**

This script:
1. Loads a consensus checkpoint and `agreement_scores.pt`
2. Maps scores to colors: 0.0 → red `[1,0,0]`, 1.0 → green `[0,1,0]`, linear interpolation
3. Temporarily replaces `features_dc` with heatmap colors and zeros out `features_rest` (disable view-dependent effects so the heatmap is clean)
4. Renders using the standard gsplat rasterizer with the same camera path
5. Restores original Gaussian parameters

```
# Color parameters (consensus-trained)
python scripts/render_agreement_heatmap.py \
  --load-config outputs/.../config.yml \
  --camera-path-filename docs/assets/circle_path.json \
  --param-group features_dc \
  --output-path renders/bicycle_heatmap_features_dc.mp4

python scripts/render_agreement_heatmap.py \
  --load-config outputs/.../config.yml \
  --camera-path-filename docs/assets/circle_path.json \
  --param-group features_rest \
  --output-path renders/bicycle_heatmap_features_rest.mp4

# Geometry parameters combined (diagnostic only — not trained, but agreement computed)
python scripts/render_agreement_heatmap.py \
  --load-config outputs/.../config.yml \
  --camera-path-filename docs/assets/circle_path.json \
  --param-group geometry \
  --output-path renders/bicycle_heatmap_geometry.mp4
```

### Step 4: Training script

**File: `scripts/run_consensus.sh`** (update)

Add the flag and heatmap rendering after training:
```bash
ns-train splatfacto-gaussian-consensus-colmap \
  ...
  --pipeline.model.store-agreement-scores True \
  ...

# Render normal video
ns-render camera-path ...

# Render heatmaps
python scripts/render_agreement_heatmap.py \
  --load-config "$CONFIG" \
  --camera-path-filename "$RENDER_PATH" \
  --param-group features_dc \
  --output-path renders/bicycle_consensus_heatmap_dc.mp4

python scripts/render_agreement_heatmap.py \
  --load-config "$CONFIG" \
  --camera-path-filename "$RENDER_PATH" \
  --param-group features_rest \
  --output-path renders/bicycle_consensus_heatmap_rest.mp4
```

## Output

Four videos with identical camera paths:
| Video | Group | Trained? | Shows |
|-------|-------|----------|-------|
| `bicycle_consensus_finetune.mp4` | — | — | Normal rendered scene |
| `bicycle_heatmap_features_dc.mp4` | `features_dc` | Yes | Base color agreement |
| `bicycle_heatmap_features_rest.mp4` | `features_rest` | Yes | View-dependent SH agreement |
| `bicycle_heatmap_geometry.mp4` | `means+scales+quats+opacities` | No (diagnostic) | Combined geometry agreement |

All heatmaps use: red = views disagree, yellow = partial, green = views agree.

The geometry heatmap averages the agreement scores of `means`, `scales`, `quats`, and `opacities` per Gaussian into a single score, giving an overall picture of geometric disagreement.

## Expected Insights

**Color groups (trained):**
- **Uniform green**: edits are consistent across views, consensus isn't doing much (batch-like)
- **Red regions at object boundaries**: views disagree on edge Gaussians, consensus protects them
- **Red regions on edited objects**: the image edits themselves are inconsistent across views (e.g., snow texture differs per view)
- **Green center, red edges**: consensus helps most at view boundaries, which is the expected pattern

**Geometry (diagnostic, combined):**
- Red regions show where views disagree on how geometry should change (position, size, rotation, opacity)
- Indicates the edit implies conflicting geometric adjustments — e.g., one view wants to move/resize a Gaussian differently than another
- Comparing geometry heatmap vs color heatmaps reveals whether disagreement is primarily about color or about structure

## Performance Note

Enabling diagnostic gradients for all 6 groups increases memory and compute per step compared to the default (2 groups only). The extra cost comes from:
- Computing gradients for 4 additional parameter groups per view
- Storing per-view gradient tensors for `means` (3 values), `scales` (3), `quats` (4), `opacities` (1) — 11 extra floats per Gaussian per view
- Running cosine similarity on 6 groups instead of 2

For ~1M Gaussians and 4 views this adds roughly 170MB of temporary gradient storage. Should fit on an RTX 2080 Ti (11GB) but worth monitoring.

## Dependencies

- Requires retraining the consensus experiment with `--pipeline.model.store-agreement-scores True`
- `matplotlib` for colormap (already a dependency)
- Standard gsplat rendering pipeline for heatmap video
