# Option 4: Per-Gaussian Consistency Loss for Multi-View Edit Refinement

## The Problem

When fine-tuning a 3D Gaussian Splatting scene from edited training images, different views may have been edited inconsistently (e.g., by an image editing model applied independently per view). The standard single-view training loop sees one image at a time and blindly follows each view's supervision signal. Gaussians visible in multiple views get pulled in conflicting directions, leading to artifacts and color flickering.

## The Idea

Add a loss term that penalizes when a single Gaussian's appearance varies across different viewpoints, beyond what is expected from legitimate view-dependent effects (e.g., specular highlights).

Since a Gaussian's color is split into:

- **`features_dc`** (base color) — view-independent, should look the same from every angle
- **`features_rest`** (spherical harmonics) — encodes view-dependent effects like shininess

We can target the consistency loss **only at `features_dc`**, leaving view-dependent appearance untouched.

## How It Works

### Core mechanism

Each training step renders K views (e.g., 4 nearby cameras). For each Gaussian, we track what its base color contribution looks like across those views and penalize high variance.

### Online variance computation

Variance can be decomposed as:

```
Var(x) = E[x^2] - E[x]^2
```

This means we only need two running accumulators per Gaussian — no need to store per-view values:

```
Per Gaussian g, maintain:
  sum_color[g]      (running sum of rendered DC color across views)
  sum_color_sq[g]   (running sum of squared rendered DC color)
  count[g]          (number of views where g is visible)
```

After rendering each view:

```python
if gaussian g is visible in this view:
    c = rendered_dc_color(g, view)
    sum_color[g]    += c
    sum_color_sq[g] += c * c
    count[g]        += 1
```

After all K views, finalize:

```python
mean = sum_color / count
variance = sum_color_sq / count - mean * mean
L_consistency = variance[count > 1].mean()  # only for Gaussians seen by 2+ views
```

Memory cost: O(num_gaussians) — same as the parameters themselves.

### Total loss

```python
L_total = L_reconstruction + lambda_consistency * L_consistency
```

Where `lambda_consistency` is a weighting hyperparameter.

## Why It Might Work

1. **Directly targets the problem**: inconsistent edits cause the same Gaussian to produce different colors from different views. This loss explicitly penalizes that.

2. **View-dependent effects are preserved**: by only applying the loss to `features_dc`, legitimate specular highlights and reflections (encoded in `features_rest`) are unaffected.

3. **Self-correcting**: unlike gradient consensus which filters/weights gradients heuristically, this loss creates an explicit optimization pressure toward consistency. The optimizer naturally finds the best compromise color that satisfies all views while minimizing variance.

4. **No gradient storage**: unlike the original consensus approach (which stores per-view gradients for every Gaussian), this only needs two small accumulators per Gaussian, computed online.

5. **Works with any number of views**: scaling from 4 to 32 views just means more accumulator updates, no memory explosion.

## When It Might Fail

- **All views are consistently wrong**: if the editing model applied the same incorrect edit everywhere, variance is zero and this loss has nothing to penalize.
- **Lambda tuning**: too high suppresses legitimate edits, too low has no effect. Requires tuning per dataset.
- **Rasterizer access needed**: tracking per-Gaussian rendered DC color requires hooks into the rasterizer. Standard gsplat returns the final composited image, not per-Gaussian contributions.

## Implementation Sketch

### Where to modify in nerfstudio

The main changes would be in `nerfstudio/models/splatfacto.py` and `nerfstudio/engine/trainer.py`.

### Step 1: Add config options to SplatfactoModelConfig

```python
# In splatfacto.py, inside SplatfactoModelConfig

consistency_loss_enabled: bool = False
"""Enable per-Gaussian DC consistency loss across multiple views."""

consistency_loss_weight: float = 0.01
"""Weight (lambda) for the consistency loss term."""

consistency_loss_num_views: int = 4
"""Number of views to render per step for consistency measurement."""
```

### Step 2: Extract per-Gaussian DC color from the rasterizer

The gsplat rasterizer computes per-Gaussian projected colors internally. To access them, you need the DC component evaluated per Gaussian (which is just `features_dc` passed through the SH0 coefficient):

```python
# DC color is view-independent, so it's simply:
dc_color = torch.sigmoid(self.features_dc)  # or however DC is decoded in splatfacto
```

Since DC is view-independent, you don't actually need per-view rendering to get it — it's the same value regardless of view direction. The **variance across views** then comes from which Gaussians are **visible** (contribute to pixels) in each view, and how much they contribute (alpha-blending weight).

A simpler approximation: just use `features_dc` directly and measure whether the reconstruction loss from different views pushes it in different directions. This avoids rasterizer modifications entirely.

### Step 3: Online accumulation in the training loop

```python
# In trainer.py, inside the multi-view training loop

# Before view loop
num_gaussians = model.num_points
dc = model.features_dc  # [N, 3] or [N, 1, 3]
sum_dc_grad = torch.zeros_like(dc)
sum_dc_grad_sq = torch.zeros_like(dc)
vis_count = torch.zeros(num_gaussians, device=dc.device)

# Inside each view iteration (after backward)
if dc.grad is not None:
    grad = dc.grad.detach()
    visible = grad.flatten(1).norm(dim=-1) > eps
    vis_f = visible.float()

    mask = visible.unsqueeze(-1) if dc.dim() == 2 else visible.unsqueeze(-1).unsqueeze(-1)
    sum_dc_grad += grad * mask
    sum_dc_grad_sq += (grad * grad) * mask
    vis_count += vis_f

# After view loop — compute variance of gradients as consistency proxy
multi_vis = vis_count > 1
mean_grad = sum_dc_grad / vis_count.clamp_min(1).unsqueeze(-1)
var_grad = sum_dc_grad_sq / vis_count.clamp_min(1).unsqueeze(-1) - mean_grad * mean_grad

L_consistency = var_grad[multi_vis].mean()
L_total = L_reconstruction + lambda_consistency * L_consistency
```

### Step 4: Alternative — direct DC regularization (simplest)

If rasterizer-level tracking is too complex, the simplest version that achieves a weaker but similar effect:

```python
# Save once before fine-tuning
original_dc = model.features_dc.detach().clone()

# Each step
L_consistency = ((model.features_dc - original_dc) ** 2).mean()
L_total = L_reconstruction + lambda_consistency * L_consistency
```

This is just a regularizer ("don't drift from original"), not true consistency. But it acts as a conservative anchor: Gaussians only change color if the reconstruction pressure is strong and consistent across views.

## Comparison with Gradient Consensus

| Aspect | Gradient Consensus | Consistency Loss |
|--------|-------------------|------------------|
| How it works | Filters/weights per-view gradients before optimizer step | Adds a loss term penalizing cross-view variance |
| What it compares | Gradient directions | Rendered colors or gradient variance |
| Memory | O(views * gaussians) or O(gaussians) with online aggregation | O(gaussians) |
| Precision | Can fully exclude outlier views | Soft penalty, cannot fully exclude |
| Requires | Custom training loop | Loss term addition, minimal code change |
| Rasterizer changes | None | Needed for true per-Gaussian color tracking, avoidable with gradient-based proxy |

## Recommended Approach

Start with the **gradient variance proxy** (Step 3) — it measures consistency using information already available (per-view gradients of `features_dc`), requires no rasterizer changes, uses O(gaussians) memory via online accumulation, and integrates naturally into the existing multi-view training loop from this repo.

If results are promising, invest in proper per-Gaussian rasterizer hooks to measure actual rendered DC color variance for a more principled signal.
