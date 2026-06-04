"""Utilities for Gaussian consensus visualization snapshots."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch


def _stack_update_proxies(
    view_grads: List[torch.Tensor],
    learning_rate: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Stack per-view optimizer-update proxies for one Gaussian parameter group."""
    return torch.stack(
        [-float(learning_rate) * grad.to(device=device, dtype=dtype, non_blocking=True) for grad in view_grads],
        dim=0,
    )


def compute_consensus_visualization_data(
    view_grads: Dict[str, List[torch.Tensor]],
    final_grads: Dict[str, torch.Tensor],
    learning_rates: Dict[str, float],
    eps: float,
    max_views: int = 0,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Compute compact per-Gaussian diagnostics for one consensus step.

    Args:
        view_grads: Raw per-view gradients keyed by Gaussian parameter group.
        final_grads: Final aggregated consensus gradients keyed by Gaussian parameter group.
        learning_rates: Learning rate for each group. Updates are visualized as ``-lr * grad``.
        eps: Numerical threshold for active/visible updates.
        max_views: Optional cap on contributing views, using the largest aggregate update norms.

    Returns:
        A tuple ``(per_gaussian, scalar_attributes, rgb_attributes)``. ``per_gaussian`` contains compact
        arrays for the Streamlit app; ``scalar_attributes`` and ``rgb_attributes`` can be rendered back into
        image space by the Gaussian model.
    """
    groups = [group for group in view_grads.keys() if group in final_grads and len(view_grads[group]) > 0]
    if len(groups) == 0:
        return {}, {}, {}

    first_group = groups[0]
    first_final = final_grads[first_group]
    device = first_final.device
    dtype = first_final.dtype
    num_views = len(view_grads[first_group])
    num_gaussians = first_final.shape[0]

    total_view_norm_sq = torch.zeros((num_views, num_gaussians), device=device, dtype=dtype)
    for group in groups:
        if len(view_grads[group]) != num_views:
            raise ValueError("All visualization groups must have the same number of per-view gradients.")
        if final_grads[group].shape[0] != num_gaussians:
            raise ValueError("All visualization groups must have the same Gaussian dimension.")
        updates = _stack_update_proxies(
            view_grads[group],
            learning_rates.get(group, 1.0),
            device=device,
            dtype=final_grads[group].dtype,
        )
        flat_updates = updates.reshape(num_views, num_gaussians, -1)
        total_view_norm_sq += flat_updates.square().sum(dim=-1).to(dtype=dtype)

    raw_view_update_norm = total_view_norm_sq.clamp_min(0.0).sqrt()
    visible = raw_view_update_norm > eps
    if max_views > 0 and max_views < num_views:
        top_values, top_indices = raw_view_update_norm.topk(k=max_views, dim=0)
        top_visible = torch.zeros_like(visible)
        top_visible.scatter_(0, top_indices, top_values > eps)
        visible = visible & top_visible

    visible_f = visible.to(dtype=dtype)
    visible_counts = visible_f.sum(dim=0)
    active = visible_counts > 0
    view_update_norm = torch.where(visible, raw_view_update_norm, torch.zeros_like(raw_view_update_norm))

    mean_update_norm_sq = torch.zeros(num_gaussians, device=device, dtype=dtype)
    final_update_norm_sq = torch.zeros(num_gaussians, device=device, dtype=dtype)
    rgb_attributes: Dict[str, torch.Tensor] = {}
    scalar_attributes: Dict[str, torch.Tensor] = {
        f"view_update_norm_{view_idx:02d}": view_update_norm[view_idx] for view_idx in range(num_views)
    }

    for group in groups:
        group_dtype = final_grads[group].dtype
        updates = _stack_update_proxies(
            view_grads[group],
            learning_rates.get(group, 1.0),
            device=device,
            dtype=group_dtype,
        )
        flat_updates = updates.reshape(num_views, num_gaussians, -1)
        mean_flat = (flat_updates * visible_f.to(dtype=group_dtype)[..., None]).sum(dim=0)
        mean_flat = mean_flat / visible_counts.clamp_min(1.0).to(dtype=group_dtype)[..., None]
        mean_update = mean_flat.reshape_as(final_grads[group])
        mean_update_norm_sq += mean_flat.square().sum(dim=-1).to(dtype=dtype)

        final_update = -float(learning_rates.get(group, 1.0)) * final_grads[group].detach()
        final_flat = final_update.reshape(num_gaussians, -1)
        final_update_norm_sq += final_flat.square().sum(dim=-1).to(dtype=dtype)

        if group == "features_dc" and mean_update.reshape(num_gaussians, -1).shape[-1] == 3:
            rgb_attributes["rgb_update_mean"] = mean_update.reshape(num_gaussians, 3)
            rgb_attributes["rgb_update_final"] = final_update.reshape(num_gaussians, 3)
        if group == "opacities":
            scalar_attributes["opacity_update_mean"] = mean_update.reshape(num_gaussians, -1)[:, 0]
            scalar_attributes["opacity_update_final"] = final_update.reshape(num_gaussians, -1)[:, 0]

    expected_norm_sq = (total_view_norm_sq * visible_f).sum(dim=0)
    expected_norm_sq = expected_norm_sq / visible_counts.clamp_min(1.0)
    mean_update_norm = mean_update_norm_sq.clamp_min(0.0).sqrt()
    final_update_norm = final_update_norm_sq.clamp_min(0.0).sqrt()
    agreement = mean_update_norm / expected_norm_sq.clamp_min(eps).sqrt()
    agreement = torch.where(active, agreement.clamp(0.0, 1.0), torch.zeros_like(agreement))
    disagreement = 1.0 - agreement
    disagreement = torch.where(active, disagreement, torch.zeros_like(disagreement))

    dominant_norm, dominant_view = view_update_norm.max(dim=0)
    dominant_view = torch.where(active, dominant_view, torch.full_like(dominant_view, -1))
    dominance_strength = dominant_norm / view_update_norm.sum(dim=0).clamp_min(eps)
    dominance_strength = torch.where(active, dominance_strength, torch.zeros_like(dominance_strength))
    suppression_ratio = final_update_norm / mean_update_norm.clamp_min(eps)
    suppression_ratio = torch.where(active, suppression_ratio, torch.zeros_like(suppression_ratio))

    scalar_attributes.update(
        {
            "visible_view_count": visible_counts,
            "mean_update_norm": mean_update_norm,
            "final_update_norm": final_update_norm,
            "agreement": agreement,
            "disagreement": disagreement,
            "dominant_view": dominant_view.clamp_min(0).to(dtype=dtype),
            "dominance_strength": dominance_strength,
            "suppression_ratio": suppression_ratio,
        }
    )
    per_gaussian = {
        "view_update_norm": view_update_norm,
        "mean_update_norm": mean_update_norm,
        "final_update_norm": final_update_norm,
        "agreement": agreement,
        "disagreement": disagreement,
        "dominant_view": dominant_view,
        "dominance_strength": dominance_strength,
        "suppression_ratio": suppression_ratio,
        "visible_counts": visible_counts,
    }
    return per_gaussian, scalar_attributes, rgb_attributes
