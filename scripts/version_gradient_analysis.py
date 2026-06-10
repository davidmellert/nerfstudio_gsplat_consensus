#!/usr/bin/env python
"""Analyze gradient variance across multiple edited versions of the same view.

For one camera view with N edited versions, runs a single gradient step with each
version independently (from the same checkpoint), then compares the per-Gaussian
updates and renders the results.

Usage:
    python scripts/version_gradient_analysis.py \
        --load-config /path/to/config.yml \
        --view-name _DSC8685 \
        --versions /path/to/images/_DSC8685_0.JPG /path/to/images/_DSC8685_1.JPG /path/to/images/_DSC8685_2.JPG \
        --output-dir outputs/version_analysis
"""

from __future__ import annotations

import argparse
import copy
import functools
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.pipelines.base_pipeline import VanillaPipeline
from nerfstudio.utils import colormaps


def build_pipeline_from_spec(spec: Dict) -> Tuple[VanillaPipeline, Dict]:
    """Build and load a pipeline from a YAML spec (same fields as experiment configs)."""
    import os
    from nerfstudio.scripts.experiment import _load_base_config, _apply_common_overrides, _set_datamanager_data

    config = _load_base_config(spec)
    _apply_common_overrides(config, spec)

    config.vis = None
    config.viewer.quit_on_train_completion = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.machine.device_type = "cuda" if torch.cuda.is_available() else "cpu"

    pipeline = config.pipeline.setup(device=device, test_mode="test")
    pipeline.to(device)

    # Load checkpoint
    load_dir = config.load_dir
    load_step = config.load_step
    if load_dir is not None:
        load_dir = Path(load_dir)
        if load_step is None:
            load_step = sorted(int(x[x.find("-") + 1 : x.find(".")]) for x in os.listdir(load_dir))[-1]
        load_path = load_dir / f"step-{load_step:09d}.ckpt"
        assert load_path.exists(), f"Checkpoint {load_path} does not exist"
        loaded_state = torch.load(load_path, map_location="cpu")
        pipeline.load_pipeline(loaded_state["pipeline"], loaded_state["step"])
        print(f"Loaded checkpoint from {load_path}")
    elif config.load_checkpoint is not None:
        loaded_state = torch.load(config.load_checkpoint, map_location="cpu")
        pipeline.load_pipeline(loaded_state["pipeline"], loaded_state["step"])
        print(f"Loaded checkpoint from {config.load_checkpoint}")
    else:
        raise ValueError("Either load_dir or load_checkpoint must be specified")

    # Save original state for resetting between versions
    original_state = {
        name: param.detach().clone()
        for name, param in pipeline.model.gauss_params.items()
    }

    return pipeline, original_state


def find_camera_index(pipeline: VanillaPipeline, view_name: str) -> int:
    """Find the camera index matching view_name in the datamanager."""
    dm = pipeline.datamanager
    filenames = [p.name for p in dm.train_dataset.image_filenames]
    matches = [i for i, f in enumerate(filenames) if view_name in f]
    if not matches:
        available = filenames[:10]
        raise ValueError(f"No camera matching '{view_name}'. Available: {available}...")
    return matches[0]


def load_edited_image(image_path: Path, target_h: int, target_w: int) -> torch.Tensor:
    """Load an edited image and resize to match target dimensions."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((target_w, target_h), Image.LANCZOS)
    return torch.from_numpy(np.array(img)).float() / 255.0


def run_single_step(
    pipeline: VanillaPipeline,
    camera: Cameras,
    target_image: torch.Tensor,
    trainable_groups: List[str],
    original_state: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Run one forward+backward pass and return per-group gradients.

    Resets model to original_state before the step.
    """
    device = next(pipeline.model.parameters()).device

    # Reset to original checkpoint state
    with torch.no_grad():
        for name, param in pipeline.model.gauss_params.items():
            param.copy_(original_state[name])

    # Disable refinement so strategy.step_pre_backward is not called
    pipeline.model.config.gaussian_disable_refinement = True

    # Enable gradients only for trainable groups
    for name, param in pipeline.model.gauss_params.items():
        param.requires_grad_(name in trainable_groups)

    # Zero existing gradients
    for name in trainable_groups:
        param = pipeline.model.gauss_params[name]
        if param.grad is not None:
            param.grad.zero_()

    # Forward pass
    pipeline.model.train()
    camera = camera.to(device)
    model_outputs = pipeline.model(camera)

    # Build batch with target image
    batch = {"image": target_image.to(device)}
    metrics_dict = pipeline.model.get_metrics_dict(model_outputs, batch)
    loss_dict = pipeline.model.get_loss_dict(model_outputs, batch, metrics_dict)
    loss = functools.reduce(torch.add, loss_dict.values())

    # Backward
    loss.backward()

    # Collect gradients
    grads = {}
    for name in trainable_groups:
        param = pipeline.model.gauss_params[name]
        if param.grad is not None:
            grads[name] = param.grad.detach().clone()
        else:
            grads[name] = torch.zeros_like(param)

    return grads


def apply_update_and_render(
    pipeline: VanillaPipeline,
    camera: Cameras,
    updated_params: Dict[str, torch.Tensor],
    original_state: Dict[str, torch.Tensor],
) -> np.ndarray:
    """Apply updated params, render the view, return as numpy RGB."""
    device = next(pipeline.model.parameters()).device

    with torch.no_grad():
        # Reset all params to original
        for name, param in pipeline.model.gauss_params.items():
            param.copy_(original_state[name])
        # Apply updated params
        for name, values in updated_params.items():
            pipeline.model.gauss_params[name].copy_(values.to(device))

    pipeline.model.eval()
    with torch.no_grad():
        outputs = pipeline.model.get_outputs_for_camera(camera.to(device))
    rgb = outputs["rgb"].cpu().numpy()
    return np.clip(rgb, 0.0, 1.0)


def render_attribute_map(
    pipeline: VanillaPipeline,
    camera: Cameras,
    attribute: torch.Tensor,
    original_state: Dict[str, torch.Tensor],
    active_mask: Optional[torch.Tensor] = None,
) -> np.ndarray:
    """Render a per-Gaussian scalar attribute as a colormap image."""
    device = next(pipeline.model.parameters()).device

    # Reset to original state for geometry
    with torch.no_grad():
        for name, param in pipeline.model.gauss_params.items():
            param.copy_(original_state[name])

    pipeline.model.eval()
    scalar_attrs = {"value": attribute.to(device)}
    with torch.no_grad():
        rendered = pipeline.model.render_gaussian_attribute_maps_for_camera(
            camera.to(device),
            scalar_attributes=scalar_attrs,
            rgb_attributes={},
            active_mask=active_mask,
        )

    if "value" not in rendered:
        return np.zeros((int(camera.height.item()), int(camera.width.item()), 3))

    value = rendered["value"].cpu()
    # Remove alpha map if present
    rendered.pop("_alpha", None)

    finite = value[torch.isfinite(value)]
    if finite.numel() > 0:
        lower = float(finite.min())
        upper = float(finite.max())
    else:
        lower, upper = 0.0, 1.0
    upper = max(upper, lower + 1e-8)

    bg_mask = torch.isnan(value[..., 0])
    value = torch.nan_to_num(value, nan=0.0)
    value = torch.clamp((value - lower) / (upper - lower), 0.0, 1.0)
    options = colormaps.ColormapOptions(colormap="turbo", normalize=False)
    result = colormaps.apply_colormap(value, colormap_options=options).cpu().numpy()
    result[bg_mask.numpy()] = 0.5
    return result


def build_dashboard(rows: List[List[Tuple[str, np.ndarray]]], output_path: Path) -> None:
    """Build a dashboard PNG from rows of labeled images."""
    from PIL import ImageDraw, ImageFont

    if not rows:
        return

    label_height = 20
    padding = 4
    # Find consistent cell size from the first image
    ref_img = rows[0][0][1]
    cell_h, cell_w = ref_img.shape[:2]

    max_cols = max(len(row) for row in rows)
    total_w = max_cols * (cell_w + padding) - padding
    total_h = len(rows) * (cell_h + label_height + padding) - padding

    canvas = np.ones((total_h, total_w, 3), dtype=np.float32) * 0.15
    pil_canvas = Image.fromarray((canvas * 255).astype(np.uint8))
    draw = ImageDraw.Draw(pil_canvas)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except (OSError, IOError):
        font = ImageFont.load_default()

    for row_idx, row in enumerate(rows):
        y_offset = row_idx * (cell_h + label_height + padding)
        for col_idx, (label, img) in enumerate(row):
            x_offset = col_idx * (cell_w + padding)
            # Draw label
            draw.text((x_offset + 2, y_offset + 2), label, fill=(255, 255, 255), font=font)
            # Paste image
            img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_uint8)
            pil_canvas.paste(pil_img, (x_offset, y_offset + label_height))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pil_canvas.save(str(output_path))
    print(f"Dashboard saved to {output_path}")


def load_yaml_config(config_path: Path) -> Dict:
    """Load a YAML analysis config file."""
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


def resolve_versions(versions_dir: Path, view_name: str, num_versions: int) -> List[Path]:
    """Find version image files for a view in a directory."""
    paths = []
    for v_idx in range(num_versions):
        candidates = list(versions_dir.glob(f"{view_name}_{v_idx}.*"))
        if not candidates:
            candidates = list(versions_dir.glob(f"*{view_name}_{v_idx}.*"))
        if not candidates:
            raise FileNotFoundError(f"No image found for {view_name}_{v_idx} in {versions_dir}")
        paths.append(candidates[0])
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="YAML analysis config file")
    parser.add_argument("--view-name", type=str, default=None, help="Override view to analyze")
    parser.add_argument("--versions", type=Path, nargs="+", default=None, help="Override version image paths")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override output directory")
    parser.add_argument("--trainable", type=str, nargs="+", default=None)
    parser.add_argument("--lr", type=float, default=None, help="Learning rate override")
    cli_args = parser.parse_args()

    cfg = load_yaml_config(cli_args.config)
    output_dir = cli_args.output_dir or Path(cfg.get("output_dir", "outputs/version_analysis"))
    trainable = cli_args.trainable or cfg.get("trainable", ["features_dc", "features_rest", "opacities"])
    lr_override = cli_args.lr or cfg.get("lr", None)
    num_versions = cfg.get("num_versions", 3)
    versions_dir = Path(cfg["versions_dir"]) if "versions_dir" in cfg else None

    # Determine which views to analyze
    if cli_args.view_name:
        view_names = [cli_args.view_name]
    elif "views" in cfg:
        view_names = cfg["views"]
    elif "view_name" in cfg:
        view_names = [cfg["view_name"]]
    else:
        raise ValueError("Specify --view-name or views in the YAML config")

    print("Building pipeline...")
    pipeline, original_state = build_pipeline_from_spec(cfg)
    device = next(pipeline.model.parameters()).device

    for view_name in view_names:
        print(f"\n{'='*60}")
        print(f"Analyzing view: {view_name}")
        print(f"{'='*60}")

        # Resolve version image paths
        if cli_args.versions:
            version_paths = cli_args.versions
        elif versions_dir is not None:
            version_paths = resolve_versions(versions_dir, view_name, num_versions)
        else:
            raise ValueError("Specify --versions or versions_dir in YAML config")

        args = argparse.Namespace(
            view_name=view_name,
            versions=version_paths,
            output_dir=output_dir,
            trainable=trainable,
            lr=lr_override,
        )
        _run_analysis(pipeline, original_state, device, args)


def _run_analysis(pipeline, original_state, device, args):
    """Run the analysis for a single view."""

    cam_idx = find_camera_index(pipeline, args.view_name)
    dm = pipeline.datamanager
    camera = dm.train_dataset.cameras[cam_idx : cam_idx + 1].to(device)
    print(f"Using camera index {cam_idx}: {dm.train_dataset.image_filenames[cam_idx].name}")

    # Get target image dimensions
    h, w = int(camera.height.item()), int(camera.width.item())

    # Get learning rates from optimizer
    learning_rates = {}
    for group in args.trainable:
        if hasattr(pipeline.model, "optimizers") and isinstance(pipeline.model.optimizers, dict):
            lr = pipeline.model._get_optimizer_lr(group)
        else:
            lr = 1e-3  # fallback
        if args.lr is not None:
            lr = args.lr
        learning_rates[group] = lr
    print(f"Learning rates: {learning_rates}")

    # Render original view
    pipeline.model.eval()
    with torch.no_grad():
        original_outputs = pipeline.model.get_outputs_for_camera(camera)
    original_rgb = np.clip(original_outputs["rgb"].cpu().numpy(), 0.0, 1.0)

    num_versions = len(args.versions)
    num_gaussians = pipeline.model.num_points
    print(f"Analyzing {num_versions} versions, {num_gaussians} Gaussians")

    # Per-version: compute gradients and updated params
    all_grads: List[Dict[str, torch.Tensor]] = []
    all_updated_params: List[Dict[str, torch.Tensor]] = []
    version_target_images: List[np.ndarray] = []

    for v_idx, version_path in enumerate(args.versions):
        print(f"\nVersion {v_idx}: {version_path.name}")
        target_image = load_edited_image(version_path, h, w)
        version_target_images.append(target_image.numpy())

        grads = run_single_step(pipeline, camera, target_image, args.trainable, original_state)
        all_grads.append(grads)

        # Compute updated params: param - lr * grad
        updated = {}
        for group in args.trainable:
            lr = learning_rates[group]
            updated[group] = original_state[group].to(device) - lr * grads[group].to(device)
        all_updated_params.append(updated)

    # Render per-version results
    version_renders: List[np.ndarray] = []
    for v_idx, updated in enumerate(all_updated_params):
        print(f"Rendering version {v_idx}...")
        rgb = apply_update_and_render(pipeline, camera, updated, original_state)
        version_renders.append(rgb)

    # Compute mean updated params and render
    mean_updated = {}
    for group in args.trainable:
        stacked = torch.stack([up[group] for up in all_updated_params], dim=0)
        mean_updated[group] = stacked.mean(dim=0)
    print("Rendering mean version...")
    mean_render = apply_update_and_render(pipeline, camera, mean_updated, original_state)

    # Compute per-Gaussian variance for each group
    variance_maps = {}
    for group in args.trainable:
        stacked = torch.stack([up[group].to(device) for up in all_updated_params], dim=0)  # [V, N, ...]
        flat = stacked.reshape(num_versions, num_gaussians, -1).float()
        per_gaussian_var = flat.var(dim=0).sum(dim=-1)  # [N] - sum variance across feature dims
        variance_maps[group] = per_gaussian_var

    # Total variance across all groups
    total_var = sum(variance_maps.values())

    # Active mask: Gaussians that got any gradient
    combined_grad_norm = torch.zeros(num_gaussians, device=device)
    for grads in all_grads:
        for group in args.trainable:
            flat = grads[group].to(device).reshape(num_gaussians, -1).float()
            combined_grad_norm += flat.norm(dim=-1)
    active_mask = combined_grad_norm > 1e-8

    print(f"Active Gaussians: {active_mask.sum().item()}/{num_gaussians}")

    # Render variance maps
    rendered_var_maps = {}
    for group, var_tensor in variance_maps.items():
        rendered_var_maps[group] = render_attribute_map(
            pipeline, camera, var_tensor, original_state, active_mask=active_mask
        )
    rendered_var_maps["total"] = render_attribute_map(
        pipeline, camera, total_var, original_state, active_mask=active_mask
    )

    # Render cosine similarity between each version's gradient and the mean gradient
    # Combined across all groups
    all_flat_grads = []
    for grads in all_grads:
        parts = []
        for group in args.trainable:
            lr = learning_rates[group]
            update = (-lr * grads[group].to(device)).reshape(num_gaussians, -1).float()
            parts.append(update)
        all_flat_grads.append(torch.cat(parts, dim=-1))  # [N, D_total]

    stacked_grads = torch.stack(all_flat_grads, dim=0)  # [V, N, D_total]
    mean_grad = stacked_grads.mean(dim=0)  # [N, D_total]
    cosine_sims = []
    for v_idx in range(num_versions):
        cos = F.cosine_similarity(stacked_grads[v_idx], mean_grad, dim=-1, eps=1e-8)
        cos = torch.where(active_mask, cos, torch.zeros_like(cos))
        cosine_sims.append(cos)

    rendered_cosine_maps = []
    for v_idx, cos in enumerate(cosine_sims):
        rendered_cosine_maps.append(
            render_attribute_map(pipeline, camera, cos, original_state, active_mask=active_mask)
        )

    # Build dashboard
    # Row 1: Original + version input images
    row1 = [("original", original_rgb)]
    for v_idx, img in enumerate(version_target_images):
        row1.append((f"version {v_idx}", img))

    # Row 2: Per-version rendered results + mean result
    row2 = []
    for v_idx, rgb in enumerate(version_renders):
        row2.append((f"result v{v_idx}", rgb))
    row2.append(("mean result", mean_render))

    # Row 3: Cosine similarity per version
    row3 = []
    for v_idx, cmap in enumerate(rendered_cosine_maps):
        row3.append((f"cos sim v{v_idx}", cmap))

    # Row 4: Variance maps
    row4 = []
    for group in args.trainable:
        short = group.replace("features_", "")
        row4.append((f"var {short}", rendered_var_maps[group]))
    row4.append(("var total", rendered_var_maps["total"]))

    dashboard_rows = [row for row in [row1, row2, row3, row4] if row]
    output_dir = args.output_dir / args.view_name
    build_dashboard(dashboard_rows, output_dir / "dashboard.png")

    # Save npz with raw data
    npz_data = {
        "original_rgb": original_rgb.astype(np.float32),
        "mean_render": mean_render.astype(np.float32),
    }
    for v_idx in range(num_versions):
        npz_data[f"version_{v_idx}_target"] = version_target_images[v_idx].astype(np.float32)
        npz_data[f"version_{v_idx}_render"] = version_renders[v_idx].astype(np.float32)
        npz_data[f"version_{v_idx}_cosine_sim"] = cosine_sims[v_idx].cpu().numpy()
    for group in args.trainable:
        npz_data[f"variance_{group}"] = variance_maps[group].cpu().numpy()
    npz_data["variance_total"] = total_var.cpu().numpy()
    npz_data["active_mask"] = active_mask.cpu().numpy()

    np.savez_compressed(output_dir / "data.npz", **npz_data)
    print(f"Data saved to {output_dir / 'data.npz'}")


if __name__ == "__main__":
    main()
