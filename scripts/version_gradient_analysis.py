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


def run_training_steps(
    pipeline: VanillaPipeline,
    camera: Cameras,
    target_image: torch.Tensor,
    trainable_groups: List[str],
    original_state: Dict[str, torch.Tensor],
    learning_rates: Dict[str, float],
    num_steps: int = 1,
) -> Dict[str, torch.Tensor]:
    """Run N gradient descent steps and return the final updated parameters.

    Resets model to original_state before training.
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

    camera = camera.to(device)
    target = target_image.to(device)

    for step in range(num_steps):
        # Zero existing gradients
        for name in trainable_groups:
            param = pipeline.model.gauss_params[name]
            if param.grad is not None:
                param.grad.zero_()

        # Forward pass
        pipeline.model.train()
        model_outputs = pipeline.model(camera)

        # Build batch with target image
        batch = {"image": target}
        metrics_dict = pipeline.model.get_metrics_dict(model_outputs, batch)
        loss_dict = pipeline.model.get_loss_dict(model_outputs, batch, metrics_dict)
        loss = functools.reduce(torch.add, loss_dict.values())

        # Backward
        loss.backward()

        # Manual SGD update
        with torch.no_grad():
            for name in trainable_groups:
                param = pipeline.model.gauss_params[name]
                if param.grad is not None:
                    param.add_(param.grad, alpha=-learning_rates[name])

    # Collect final updated parameters
    updated = {}
    for name in trainable_groups:
        updated[name] = pipeline.model.gauss_params[name].detach().clone()

    return updated


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
) -> Tuple[np.ndarray, float, float]:
    """Render a per-Gaussian scalar attribute as a colormap image.

    Returns (image, lower, upper) where lower/upper are the data range used for scaling.
    """
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
        h, w = int(camera.height.item()), int(camera.width.item())
        return np.zeros((h, w, 3)), 0.0, 1.0

    value = rendered["value"].cpu()
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
    return result, lower, upper


def add_colorbar(image: np.ndarray, lower: float, upper: float, bar_width: int = 30) -> np.ndarray:
    """Add a vertical colorbar to the right side of an image."""
    from PIL import ImageDraw, ImageFont

    h, w = image.shape[:2]
    # Create colorbar gradient (turbo colormap, vertical)
    gradient = torch.linspace(1, 0, h).unsqueeze(-1)  # top=high, bottom=low
    options = colormaps.ColormapOptions(colormap="turbo", normalize=False)
    bar = colormaps.apply_colormap(gradient, colormap_options=options).numpy()  # [H, 1, 3]
    bar = np.repeat(bar, bar_width, axis=1)  # [H, bar_width, 3]

    # Combine image + bar + label area
    label_width = bar_width * 3
    combined = np.ones((h, w + bar_width + label_width, 3), dtype=np.float32) * 0.15
    combined[:, :w] = image
    combined[:, w:w + bar_width] = bar

    # Draw tick labels
    pil_img = Image.fromarray((np.clip(combined, 0, 1) * 255).astype(np.uint8))
    draw = ImageDraw.Draw(pil_img)
    font_size = max(10, min(h // 30, 24))
    font = None
    for fp in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/System/Library/Fonts/Helvetica.ttc"]:
        try:
            font = ImageFont.truetype(fp, font_size)
            break
        except (OSError, IOError):
            continue
    if font is None:
        font = ImageFont.load_default()

    x_label = w + bar_width + 4
    # Format values compactly
    def fmt(v: float) -> str:
        if abs(v) < 1e-3 or abs(v) > 1e4:
            return f"{v:.1e}"
        return f"{v:.4f}"

    draw.text((x_label, 2), fmt(upper), fill=(220, 220, 220), font=font)
    mid_y = h // 2 - font_size // 2
    draw.text((x_label, mid_y), fmt((lower + upper) / 2), fill=(220, 220, 220), font=font)
    draw.text((x_label, h - font_size - 4), fmt(lower), fill=(220, 220, 220), font=font)

    return np.array(pil_img).astype(np.float32) / 255.0


def build_dashboard(
    rows: List[List[Tuple[str, np.ndarray]]],
    output_path: Path,
    row_titles: Optional[List[str]] = None,
) -> None:
    """Build a dashboard PNG from rows of labeled images.

    Args:
        rows: List of rows, each row is a list of (label, image) tuples.
        output_path: Where to save the dashboard PNG.
        row_titles: Optional titles for each row (shown on the left side).
    """
    from PIL import ImageDraw, ImageFont

    if not rows:
        return

    # Find max cell size across all images
    cell_h = max(img.shape[0] for row in rows for _, img in row)
    cell_w = max(img.shape[1] for row in rows for _, img in row)

    # Scale font size relative to image size
    font_size = max(16, min(cell_h // 20, 48))
    label_height = font_size + 10
    padding = 6
    row_title_width = font_size * 10 if row_titles else 0

    # Try to load a good font at the right size
    font = None
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSMono.ttf",
    ]
    for fp in font_paths:
        try:
            font = ImageFont.truetype(fp, font_size)
            break
        except (OSError, IOError):
            continue
    if font is None:
        font = ImageFont.load_default()

    max_cols = max(len(row) for row in rows)
    total_w = row_title_width + max_cols * (cell_w + padding) - padding
    total_h = len(rows) * (cell_h + label_height + padding) - padding

    canvas = np.ones((total_h, total_w, 3), dtype=np.float32) * 0.15
    pil_canvas = Image.fromarray((canvas * 255).astype(np.uint8))
    draw = ImageDraw.Draw(pil_canvas)

    for row_idx, row in enumerate(rows):
        y_offset = row_idx * (cell_h + label_height + padding)

        # Draw row title on the left
        if row_titles and row_idx < len(row_titles):
            title_y = y_offset + label_height + cell_h // 2 - font_size // 2
            draw.text((4, title_y), row_titles[row_idx], fill=(200, 200, 200), font=font)

        for col_idx, (label, img) in enumerate(row):
            x_offset = row_title_width + col_idx * (cell_w + padding)
            # Draw label above image
            draw.text(
                (x_offset + 4, y_offset + 2),
                label,
                fill=(255, 255, 255),
                font=font,
            )
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
    parser.add_argument("--num-steps", type=int, default=None, help="Number of gradient steps per version")
    cli_args = parser.parse_args()

    cfg = load_yaml_config(cli_args.config)
    output_dir = cli_args.output_dir or Path(cfg.get("output_dir", "outputs/version_analysis"))
    trainable = cli_args.trainable or cfg.get("trainable", ["features_dc", "features_rest", "opacities"])
    lr_override = cli_args.lr or cfg.get("lr", None)
    num_steps = cli_args.num_steps or cfg.get("num_steps", 100)
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
            num_steps=num_steps,
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

    num_steps = args.num_steps
    print(f"Training steps per version: {num_steps}")

    # Per-version: run N training steps and collect final params
    all_updated_params: List[Dict[str, torch.Tensor]] = []
    version_target_images: List[np.ndarray] = []

    for v_idx, version_path in enumerate(args.versions):
        print(f"\nVersion {v_idx}: {version_path.name}")
        target_image = load_edited_image(version_path, h, w)
        version_target_images.append(target_image.numpy())

        updated = run_training_steps(
            pipeline, camera, target_image, args.trainable,
            original_state, learning_rates, num_steps=num_steps,
        )
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

    # Active mask: Gaussians that changed in any version
    combined_delta_norm = torch.zeros(num_gaussians, device=device)
    for updated in all_updated_params:
        for group in args.trainable:
            delta = (updated[group].to(device) - original_state[group].to(device)).reshape(num_gaussians, -1).float()
            combined_delta_norm += delta.norm(dim=-1)
    active_mask = combined_delta_norm > 1e-8

    print(f"Active Gaussians: {active_mask.sum().item()}/{num_gaussians}")

    # Render variance maps (with colorbar)
    rendered_var_maps = {}
    for group, var_tensor in variance_maps.items():
        img, lo, hi = render_attribute_map(
            pipeline, camera, var_tensor, original_state, active_mask=active_mask
        )
        rendered_var_maps[group] = add_colorbar(img, lo, hi)
    img, lo, hi = render_attribute_map(
        pipeline, camera, total_var, original_state, active_mask=active_mask
    )
    rendered_var_maps["total"] = add_colorbar(img, lo, hi)

    # Build dashboard
    # Row 1: Original + version input images
    row1 = [("Original Render", original_rgb)]
    for v_idx, img in enumerate(version_target_images):
        row1.append((f"Edit Version {v_idx}", img))

    # Row 2: Per-version rendered results + mean result
    row2 = []
    for v_idx, rgb in enumerate(version_renders):
        row2.append((f"After {num_steps} Steps (v{v_idx})", rgb))
    row2.append(("Mean of All Versions", mean_render))

    # Row 3: Variance maps
    row3 = []
    for group in args.trainable:
        short = group.replace("features_", "")
        row3.append((f"Variance: {short}", rendered_var_maps[group]))
    row3.append(("Variance: Total", rendered_var_maps["total"]))

    dashboard_rows = [row for row in [row1, row2, row3] if row]
    row_titles = ["Inputs", "Renders", "Variance"]
    output_dir = args.output_dir / args.view_name
    build_dashboard(dashboard_rows, output_dir / "dashboard.png", row_titles=row_titles)

    # Save npz with raw data
    npz_data = {
        "original_rgb": original_rgb.astype(np.float32),
        "mean_render": mean_render.astype(np.float32),
    }
    for v_idx in range(num_versions):
        npz_data[f"version_{v_idx}_target"] = version_target_images[v_idx].astype(np.float32)
        npz_data[f"version_{v_idx}_render"] = version_renders[v_idx].astype(np.float32)
    for group in args.trainable:
        npz_data[f"variance_{group}"] = variance_maps[group].cpu().numpy()
    npz_data["variance_total"] = total_var.cpu().numpy()
    npz_data["active_mask"] = active_mask.cpu().numpy()

    np.savez_compressed(output_dir / "data.npz", **npz_data)
    print(f"Data saved to {output_dir / 'data.npz'}")


if __name__ == "__main__":
    main()
