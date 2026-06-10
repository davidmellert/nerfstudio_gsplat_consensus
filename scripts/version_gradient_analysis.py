#!/usr/bin/env python
"""Analyze gradient variance across multiple edited versions of the same view.

For one camera view with N edited versions, runs N training steps with each
version independently (from the same checkpoint), using the real nerfstudio
Trainer with proper Adam optimizers and LR schedules. Then compares the
per-Gaussian parameter updates and renders the results.

Usage:
    python scripts/version_gradient_analysis.py --config configs/experiments/version_analysis_tobi.yml
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.engine.callbacks import TrainingCallbackLocation
from nerfstudio.engine.trainer import Trainer
from nerfstudio.pipelines.base_pipeline import VanillaPipeline
from nerfstudio.utils import colormaps


def build_trainer_from_spec(spec: Dict) -> Tuple[Trainer, Dict]:
    """Build a full Trainer (with optimizers) from a YAML spec and load checkpoint."""
    from nerfstudio.scripts.experiment import _load_base_config, _apply_common_overrides

    config = _load_base_config(spec)
    _apply_common_overrides(config, spec)

    config.vis = None
    config.viewer.quit_on_train_completion = True
    config.machine.device_type = "cuda" if torch.cuda.is_available() else "cpu"

    # Set trainable param groups so train_iteration uses the right code path
    trainable = spec.get("trainable", ["features_dc", "features_rest", "opacities"])
    config.pipeline.model.gaussian_trainable_param_groups = trainable
    config.pipeline.model.gaussian_disable_refinement = True

    trainer = Trainer(config)
    trainer.setup(test_mode="test")
    # setup() calls _load_checkpoint() which loads pipeline + optimizers + schedulers

    # Save original state for resetting between versions
    original_state = {
        name: param.detach().clone()
        for name, param in trainer.pipeline.model.gauss_params.items()
    }

    # Also save optimizer + scheduler state for resetting
    original_optim_state = {
        k: copy.deepcopy(v.state_dict())
        for k, v in trainer.optimizers.optimizers.items()
    }
    original_sched_state = {
        k: copy.deepcopy(v.state_dict())
        for k, v in trainer.optimizers.schedulers.items()
    }
    original_scaler_state = copy.deepcopy(trainer.grad_scaler.state_dict())

    original_state["_optim_state"] = original_optim_state
    original_state["_sched_state"] = original_sched_state
    original_state["_scaler_state"] = original_scaler_state
    original_state["_start_step"] = trainer._start_step

    return trainer, original_state


def reset_trainer(trainer: Trainer, original_state: Dict) -> None:
    """Reset model params, optimizers, schedulers, and grad scaler to checkpoint state."""
    with torch.no_grad():
        for name, param in trainer.pipeline.model.gauss_params.items():
            if name in original_state:
                param.copy_(original_state[name])

    for k, v in original_state["_optim_state"].items():
        if k in trainer.optimizers.optimizers:
            trainer.optimizers.optimizers[k].load_state_dict(copy.deepcopy(v))
    for k, v in original_state["_sched_state"].items():
        if k in trainer.optimizers.schedulers:
            trainer.optimizers.schedulers[k].load_state_dict(copy.deepcopy(v))
    trainer.grad_scaler.load_state_dict(copy.deepcopy(original_state["_scaler_state"]))
    trainer._start_step = original_state["_start_step"]


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


def patch_next_train(trainer: Trainer, cam_idx: int, target_image: torch.Tensor):
    """Monkey-patch datamanager.next_train to always return a fixed camera + edited image."""
    dm = trainer.pipeline.datamanager
    device = trainer.device

    def fixed_next_train(step):
        dm.train_count += 1
        camera = dm.train_cameras[cam_idx : cam_idx + 1].to(device)
        if camera.metadata is None:
            camera.metadata = {}
        camera.metadata["cam_idx"] = cam_idx
        data = {"image": target_image.to(device)}
        return camera, data

    dm.next_train = fixed_next_train


def run_training_steps(
    trainer: Trainer,
    cam_idx: int,
    target_image: torch.Tensor,
    original_state: Dict,
    num_steps: int,
    trainable_groups: List[str],
) -> Dict[str, torch.Tensor]:
    """Run N real training steps using the Trainer's train_iteration, return final params."""
    reset_trainer(trainer, original_state)
    patch_next_train(trainer, cam_idx, target_image)

    start_step = original_state["_start_step"]
    for step_offset in range(num_steps):
        step = start_step + step_offset
        trainer.pipeline.train()

        # Run callbacks (needed for gsplat strategy)
        for callback in trainer.callbacks:
            callback.run_callback_at_location(
                step, location=TrainingCallbackLocation.BEFORE_TRAIN_ITERATION,
            )

        loss, loss_dict, metrics_dict = trainer.train_iteration(step)

        for callback in trainer.callbacks:
            callback.run_callback_at_location(
                step, location=TrainingCallbackLocation.AFTER_TRAIN_ITERATION,
            )

        if step_offset % 50 == 0 or step_offset == num_steps - 1:
            print(f"  step {step_offset+1}/{num_steps}, loss={loss:.6f}")

    # Collect final params
    updated = {}
    for name in trainable_groups:
        updated[name] = trainer.pipeline.model.gauss_params[name].detach().clone()
    return updated


def render_with_params(
    pipeline: VanillaPipeline,
    camera: Cameras,
    updated_params: Dict[str, torch.Tensor],
    original_state: Dict[str, torch.Tensor],
) -> np.ndarray:
    """Apply updated params, render the view, return as numpy RGB."""
    device = next(pipeline.model.parameters()).device
    with torch.no_grad():
        for name, param in pipeline.model.gauss_params.items():
            if name in original_state and not name.startswith("_"):
                param.copy_(original_state[name])
        for name, values in updated_params.items():
            pipeline.model.gauss_params[name].copy_(values.to(device))

    pipeline.model.eval()
    with torch.no_grad():
        outputs = pipeline.model.get_outputs_for_camera(camera.to(device))
    return np.clip(outputs["rgb"].cpu().numpy(), 0.0, 1.0)


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
    with torch.no_grad():
        for name, param in pipeline.model.gauss_params.items():
            if name in original_state and not name.startswith("_"):
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
    gradient = torch.linspace(1, 0, h).view(h, 1, 1)
    options = colormaps.ColormapOptions(colormap="turbo", normalize=False)
    bar = colormaps.apply_colormap(gradient, colormap_options=options).numpy()
    bar = np.repeat(bar, bar_width, axis=1)

    label_width = bar_width * 3
    combined = np.ones((h, w + bar_width + label_width, 3), dtype=np.float32) * 0.15
    combined[:, :w] = image
    combined[:, w:w + bar_width] = bar

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
    """Build a dashboard PNG from rows of labeled images."""
    from PIL import ImageDraw, ImageFont

    if not rows:
        return

    cell_h = max(img.shape[0] for row in rows for _, img in row)
    cell_w = max(img.shape[1] for row in rows for _, img in row)

    font_size = max(16, min(cell_h // 20, 48))
    label_height = font_size + 10
    padding = 6
    row_title_width = font_size * 10 if row_titles else 0

    font = None
    for fp in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]:
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
        if row_titles and row_idx < len(row_titles):
            title_y = y_offset + label_height + cell_h // 2 - font_size // 2
            draw.text((4, title_y), row_titles[row_idx], fill=(200, 200, 200), font=font)

        for col_idx, (label, img) in enumerate(row):
            x_offset = row_title_width + col_idx * (cell_w + padding)
            draw.text((x_offset + 4, y_offset + 2), label, fill=(255, 255, 255), font=font)
            img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_uint8)
            pil_canvas.paste(pil_img, (x_offset, y_offset + label_height))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pil_canvas.save(str(output_path))
    print(f"Dashboard saved to {output_path}")


def load_yaml_config(config_path: Path) -> Dict:
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


def resolve_versions(versions_dir: Path, view_name: str, num_versions: int) -> List[Path]:
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
    parser.add_argument("--num-steps", type=int, default=None, help="Number of gradient steps per version")
    cli_args = parser.parse_args()

    cfg = load_yaml_config(cli_args.config)
    output_dir = cli_args.output_dir or Path(cfg.get("output_dir", "outputs/version_analysis"))
    trainable = cli_args.trainable or cfg.get("trainable", ["features_dc", "features_rest", "opacities"])
    num_steps = cli_args.num_steps or cfg.get("num_steps", 200)
    num_versions = cfg.get("num_versions", 3)
    versions_dir = Path(cfg["versions_dir"]) if "versions_dir" in cfg else None

    if cli_args.view_name:
        view_names = [cli_args.view_name]
    elif "views" in cfg:
        view_names = cfg["views"]
    elif "view_name" in cfg:
        view_names = [cfg["view_name"]]
    else:
        raise ValueError("Specify --view-name or views in the YAML config")

    print("Building trainer (with optimizers from checkpoint)...")
    trainer, original_state = build_trainer_from_spec(cfg)
    pipeline = trainer.pipeline
    device = trainer.device

    # Print actual optimizer learning rates
    for group in trainable:
        if group in trainer.optimizers.optimizers:
            for pg in trainer.optimizers.optimizers[group].param_groups:
                print(f"  {group}: lr={pg['lr']:.2e}")

    for view_name in view_names:
        print(f"\n{'='*60}")
        print(f"Analyzing view: {view_name}")
        print(f"{'='*60}")

        if cli_args.versions:
            version_paths = cli_args.versions
        elif versions_dir is not None:
            version_paths = resolve_versions(versions_dir, view_name, num_versions)
        else:
            raise ValueError("Specify --versions or versions_dir in YAML config")

        _run_analysis(trainer, original_state, device, view_name, version_paths,
                      output_dir, trainable, num_steps)


def _run_analysis(trainer, original_state, device, view_name, version_paths,
                  output_dir, trainable, num_steps):
    """Run the analysis for a single view."""
    pipeline = trainer.pipeline

    cam_idx = find_camera_index(pipeline, view_name)
    dm = pipeline.datamanager
    camera = dm.train_dataset.cameras[cam_idx : cam_idx + 1].to(device)
    print(f"Using camera index {cam_idx}: {dm.train_dataset.image_filenames[cam_idx].name}")

    h, w = int(camera.height.item()), int(camera.width.item())

    # Render original view
    reset_trainer(trainer, original_state)
    pipeline.model.eval()
    with torch.no_grad():
        original_outputs = pipeline.model.get_outputs_for_camera(camera)
    original_rgb = np.clip(original_outputs["rgb"].cpu().numpy(), 0.0, 1.0)

    num_versions = len(version_paths)
    num_gaussians = pipeline.model.num_points
    print(f"Analyzing {num_versions} versions, {num_gaussians} Gaussians")
    print(f"Training steps per version: {num_steps}")

    # Per-version: run real training steps and collect final params
    all_updated_params: List[Dict[str, torch.Tensor]] = []
    version_target_images: List[np.ndarray] = []

    for v_idx, version_path in enumerate(version_paths):
        print(f"\nVersion {v_idx}: {version_path.name}")
        target_image = load_edited_image(version_path, h, w)
        version_target_images.append(target_image.numpy())

        updated = run_training_steps(
            trainer, cam_idx, target_image, original_state,
            num_steps=num_steps, trainable_groups=trainable,
        )
        all_updated_params.append(updated)

    # Render per-version results
    version_renders: List[np.ndarray] = []
    for v_idx, updated in enumerate(all_updated_params):
        print(f"Rendering version {v_idx}...")
        rgb = render_with_params(pipeline, camera, updated, original_state)
        version_renders.append(rgb)

    # Compute mean updated params and render
    mean_updated = {}
    for group in trainable:
        stacked = torch.stack([up[group] for up in all_updated_params], dim=0)
        mean_updated[group] = stacked.mean(dim=0)
    print("Rendering mean version...")
    mean_render = render_with_params(pipeline, camera, mean_updated, original_state)

    # Compute per-Gaussian variance for each group
    variance_maps = {}
    for group in trainable:
        stacked = torch.stack([up[group].to(device) for up in all_updated_params], dim=0)
        flat = stacked.reshape(num_versions, num_gaussians, -1).float()
        per_gaussian_var = flat.var(dim=0).sum(dim=-1)
        variance_maps[group] = per_gaussian_var

    total_var = sum(variance_maps.values())

    # Active mask: Gaussians that changed in any version
    combined_delta_norm = torch.zeros(num_gaussians, device=device)
    for updated in all_updated_params:
        for group in trainable:
            delta = (updated[group].to(device) - original_state[group].to(device)).reshape(num_gaussians, -1).float()
            combined_delta_norm += delta.norm(dim=-1)
    active_mask = combined_delta_norm > 1e-8

    print(f"Active Gaussians: {active_mask.sum().item()}/{num_gaussians}")

    # Render variance maps (with colorbar)
    # Reset to original state for rendering
    reset_trainer(trainer, original_state)
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
    row1 = [("Original Render", original_rgb)]
    for v_idx, img in enumerate(version_target_images):
        row1.append((f"Edit Version {v_idx}", img))

    row2 = []
    for v_idx, rgb in enumerate(version_renders):
        row2.append((f"After {num_steps} Steps (v{v_idx})", rgb))
    row2.append(("Mean of All Versions", mean_render))

    row3 = []
    for group in trainable:
        short = group.replace("features_", "")
        row3.append((f"Variance: {short}", rendered_var_maps[group]))
    row3.append(("Variance: Total", rendered_var_maps["total"]))

    dashboard_rows = [row for row in [row1, row2, row3] if row]
    row_titles = ["Inputs", "Renders", "Variance"]
    out_dir = output_dir / view_name
    build_dashboard(dashboard_rows, out_dir / "dashboard.png", row_titles=row_titles)

    # Save npz
    npz_data = {
        "original_rgb": original_rgb.astype(np.float32),
        "mean_render": mean_render.astype(np.float32),
    }
    for v_idx in range(num_versions):
        npz_data[f"version_{v_idx}_target"] = version_target_images[v_idx].astype(np.float32)
        npz_data[f"version_{v_idx}_render"] = version_renders[v_idx].astype(np.float32)
    for group in trainable:
        npz_data[f"variance_{group}"] = variance_maps[group].cpu().numpy()
    npz_data["variance_total"] = total_var.cpu().numpy()
    npz_data["active_mask"] = active_mask.cpu().numpy()

    np.savez_compressed(out_dir / "data.npz", **npz_data)
    print(f"Data saved to {out_dir / 'data.npz'}")


if __name__ == "__main__":
    main()
