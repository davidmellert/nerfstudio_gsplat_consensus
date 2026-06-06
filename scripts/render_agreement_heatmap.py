#!/usr/bin/env python
"""Render a flyaround video with Gaussians colored by consensus agreement score.

Each Gaussian's color is replaced with a red-green heatmap value:
  - Red (0.0): views disagreed on gradient direction
  - Yellow (0.5): partial agreement
  - Green (1.0): views fully agreed

Usage:
    python scripts/render_agreement_heatmap.py \
        --load-config outputs/.../config.yml \
        --camera-path-filename docs/assets/circle_path.json \
        --param-group features_dc \
        --output-path renders/heatmap_features_dc.mp4
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import mediapy as media
import torch
from rich.console import Console
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from nerfstudio.cameras.camera_paths import get_path_from_json
from nerfstudio.utils.eval_utils import eval_setup

CONSOLE = Console(width=120)


def scores_to_heatmap_colors(scores: torch.Tensor) -> torch.Tensor:
    """Map agreement scores [0, 1] to RGB colors: red -> yellow -> green.

    Args:
        scores: Per-Gaussian agreement scores, shape [N].

    Returns:
        SH DC colors (in linear space before sigmoid), shape [N, 3].
        Since splatfacto applies sigmoid to features_dc when sh_degree=0,
        but uses SH basis when sh_degree>0, we return raw RGB in [0,1]
        and let the caller handle the encoding.
    """
    scores = scores.clamp(0.0, 1.0)
    # Red channel: 1 -> 0 as score goes 0 -> 1
    r = 1.0 - scores
    # Green channel: 0 -> 1 as score goes 0 -> 1
    g = scores
    # Blue channel: 0 always
    b = torch.zeros_like(scores)
    return torch.stack([r, g, b], dim=-1)


def rgb_to_sh_dc(rgb: torch.Tensor) -> torch.Tensor:
    """Convert linear RGB [0,1] to 0th-order spherical harmonic coefficients.

    The SH basis for degree 0 is C0 = 0.28209479177387814.
    The rendering pipeline computes: color = C0 * features_dc + 0.5
    So: features_dc = (rgb - 0.5) / C0
    """
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


@dataclass
class HeatmapConfig:
    load_config: Path
    camera_path_filename: Path
    output_path: Path
    param_group: str = "features_dc"
    rendered_resolution_scaling_factor: float = 1.0
    seconds: float = 5.0


def render_heatmap(cfg: HeatmapConfig) -> None:
    """Load model, replace colors with heatmap, render video."""

    # Load pipeline
    CONSOLE.print(f"[bold]Loading config from {cfg.load_config}")
    _, pipeline, _, _ = eval_setup(cfg.load_config, test_mode="inference")
    model = pipeline.model
    device = pipeline.device

    # Load agreement scores
    ckpt_dir = cfg.load_config.parent / "nerfstudio_models"
    scores_path = ckpt_dir / "agreement_scores.pt"
    if not scores_path.exists():
        CONSOLE.print(f"[bold red]Agreement scores not found at {scores_path}")
        CONSOLE.print("Train with --pipeline.model.store-agreement-scores True first.")
        sys.exit(1)

    agreement_data = torch.load(scores_path, map_location="cpu")
    if cfg.param_group not in agreement_data:
        available = ", ".join(agreement_data.keys())
        CONSOLE.print(f"[bold red]Group '{cfg.param_group}' not in agreement_scores.pt. Available: {available}")
        sys.exit(1)

    scores = agreement_data[cfg.param_group]  # shape [N]
    CONSOLE.print(
        f"[bold green]Loaded agreement scores for '{cfg.param_group}': "
        f"mean={scores.mean():.4f}, min={scores.min():.4f}, max={scores.max():.4f}"
    )

    # Load camera path
    with open(cfg.camera_path_filename, "r", encoding="utf-8") as f:
        camera_path_data = json.load(f)
    camera_path = get_path_from_json(camera_path_data)
    seconds = camera_path_data.get("seconds", cfg.seconds)
    camera_path.rescale_output_resolution(cfg.rendered_resolution_scaling_factor)
    cameras = camera_path.to(device)

    # Save original parameters
    original_dc = model.gauss_params["features_dc"].data.clone()
    original_rest = model.gauss_params["features_rest"].data.clone()

    # Replace colors with heatmap
    heatmap_rgb = scores_to_heatmap_colors(scores)
    heatmap_sh = rgb_to_sh_dc(heatmap_rgb).to(device=device, dtype=original_dc.dtype)
    model.gauss_params["features_dc"].data.copy_(heatmap_sh)
    model.gauss_params["features_rest"].data.zero_()  # disable view-dependent effects

    # Render frames
    fps = len(cameras) / seconds
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)

    frames = []
    progress = Progress(
        TextColumn(":movie_camera: Rendering heatmap :movie_camera:"),
        BarColumn(),
        TaskProgressColumn(
            text_format="[progress.percentage]{task.completed}/{task.total:>.0f}({task.percentage:>3.1f}%)",
        ),
        TimeRemainingColumn(elapsed_when_finished=False, compact=False),
        TimeElapsedColumn(),
    )

    with progress:
        for camera_idx in progress.track(range(cameras.size), description=""):
            with torch.no_grad():
                outputs = model.get_outputs_for_camera(cameras[camera_idx : camera_idx + 1])
            rgb = outputs["rgb"].cpu().numpy()
            frames.append(rgb)

    media.write_video(str(cfg.output_path), frames, fps=fps)
    CONSOLE.print(f"[bold green]Saved heatmap video to {cfg.output_path}")

    # Restore original parameters
    model.gauss_params["features_dc"].data.copy_(original_dc)
    model.gauss_params["features_rest"].data.copy_(original_rest)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--load-config", type=Path, required=True, help="Path to config.yml from training run")
    parser.add_argument("--camera-path-filename", type=Path, required=True, help="Path to camera path JSON")
    parser.add_argument("--output-path", type=Path, required=True, help="Output video path (.mp4)")
    parser.add_argument(
        "--param-group",
        type=str,
        default="features_dc",
        help="Which agreement score to visualize: features_dc, features_rest, or geometry",
    )
    parser.add_argument("--seconds", type=float, default=5.0, help="Video duration in seconds")
    parser.add_argument("--resolution-scale", type=float, default=1.0, help="Scale factor for render resolution")
    args = parser.parse_args()

    cfg = HeatmapConfig(
        load_config=args.load_config,
        camera_path_filename=args.camera_path_filename,
        output_path=args.output_path,
        param_group=args.param_group,
        seconds=args.seconds,
        rendered_resolution_scaling_factor=args.resolution_scale,
    )
    render_heatmap(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
