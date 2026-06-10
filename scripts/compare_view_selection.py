#!/usr/bin/env python
"""Compare reference view selection strategies for Gaussian consensus.

Loads a colmap scene and for each anchor camera shows which reference views
are selected by the original pose_neighborhood strategy vs. the new
lookat_twostage strategy.

Usage:
    python scripts/compare_view_selection.py \
        --data /path/to/scene \
        --images-path images \
        --downscale-factor 8 \
        --anchor _DSC8698 \
        --num-refs 3
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F


def load_scene(data: Path, images_path: str, downscale_factor: int):
    """Load colmap scene and return camera poses + image filenames."""
    from nerfstudio.data.dataparsers.colmap_dataparser import ColmapDataParserConfig

    config = ColmapDataParserConfig(
        data=data,
        images_path=Path(images_path),
        downscale_factor=downscale_factor,
        eval_mode="all",
        skip_missing_images=True,
        auto_downscale_missing_images=False,
    )
    parser = config.setup()
    outputs = parser.get_dataparser_outputs(split="train")
    cameras = outputs.cameras
    filenames = [p.name for p in outputs.image_filenames]
    return cameras, filenames


def get_origins_directions(cameras):
    """Extract camera origins and z-axis directions from camera_to_worlds."""
    c2w = cameras.camera_to_worlds.float()
    origins = c2w[..., 3]
    directions = F.normalize(c2w[..., :3, 2], dim=-1)
    return origins, directions


# --- Original: pose_neighborhood (position + direction, random from pool) ---

def pose_neighborhood_select(
    anchor_idx: int,
    origins: torch.Tensor,
    directions: torch.Tensor,
    num_refs: int,
    pool_size: int = 16,
    position_weight: float = 1.0,
    direction_weight: float = 0.25,
    seed: int = 42,
) -> tuple:
    anchor_origin = origins[anchor_idx]
    anchor_direction = directions[anchor_idx]

    position_distances = torch.linalg.norm(origins - anchor_origin, dim=-1)
    nonzero = position_distances[position_distances > 1e-8]
    position_scale = nonzero.median().clamp_min(1e-8) if nonzero.numel() > 0 else torch.tensor(1.0)
    position_score = position_distances / position_scale

    direction_dot = torch.clamp((directions * anchor_direction).sum(dim=-1), -1.0, 1.0)
    direction_score = 1.0 - direction_dot

    score = position_weight * position_score + direction_weight * direction_score
    score[anchor_idx] = torch.inf

    pool = torch.argsort(score)[:pool_size].tolist()
    rng = random.Random(seed + anchor_idx)
    refs = rng.sample(pool, k=min(num_refs, len(pool)))
    return refs, score, pool


# --- New: lookat_twostage (filter by look-at point, rank by position) ---

def lookat_twostage_select(
    anchor_idx: int,
    origins: torch.Tensor,
    directions: torch.Tensor,
    scene_center: torch.Tensor,
    num_refs: int,
) -> tuple:
    anchor_origin = origins[anchor_idx]
    # Negate z-axis: after COLMAP->OpenGL convention flip, z points backward
    viewing_dirs = -directions

    t = torch.linalg.norm(origins - scene_center.unsqueeze(0), dim=-1)
    look_at_points = origins + t.unsqueeze(-1) * viewing_dirs

    # Stage 1: top pool by look-at distance
    anchor_look_at = look_at_points[anchor_idx]
    lookat_dists = torch.linalg.norm(look_at_points - anchor_look_at, dim=-1)
    lookat_dists[anchor_idx] = torch.inf
    pool_size = min(4 * num_refs, origins.shape[0] - 1)
    lookat_pool = torch.argsort(lookat_dists)[:pool_size].tolist()

    # Stage 2: rank pool by position, take top num_refs
    position_dists = torch.linalg.norm(origins - anchor_origin, dim=-1)
    pool_sorted = sorted(lookat_pool, key=lambda i: position_dists[i].item())
    refs = pool_sorted[:num_refs]

    return refs, lookat_dists, position_dists, lookat_pool, pool_sorted


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, help="Path to scene data")
    parser.add_argument("--images-path", type=str, default="images", help="Images subfolder name")
    parser.add_argument("--downscale-factor", type=int, default=None, help="Downscale factor")
    parser.add_argument("--anchor", type=str, default=None, help="Specific anchor image name (partial match)")
    parser.add_argument("--num-anchors", type=int, default=4, help="Number of random anchors (ignored if --anchor)")
    parser.add_argument("--num-refs", type=int, default=3, help="Number of reference views per anchor")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=Path, default=Path("view_selection_comparison.json"), help="Output JSON")
    args = parser.parse_args()

    print(f"Loading scene from {args.data}...")
    cameras, filenames = load_scene(args.data, args.images_path, args.downscale_factor)
    origins, directions = get_origins_directions(cameras)
    scene_center = origins.mean(dim=0)
    n = len(filenames)
    print(f"Loaded {n} cameras. Scene center: {scene_center.tolist()}")

    if args.anchor:
        matches = [i for i, f in enumerate(filenames) if args.anchor in f]
        if not matches:
            print(f"No image matching '{args.anchor}' found.")
            return
        anchor_indices = matches
    else:
        rng = random.Random(args.seed)
        anchor_indices = rng.sample(range(n), k=min(args.num_anchors, n))

    results = []
    for anchor_idx in anchor_indices:
        print(f"\nAnchor: {filenames[anchor_idx]} (idx {anchor_idx})")

        # --- pose_neighborhood ---
        orig_refs, orig_score, orig_pool = pose_neighborhood_select(
            anchor_idx, origins, directions, args.num_refs, seed=args.seed,
        )
        print(f"\n  pose_neighborhood (original):")
        print(f"    Selected: {[filenames[i] for i in orig_refs]}")
        for rank, i in enumerate(orig_pool):
            marker = " <-" if i in orig_refs else ""
            print(f"      {rank+1:2d}. {filenames[i]:20s} (idx {i:3d}, score {orig_score[i]:.4f}){marker}")

        # --- lookat_twostage ---
        la_refs, la_dists, pos_dists, la_pool, la_sorted = lookat_twostage_select(
            anchor_idx, origins, directions, scene_center, args.num_refs,
        )
        print(f"\n  lookat_twostage (new):")
        print(f"    Selected: {[filenames[i] for i in la_refs]}")
        print(f"    Stage 1 - Top-{len(la_pool)} by look-at distance:")
        for rank, i in enumerate(la_pool):
            print(f"      {rank+1:2d}. {filenames[i]:20s} (idx {i:3d}, lookat {la_dists[i]:.4f})")
        print(f"    Stage 2 - Ranked by position:")
        for rank, i in enumerate(la_sorted):
            marker = " <-" if i in la_refs else ""
            print(f"      {rank+1:2d}. {filenames[i]:20s} (idx {i:3d}, pos {pos_dists[i]:.4f}, lookat {la_dists[i]:.4f}){marker}")

        results.append({
            "anchor_idx": anchor_idx,
            "anchor_image": filenames[anchor_idx],
            "pose_neighborhood": [filenames[i] for i in orig_refs],
            "lookat_twostage": [filenames[i] for i in la_refs],
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"num_cameras": n, "anchors": results}, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
