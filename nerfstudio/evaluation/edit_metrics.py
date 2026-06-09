"""Post-training metrics for 3D edit experiments."""

from __future__ import annotations

import copy
import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

from nerfstudio.configs.evaluation_config import EditEvaluationConfig, EvaluationReferenceConfig
from nerfstudio.configs.method_configs import all_methods
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils.eval_utils import eval_load_checkpoint, eval_setup
from nerfstudio.utils.rich_utils import CONSOLE

GAUSSIAN_PARAM_GROUPS = ("means", "scales", "quats", "features_dc", "features_rest", "opacities")
SUPPORTED_METRICS = {
    "reconstruction",
    "clip_text",
    "clip_direction",
    "clip_image",
    "clip_direction_consistency",
    "gaussian_coverage",
    "gaussian_delta",
    "gaussian_scene",
}
METRIC_ALIASES = {
    "clipdir": "clip_direction",
    "clip_directional": "clip_direction",
    "clip_similarity": "clip_text",
    "clip_text_image": "clip_text",
    "clip_image_image": "clip_image",
    "clip_consistency": "clip_direction_consistency",
    "coverage": "gaussian_coverage",
    "gaussian_deltas": "gaussian_delta",
    "scene": "gaussian_scene",
}
CLIP_METRICS = {"clip_text", "clip_direction", "clip_image", "clip_direction_consistency"}
REFERENCE_METRICS = {"clip_direction", "clip_image", "clip_direction_consistency", "gaussian_delta"}


def normalize_metrics(metrics: Sequence[str]) -> Tuple[str, ...]:
    """Normalize metric names and aliases."""

    if isinstance(metrics, str):
        metrics = (metrics,)
    normalized: List[str] = []
    for metric in metrics:
        key = str(metric).strip().lower().replace("-", "_")
        key = METRIC_ALIASES.get(key, key)
        if key not in SUPPORTED_METRICS:
            valid = ", ".join(sorted(SUPPORTED_METRICS))
            raise ValueError(f"Unknown edit evaluation metric '{metric}'. Valid metrics: {valid}")
        if key not in normalized:
            normalized.append(key)
    return tuple(normalized)


def _has_reference(reference: EvaluationReferenceConfig, saved_config: Optional[TrainerConfig] = None) -> bool:
    if reference.load_checkpoint is not None or reference.load_dir is not None:
        return True
    return saved_config is not None and (saved_config.load_checkpoint is not None or saved_config.load_dir is not None)


def validate_metric_selection(
    evaluation: EditEvaluationConfig,
    saved_config: Optional[TrainerConfig] = None,
    require_clip_dependency: bool = False,
) -> Tuple[str, ...]:
    """Validate metric dependencies and return normalized metric names."""

    metrics = normalize_metrics(evaluation.metrics)
    requested = set(metrics)
    if "clip_text" in requested and not evaluation.clip.prompt:
        raise ValueError("evaluation.clip.prompt is required for metric 'clip_text'.")
    if "clip_direction" in requested:
        if not evaluation.clip.prompt:
            raise ValueError("evaluation.clip.prompt is required for metric 'clip_direction'.")
        if not evaluation.clip.source_prompt:
            raise ValueError("evaluation.clip.source_prompt is required for metric 'clip_direction'.")
    if requested & REFERENCE_METRICS and not _has_reference(evaluation.reference, saved_config):
        needed = ", ".join(sorted(requested & REFERENCE_METRICS))
        raise ValueError(
            f"A reference checkpoint is required for metrics: {needed}. "
            "Set evaluation.reference.load_dir/load_step or evaluation.reference.load_checkpoint."
        )
    if require_clip_dependency and requested & CLIP_METRICS:
        _import_clip()
    return metrics


def summarize_per_image_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Aggregate numeric per-image rows with mean/std/min/max/count."""

    values_by_key: Dict[str, List[float]] = {}
    for row in rows:
        for key, value in row.items():
            if key in {"image_name", "image_idx", "row_index"}:
                continue
            number = _as_float(value)
            if number is None or not math.isfinite(number):
                continue
            values_by_key.setdefault(key, []).append(number)

    summary: Dict[str, Dict[str, float]] = {}
    for key, values in sorted(values_by_key.items()):
        arr = np.asarray(values, dtype=np.float64)
        summary[key] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "count": float(arr.size),
        }
    return summary


def compute_gaussian_delta_stats(
    final_params: Mapping[str, torch.Tensor],
    reference_params: Mapping[str, torch.Tensor],
) -> Dict[str, Any]:
    """Compare Gaussian parameter tensors from final and reference checkpoints."""

    final_count = _infer_gaussian_count(final_params)
    reference_count = _infer_gaussian_count(reference_params)
    result: Dict[str, Any] = {
        "final_count": final_count,
        "reference_count": reference_count,
        "count_delta": None if final_count is None or reference_count is None else final_count - reference_count,
        "groups": {},
        "skipped_groups": {},
    }
    for group in GAUSSIAN_PARAM_GROUPS:
        if group not in final_params:
            result["skipped_groups"][group] = "missing from final model"
            continue
        if group not in reference_params:
            result["skipped_groups"][group] = "missing from reference model"
            continue

        final_tensor = final_params[group].detach().float().cpu()
        reference_tensor = reference_params[group].detach().float().cpu()
        if final_tensor.shape[:1] != reference_tensor.shape[:1]:
            result["skipped_groups"][group] = (
                f"gaussian count mismatch: final={final_tensor.shape[0]}, reference={reference_tensor.shape[0]}"
            )
            continue
        if final_tensor.shape != reference_tensor.shape:
            result["skipped_groups"][group] = (
                f"shape mismatch: final={tuple(final_tensor.shape)}, reference={tuple(reference_tensor.shape)}"
            )
            continue

        diff = final_tensor - reference_tensor
        per_gaussian = diff.reshape(diff.shape[0], -1).norm(dim=-1)
        reference_norm = reference_tensor.reshape(reference_tensor.shape[0], -1).norm(dim=-1)
        result["groups"][group] = {
            "delta_norm_mean": _tensor_stat(per_gaussian, "mean"),
            "delta_norm_rms": float(torch.sqrt(torch.mean(per_gaussian.square())).item()),
            "delta_norm_p95": _tensor_stat(per_gaussian, "p95"),
            "delta_norm_max": _tensor_stat(per_gaussian, "max"),
            "element_rms": float(torch.sqrt(torch.mean(diff.reshape(-1).square())).item()),
            "relative_l2_mean": float((per_gaussian / reference_norm.clamp_min(1e-8)).mean().item()),
        }
    return result


def evaluate_edit_metrics(
    load_config: Path,
    output_dir: Optional[Path] = None,
    eval_num_rays_per_chunk: Optional[int] = None,
) -> Path:
    """Run configured edit metrics and return the summary JSON path."""

    saved_config = _load_config(load_config)
    evaluation = copy.deepcopy(saved_config.evaluation)
    if output_dir is not None:
        evaluation.output_dir = output_dir
    if not evaluation.enabled:
        raise ValueError("Edit evaluation is disabled in config. Set evaluation.enabled: true.")

    metrics = validate_metric_selection(evaluation, saved_config=saved_config, require_clip_dependency=False)
    requested = set(metrics)
    validate_metric_selection(evaluation, saved_config=saved_config, require_clip_dependency=bool(requested & CLIP_METRICS))

    config, pipeline, checkpoint_path, step = eval_setup(
        load_config, eval_num_rays_per_chunk=eval_num_rays_per_chunk, test_mode="test"
    )
    eval_dir = _resolve_output_dir(evaluation.output_dir, config)
    eval_dir.mkdir(parents=True, exist_ok=True)
    render_dir = eval_dir / "renders"
    if evaluation.save_rendered_images:
        render_dir.mkdir(parents=True, exist_ok=True)

    reference_pipeline = None
    reference_checkpoint = None
    reference_step = None
    reference_spec = _reference_from_config(evaluation.reference, saved_config)
    skipped: Dict[str, str] = {}
    if requested & REFERENCE_METRICS:
        reference_pipeline, reference_checkpoint, reference_step = _load_reference_pipeline(
            saved_config,
            reference_spec,
            eval_num_rays_per_chunk=eval_num_rays_per_chunk,
        )

    clip_evaluator = None
    target_text = None
    source_text = None
    text_direction = None
    if requested & CLIP_METRICS:
        device = _pipeline_device(pipeline)
        clip_evaluator = ClipEvaluator(evaluation.clip.model, device=device)
        if evaluation.clip.prompt:
            target_text = clip_evaluator.encode_text(evaluation.clip.prompt)
        if evaluation.clip.source_prompt:
            source_text = clip_evaluator.encode_text(evaluation.clip.source_prompt)
        if target_text is not None and source_text is not None:
            text_direction = _normalize_feature(target_text - source_text)

    dataloader = getattr(pipeline.datamanager, "fixed_indices_eval_dataloader", None)
    if dataloader is None:
        raise ValueError("The configured datamanager does not expose fixed_indices_eval_dataloader.")

    rows: List[Dict[str, Any]] = []
    edit_directions: List[Optional[torch.Tensor]] = []
    camera_positions: List[torch.Tensor] = []
    with torch.no_grad():
        for row_index, (camera, batch) in enumerate(dataloader):
            row: Dict[str, Any] = _image_row_metadata(dataloader, batch, row_index)
            camera_positions.append(_camera_position(camera))

            _sync_if_cuda(pipeline)
            start = time.perf_counter()
            outputs = pipeline.model.get_outputs_for_camera(camera)
            _sync_if_cuda(pipeline)
            row["render_time_s"] = time.perf_counter() - start
            final_rgb = outputs["rgb"].detach()

            if "reconstruction" in requested:
                row.update(_compute_reconstruction_metrics(pipeline.model, outputs, batch))
            if "gaussian_coverage" in requested:
                row.update(_compute_coverage_metrics(pipeline.model, outputs))

            reference_rgb = None
            if reference_pipeline is not None:
                reference_outputs = reference_pipeline.model.get_outputs_for_camera(camera)
                reference_rgb = reference_outputs["rgb"].detach()

            if clip_evaluator is not None:
                final_clip = clip_evaluator.encode_image(final_rgb)
                reference_clip = clip_evaluator.encode_image(reference_rgb) if reference_rgb is not None else None
                if "clip_text" in requested and target_text is not None:
                    row["clip_text"] = _cosine(final_clip, target_text)
                if "clip_image" in requested and reference_clip is not None:
                    row["clip_image"] = _cosine(final_clip, reference_clip)
                if reference_clip is not None:
                    image_direction = _normalize_feature(final_clip - reference_clip)
                    edit_directions.append(image_direction)
                    if "clip_direction" in requested and text_direction is not None:
                        row["clip_direction"] = _cosine(image_direction, text_direction)
                else:
                    edit_directions.append(None)
            elif "clip_direction_consistency" in requested:
                edit_directions.append(None)

            if evaluation.save_rendered_images:
                image_key = f"{int(row['image_idx']):04d}"
                _save_image(render_dir / f"{image_key}_final.png", final_rgb)
                if reference_rgb is not None:
                    _save_image(render_dir / f"{image_key}_reference.png", reference_rgb)
            rows.append(row)

    if "clip_direction_consistency" in requested:
        consistency = _compute_direction_consistency(edit_directions, camera_positions, evaluation.clip.neighbor_count)
        if consistency is None:
            skipped["clip_direction_consistency"] = "requires at least two reference/final CLIP image-direction vectors"
        else:
            for row, score in zip(rows, consistency):
                row["clip_direction_consistency"] = score

    summary: Dict[str, Any] = {
        "experiment_name": config.experiment_name,
        "method_name": config.method_name,
        "checkpoint": str(checkpoint_path),
        "step": step,
        "reference_checkpoint": str(reference_checkpoint) if reference_checkpoint is not None else None,
        "reference_step": reference_step,
        "metrics_requested": list(metrics),
        "num_images": len(rows),
        "aggregate": summarize_per_image_rows(rows),
        "skipped_metrics": skipped,
    }
    if "gaussian_scene" in requested:
        summary["gaussian_scene"] = compute_gaussian_scene_stats(_gaussian_params(pipeline.model))
    if "gaussian_delta" in requested and reference_pipeline is not None:
        summary["gaussian_delta"] = compute_gaussian_delta_stats(
            _gaussian_params(pipeline.model), _gaussian_params(reference_pipeline.model)
        )

    summary_path = eval_dir / "metrics_summary.json"
    summary_path.write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    if evaluation.save_per_image:
        _write_rows_csv(eval_dir / "metrics_per_image.csv", rows)
    CONSOLE.print(f"Saved edit evaluation metrics to: {summary_path}")
    return summary_path


class ClipEvaluator:
    """Small OpenAI CLIP wrapper with IN2N-style image/text cosine metrics."""

    def __init__(self, model_name: str, device: torch.device) -> None:
        clip = _import_clip()
        self.clip = clip
        self.device = device
        self.model, _ = clip.load(model_name, device=device)
        self.model.eval()
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def encode_text(self, text: str) -> torch.Tensor:
        tokens = self.clip.tokenize([text]).to(self.device)
        features = self.model.encode_text(tokens).float()
        return _normalize_feature(features[0]).cpu()

    @torch.no_grad()
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        x = self._preprocess(image)
        features = self.model.encode_image(x).float()
        return _normalize_feature(features[0]).cpu()

    def _preprocess(self, image: torch.Tensor) -> torch.Tensor:
        x = image.detach().to(self.device).float().clamp(0.0, 1.0)
        if x.ndim != 3 or x.shape[-1] != 3:
            raise ValueError(f"CLIP image tensors must have shape [H, W, 3], got {tuple(x.shape)}")
        x = x.permute(2, 0, 1)[None, ...]
        _, _, height, width = x.shape
        short_side = min(height, width)
        scale = 224.0 / max(short_side, 1)
        new_height = max(int(round(height * scale)), 224)
        new_width = max(int(round(width * scale)), 224)
        x = F.interpolate(x, size=(new_height, new_width), mode="bicubic", align_corners=False)
        top = (new_height - 224) // 2
        left = (new_width - 224) // 2
        x = x[:, :, top : top + 224, left : left + 224]
        return (x - self.mean) / self.std


def compute_gaussian_scene_stats(params: Mapping[str, torch.Tensor]) -> Dict[str, Any]:
    """Summarize final Gaussian parameter distributions."""

    summary: Dict[str, Any] = {"num_gaussians": _infer_gaussian_count(params)}
    if "opacities" in params:
        opacity = torch.sigmoid(params["opacities"].detach().float().cpu()).reshape(-1)
        summary.update(_prefixed_stats("opacity", opacity))
        for threshold in (0.01, 0.1, 0.5):
            summary[f"opacity_active_fraction_gt_{threshold:g}"] = float((opacity > threshold).float().mean().item())
    if "scales" in params:
        scales = torch.exp(params["scales"].detach().float().cpu())
        summary.update(_prefixed_stats("scale", scales.reshape(-1)))
        anisotropy = scales.amax(dim=-1) / scales.amin(dim=-1).clamp_min(1e-8)
        summary.update(_prefixed_stats("anisotropy", anisotropy.reshape(-1)))
    for group in ("features_dc", "features_rest"):
        if group in params:
            values = params[group].detach().float().cpu()
            norms = values.reshape(values.shape[0], -1).norm(dim=-1)
            summary.update(_prefixed_stats(f"{group}_norm", norms))
    return summary


def _load_config(config_path: Path) -> TrainerConfig:
    config = yaml.load(config_path.read_text(), Loader=yaml.Loader)
    if not isinstance(config, TrainerConfig):
        raise ValueError(f"Expected TrainerConfig in {config_path}, got {type(config).__name__}")
    if not hasattr(config, "evaluation"):
        config.evaluation = EditEvaluationConfig()
    return config


def _reference_from_config(
    reference: EvaluationReferenceConfig, saved_config: TrainerConfig
) -> EvaluationReferenceConfig:
    resolved = copy.deepcopy(reference)
    if resolved.load_checkpoint is None and resolved.load_dir is None:
        resolved.load_checkpoint = saved_config.load_checkpoint
        resolved.load_dir = saved_config.load_dir
    if resolved.load_step is None:
        resolved.load_step = saved_config.load_step
    return resolved


def _load_reference_pipeline(
    saved_config: TrainerConfig,
    reference: EvaluationReferenceConfig,
    eval_num_rays_per_chunk: Optional[int] = None,
) -> Tuple[Pipeline, Path, int]:
    config = copy.deepcopy(saved_config)
    if config.method_name in all_methods:
        config.pipeline.datamanager._target = all_methods[config.method_name].pipeline.datamanager._target
    if eval_num_rays_per_chunk:
        config.pipeline.model.eval_num_rays_per_chunk = eval_num_rays_per_chunk
    config.load_dir = reference.load_dir
    config.load_step = reference.load_step
    config.load_checkpoint = reference.load_checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = config.pipeline.setup(device=device, test_mode="test")
    assert isinstance(pipeline, Pipeline)
    pipeline.eval()
    if reference.load_checkpoint is not None:
        checkpoint_path = reference.load_checkpoint
        loaded_state = torch.load(checkpoint_path, map_location="cpu")
        pipeline.load_pipeline(loaded_state["pipeline"], loaded_state["step"])
        CONSOLE.print(f":white_check_mark: Done loading reference checkpoint from {checkpoint_path}")
        return pipeline, checkpoint_path, int(loaded_state["step"])
    if reference.load_dir is None:
        raise ValueError("Reference load_dir or load_checkpoint is required.")
    checkpoint_path, step = eval_load_checkpoint(config, pipeline)
    return pipeline, checkpoint_path, step


def _resolve_output_dir(output_dir: Path, config: TrainerConfig) -> Path:
    return output_dir if output_dir.is_absolute() else config.get_base_dir() / output_dir


def _pipeline_device(pipeline: Pipeline) -> torch.device:
    model = pipeline.model
    device = getattr(model, "device", None)
    if device is not None:
        return torch.device(device)
    return next(model.parameters()).device


def _sync_if_cuda(pipeline: Pipeline) -> None:
    device = _pipeline_device(pipeline)
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize(device)


def _image_row_metadata(dataloader: Any, batch: Mapping[str, Any], row_index: int) -> Dict[str, Any]:
    image_idx = _as_int(batch.get("image_idx", row_index))
    row = {"row_index": row_index, "image_idx": image_idx}
    filenames = getattr(getattr(dataloader, "input_dataset", None), "image_filenames", None)
    if filenames is not None and 0 <= image_idx < len(filenames):
        row["image_name"] = Path(filenames[image_idx]).name
    return row


def _compute_reconstruction_metrics(model: Any, outputs: Mapping[str, torch.Tensor], batch: Mapping[str, Any]) -> Dict[str, float]:
    metrics, _ = model.get_image_metrics_and_images(outputs, batch)
    row = {key: float(value) for key, value in metrics.items()}
    gt_img = model.composite_with_background(model.get_gt_img(batch["image"]), outputs["background"])
    pred_img = outputs["rgb"]
    if "mask" in batch:
        mask = model._downscale_if_required(batch["mask"]).to(getattr(model, "device", pred_img.device))
        gt_img = gt_img * mask
        pred_img = pred_img * mask
    diff = gt_img - pred_img
    l1 = torch.abs(diff).mean()
    mse = diff.square().mean()
    ssim_value = row.get("ssim")
    if ssim_value is None:
        gt_for_ssim = torch.moveaxis(gt_img, -1, 0)[None, ...]
        pred_for_ssim = torch.moveaxis(pred_img, -1, 0)[None, ...]
        ssim_value = float(model.ssim(gt_for_ssim, pred_for_ssim))
    dssim = 1.0 - float(ssim_value)
    ssim_lambda = float(getattr(model.config, "ssim_lambda", 0.2))
    row.update(
        {
            "l1": float(l1.item()),
            "mse": float(mse.item()),
            "dssim": dssim,
            "main_loss": (1.0 - ssim_lambda) * float(l1.item()) + ssim_lambda * dssim,
        }
    )
    return row


def _compute_coverage_metrics(model: Any, outputs: Mapping[str, torch.Tensor]) -> Dict[str, float]:
    row: Dict[str, float] = {}
    accumulation = outputs.get("accumulation")
    if isinstance(accumulation, torch.Tensor):
        values = accumulation.detach().float().cpu().reshape(-1)
        row.update(_prefixed_stats("accumulation", values))
        for threshold in (0.1, 0.5, 0.9):
            row[f"low_alpha_fraction_lt_{threshold:g}"] = float((values < threshold).float().mean().item())

    info = getattr(model, "info", {}) or {}
    radii = None
    if isinstance(info, Mapping):
        radii = info.get("radii")
    if isinstance(radii, torch.Tensor):
        radii_cpu = radii.detach().float().cpu()
        num_gaussians = _infer_gaussian_count(_gaussian_params(model))
        if radii_cpu.ndim > 1 and num_gaussians is not None and radii_cpu.shape[-1] == num_gaussians:
            visible = (radii_cpu > 0).reshape(-1, num_gaussians).any(dim=0)
            visible_radii = radii_cpu.reshape(-1, num_gaussians)[:, visible]
        else:
            visible = radii_cpu.reshape(-1) > 0
            visible_radii = radii_cpu.reshape(-1)[visible]
        row["visible_gaussian_count"] = float(visible.sum().item())
        if num_gaussians:
            row["visible_gaussian_fraction"] = row["visible_gaussian_count"] / float(num_gaussians)
        if visible_radii.numel() > 0:
            row["visible_radius_mean"] = float(visible_radii.mean().item())
            row["visible_radius_max"] = float(visible_radii.max().item())
    return row


def _compute_direction_consistency(
    edit_directions: Sequence[Optional[torch.Tensor]], camera_positions: Sequence[torch.Tensor], neighbor_count: int
) -> Optional[List[float]]:
    valid_indices = [idx for idx, direction in enumerate(edit_directions) if direction is not None]
    if len(valid_indices) < 2:
        return None
    k = max(1, int(neighbor_count))
    positions = torch.stack([camera_positions[idx].float().cpu() for idx in valid_indices], dim=0)
    directions = torch.stack([edit_directions[idx].float().cpu() for idx in valid_indices if edit_directions[idx] is not None])
    distances = torch.cdist(positions, positions)
    scores_by_index: Dict[int, float] = {}
    for local_idx, global_idx in enumerate(valid_indices):
        distances[local_idx, local_idx] = torch.inf
        neighbors = torch.topk(distances[local_idx], k=min(k, len(valid_indices) - 1), largest=False).indices
        scores = [float(torch.dot(directions[local_idx], directions[int(neighbor)]).item()) for neighbor in neighbors]
        scores_by_index[global_idx] = float(np.mean(scores)) if scores else float("nan")
    return [scores_by_index.get(idx, float("nan")) for idx in range(len(edit_directions))]


def _camera_position(camera: Any) -> torch.Tensor:
    c2w = camera.camera_to_worlds.detach().cpu()
    if c2w.ndim == 3:
        return c2w[0, :3, 3]
    return c2w[:3, 3]


def _gaussian_params(model: Any) -> Dict[str, torch.Tensor]:
    params = getattr(model, "gauss_params", None)
    if params is None:
        return {}
    return {name: value.detach() for name, value in params.items()}


def _infer_gaussian_count(params: Mapping[str, torch.Tensor]) -> Optional[int]:
    for value in params.values():
        if isinstance(value, torch.Tensor) and value.ndim > 0:
            return int(value.shape[0])
    return None


def _prefixed_stats(prefix: str, values: torch.Tensor) -> Dict[str, float]:
    if values.numel() == 0:
        return {}
    flat = values.detach().float().cpu().reshape(-1)
    return {
        f"{prefix}_mean": _tensor_stat(flat, "mean"),
        f"{prefix}_std": _tensor_stat(flat, "std"),
        f"{prefix}_min": _tensor_stat(flat, "min"),
        f"{prefix}_max": _tensor_stat(flat, "max"),
        f"{prefix}_p95": _tensor_stat(flat, "p95"),
    }


def _tensor_stat(values: torch.Tensor, stat: str) -> float:
    flat = values.detach().float().cpu().reshape(-1)
    if flat.numel() == 0:
        return float("nan")
    if stat == "mean":
        return float(flat.mean().item())
    if stat == "std":
        return float(flat.std(unbiased=False).item())
    if stat == "min":
        return float(flat.min().item())
    if stat == "max":
        return float(flat.max().item())
    if stat == "p95":
        return float(torch.quantile(flat, 0.95).item())
    raise ValueError(f"Unknown tensor stat '{stat}'")


def _normalize_feature(feature: torch.Tensor) -> torch.Tensor:
    return F.normalize(feature.float(), dim=0)


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.dot(_normalize_feature(left.cpu()), _normalize_feature(right.cpu())).item())


def _as_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        return float(value.detach().cpu().item())
    return None


def _as_int(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.detach().cpu().item())
    return int(value)


def _save_image(path: Path, image: torch.Tensor) -> None:
    array = image.detach().float().cpu().clamp(0.0, 1.0).numpy()
    array = (array * 255.0).round().astype(np.uint8)
    Image.fromarray(array).save(path)


def _write_rows_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key, "")) for key in keys})


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return float(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _import_clip() -> Any:
    try:
        import clip  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "CLIP metrics require the optional OpenAI CLIP package. "
            "Install it with: python -m pip install git+https://github.com/openai/CLIP.git"
        ) from exc
    return clip
