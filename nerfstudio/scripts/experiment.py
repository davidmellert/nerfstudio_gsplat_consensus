#!/usr/bin/env python
"""Run one or more Nerfstudio experiments from a small YAML file."""

from __future__ import annotations

import copy
import dataclasses
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import tyro
import yaml

from nerfstudio.configs.method_configs import method_configs
from nerfstudio.data.dataparsers.colmap_dataparser import ColmapDataParserConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.scripts.render import RenderCameraPath
from nerfstudio.scripts.train import launch, train_loop
from nerfstudio.utils.available_devices import get_available_devices
from nerfstudio.utils.rich_utils import CONSOLE

GAUSSIAN_PARAM_GROUPS = ("means", "scales", "quats", "features_dc", "features_rest", "opacities")
APPEARANCE_PARAM_GROUPS = ("features_dc", "features_rest")
GROUP_ALIASES = {
    "mean": "means",
    "means": "means",
    "scale": "scales",
    "scales": "scales",
    "quat": "quats",
    "quats": "quats",
    "rotation": "quats",
    "rotations": "quats",
    "features_dc": "features_dc",
    "feature_dc": "features_dc",
    "dc": "features_dc",
    "features_rest": "features_rest",
    "feature_rest": "features_rest",
    "rest": "features_rest",
    "opacity": "opacities",
    "opacities": "opacities",
}
PATH_KEYS = {
    "base_config",
    "camera_path",
    "camera_path_filename",
    "data",
    "load_checkpoint",
    "load_dir",
    "output_dir",
    "output_path",
    "relative_model_dir",
}
RESERVED_RUN_KEYS = {
    "base_config",
    "consensus",
    "datamanager",
    "dataparser",
    "defaults",
    "logging",
    "machine",
    "method",
    "model",
    "name",
    "overrides",
    "render",
    "runs",
    "suite_name",
    "trainer",
    "trainable",
    "viewer",
}


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-._")
    return slug or "run"


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)  # type: ignore[arg-type]
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _as_path(value: Any) -> Path:
    return value if isinstance(value, Path) else Path(value)


def _resolve_data_path(value: Any) -> Path:
    path = _as_path(value)
    if path.exists() or path.is_absolute():
        return path
    data_path = Path("data") / path
    if data_path.exists():
        return data_path
    return path


def _set_datamanager_data(config: TrainerConfig, value: Any) -> None:
    data_path = _resolve_data_path(value)
    config.data = data_path
    config.pipeline.datamanager.data = data_path
    dataparser = getattr(config.pipeline.datamanager, "dataparser", None)
    if dataparser is not None and hasattr(dataparser, "data"):
        dataparser.data = data_path


def _coerce_value(name: str, value: Any, current: Any = None) -> Any:
    if value is None:
        return None
    if isinstance(current, Path) or name in PATH_KEYS or name.endswith(("_path", "_dir", "_filename")):
        return _as_path(value)
    if isinstance(current, tuple) and isinstance(value, list):
        return tuple(value)
    return value


def _apply_object_overrides(target: Any, values: Mapping[str, Any]) -> None:
    for key, value in values.items():
        if not hasattr(target, key):
            raise ValueError(f"{type(target).__name__} has no field '{key}'")
        setattr(target, key, _coerce_value(key, value, getattr(target, key)))


def _set_dotted(target: Any, dotted_key: str, value: Any) -> None:
    obj = target
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        if not hasattr(obj, part):
            raise ValueError(f"Cannot resolve override '{dotted_key}': missing '{part}'")
        obj = getattr(obj, part)
    final = parts[-1]
    if not hasattr(obj, final):
        raise ValueError(f"Cannot resolve override '{dotted_key}': missing '{final}'")
    setattr(obj, final, _coerce_value(final, value, getattr(obj, final)))


def _normalize_group_name(name: str) -> str:
    key = name.strip().lower().replace("-", "_")
    if key not in GROUP_ALIASES:
        raise ValueError(f"Unknown Gaussian parameter group '{name}'. Valid groups: {GAUSSIAN_PARAM_GROUPS}")
    return GROUP_ALIASES[key]


def _normalize_trainable_groups(value: Any) -> Optional[Tuple[str, ...]]:
    if value is None:
        return None
    if isinstance(value, str):
        key = value.strip().lower().replace("-", "_")
        if key in ("all", "true"):
            return GAUSSIAN_PARAM_GROUPS
        if key in ("appearance", "color", "colors"):
            return APPEARANCE_PARAM_GROUPS
        if key in ("none", "false"):
            raise ValueError("At least one Gaussian parameter group must be trainable.")
        return (_normalize_group_name(value),)
    if isinstance(value, Mapping):
        groups = [_normalize_group_name(name) for name, enabled in value.items() if bool(enabled)]
    else:
        groups = [_normalize_group_name(str(name)) for name in value]
    deduped = tuple(group for group in GAUSSIAN_PARAM_GROUPS if group in set(groups))
    if len(deduped) == 0:
        raise ValueError("At least one Gaussian parameter group must be trainable.")
    return deduped


def _load_base_config(run_spec: Mapping[str, Any]) -> TrainerConfig:
    base_config = run_spec.get("base_config")
    if base_config is not None:
        config = yaml.load(_as_path(base_config).read_text(), Loader=yaml.Loader)
        if not isinstance(config, TrainerConfig):
            raise ValueError(f"base_config must contain a TrainerConfig, got {type(config).__name__}")
        config = copy.deepcopy(config)
        config.load_config = None
        config.timestamp = "{timestamp}"
        return config

    method = str(run_spec.get("method", "splatfacto"))
    if method not in method_configs:
        known = ", ".join(sorted(method_configs.keys()))
        raise ValueError(f"Unknown method '{method}'. Known methods: {known}")
    config = copy.deepcopy(method_configs[method])
    if not isinstance(config, TrainerConfig):
        raise ValueError(f"Method '{method}' does not resolve to a TrainerConfig")
    return config


def _apply_dataparser(config: TrainerConfig, dataparser_spec: Any) -> None:
    if dataparser_spec is None:
        return
    if isinstance(dataparser_spec, str):
        if dataparser_spec != "colmap":
            raise ValueError("Only dataparser: colmap is supported as a shorthand string.")
        config.pipeline.datamanager.dataparser = ColmapDataParserConfig(load_3D_points=True)
        return
    if not isinstance(dataparser_spec, Mapping):
        raise ValueError("dataparser must be either 'colmap' or a mapping of dataparser fields.")
    _apply_object_overrides(config.pipeline.datamanager.dataparser, dataparser_spec)


def _apply_common_overrides(config: TrainerConfig, run_spec: Mapping[str, Any]) -> None:
    trainer_fields = {field.name for field in dataclasses.fields(config)}
    for key, value in run_spec.items():
        if key in RESERVED_RUN_KEYS or key not in trainer_fields:
            continue
        setattr(config, key, _coerce_value(key, value, getattr(config, key)))

    if "trainer" in run_spec:
        _apply_object_overrides(config, run_spec["trainer"])
    if "machine" in run_spec:
        _apply_object_overrides(config.machine, run_spec["machine"])
    if "viewer" in run_spec:
        _apply_object_overrides(config.viewer, run_spec["viewer"])
    if "logging" in run_spec:
        _apply_object_overrides(config.logging, run_spec["logging"])
    if "datamanager" in run_spec:
        _apply_object_overrides(config.pipeline.datamanager, run_spec["datamanager"])
    _apply_dataparser(config, run_spec.get("dataparser"))
    if "model" in run_spec:
        _apply_object_overrides(config.pipeline.model, run_spec["model"])
    for dotted_key, value in run_spec.get("overrides", {}).items():
        _set_dotted(config, dotted_key, value)

    if config.data is not None:
        _set_datamanager_data(config, config.data)
    elif config.pipeline.datamanager.data is not None:
        _set_datamanager_data(config, config.pipeline.datamanager.data)


def _normalize_consensus_mode(value: Any) -> str:
    key = str(value).strip().lower().replace("-", "_")
    if key in ("standard", "default", "off", "none"):
        return "standard"
    if key in ("consensus", "gaussian_consensus", "stored", "offline", "normal"):
        return "consensus"
    if key in ("online", "online_consensus"):
        return "online"
    if key in ("batch", "sequential_batch"):
        return "batch"
    raise ValueError("consensus.mode must be one of standard, consensus, online, or batch")


def _apply_consensus(config: TrainerConfig, run_spec: Mapping[str, Any]) -> None:
    consensus = run_spec.get("consensus", {}) or {}
    if not isinstance(consensus, Mapping):
        raise ValueError("consensus must be a mapping")

    model_config = config.pipeline.model
    mode = _normalize_consensus_mode(consensus.get("mode", run_spec.get("mode", "standard")))
    model_config.gaussian_consensus_enabled = mode in ("consensus", "online", "batch")
    if model_config.gaussian_consensus_enabled:
        model_config.gaussian_consensus_mode = "consensus" if mode == "online" else mode

    aggregator = consensus.get("aggregator", consensus.get("aggregation"))
    if aggregator is not None:
        aggregator = str(aggregator).strip().lower()
        if aggregator == "cos":
            aggregator = "cosine"
        if aggregator not in ("mean", "cosine"):
            raise ValueError("consensus.aggregator must be mean or cosine")
        model_config.gaussian_consensus_aggregator = aggregator

    accumulation = consensus.get("accumulation", consensus.get("implementation"))
    if accumulation is None and "online" in consensus:
        accumulation = "online" if bool(consensus["online"]) else "stored"
    if accumulation is None:
        accumulation = "online" if mode == "online" else "stored"
    if accumulation is not None:
        accumulation = str(accumulation).strip().lower()
        if accumulation in ("normal", "offline", "store", "stored"):
            accumulation = "stored"
        if accumulation not in ("online", "stored"):
            raise ValueError("consensus.accumulation must be online or stored")
        model_config.gaussian_consensus_accumulation = accumulation

    for source_name, target_name in (
        ("num_views", "gaussian_consensus_num_views"),
        ("max_views_per_gaussian", "gaussian_consensus_max_views_per_gaussian"),
        ("view_sampling", "gaussian_consensus_view_sampling"),
        ("neighbor_pool_size", "gaussian_consensus_neighbor_pool_size"),
        ("position_weight", "gaussian_consensus_position_weight"),
        ("direction_weight", "gaussian_consensus_direction_weight"),
        ("min_alignment", "gaussian_consensus_min_alignment"),
        ("gaussian_chunk_size", "gaussian_consensus_gaussian_chunk_size"),
        ("densify_min_view_support", "gaussian_consensus_densify_min_view_support"),
        ("store_grads_on_cpu", "gaussian_consensus_store_grads_on_cpu"),
    ):
        if source_name in consensus:
            setattr(model_config, target_name, consensus[source_name])

    visualization = consensus.get("visualization", {}) or {}
    if not isinstance(visualization, Mapping):
        raise ValueError("consensus.visualization must be a mapping")
    for source_name, target_name in (
        ("enabled", "gaussian_consensus_visualization_enabled"),
        ("interval", "gaussian_consensus_visualization_interval"),
        ("window", "gaussian_consensus_visualization_window"),
        ("capture_window", "gaussian_consensus_visualization_window"),
        ("output_dir", "gaussian_consensus_visualization_output_dir"),
        ("save_png", "gaussian_consensus_visualization_save_png"),
        ("save_npz", "gaussian_consensus_visualization_save_npz"),
        ("save_per_gaussian", "gaussian_consensus_visualization_save_per_gaussian"),
    ):
        if source_name in visualization:
            setattr(model_config, target_name, visualization[source_name])
    if "save_raw" in visualization and "save_npz" not in visualization:
        model_config.gaussian_consensus_visualization_save_npz = bool(visualization["save_raw"])
    if "groups" in visualization:
        visualization_groups = visualization["groups"]
        if visualization_groups in (None, "") or visualization_groups == []:
            model_config.gaussian_consensus_visualization_groups = ()
        else:
            groups_for_visualization = _normalize_trainable_groups(visualization_groups)
            assert groups_for_visualization is not None
            model_config.gaussian_consensus_visualization_groups = groups_for_visualization

    densification = consensus.get("densification", consensus.get("densify", False))
    trainable = consensus.get("trainable", consensus.get("trainable_params", run_spec.get("trainable")))
    groups = _normalize_trainable_groups(trainable)

    densification_enabled = bool(densification)
    model_config.gaussian_disable_refinement = not densification_enabled
    model_config.gaussian_consensus_disable_refinement = not densification_enabled
    if densification_enabled and groups is None and model_config.gaussian_consensus_enabled:
        groups = GAUSSIAN_PARAM_GROUPS
    if densification_enabled and groups is not None and set(groups) != set(GAUSSIAN_PARAM_GROUPS):
        raise ValueError("Densification requires all Gaussian parameter groups to be trainable.")

    if groups is not None:
        model_config.gaussian_trainable_param_groups = groups
        model_config.gaussian_consensus_trainable_param_groups = groups


def _build_trainer_config(run_spec: Mapping[str, Any], suite_name: Optional[str], run_name: str) -> TrainerConfig:
    config = _load_base_config(run_spec)
    _apply_common_overrides(config, run_spec)
    _apply_consensus(config, run_spec)

    if suite_name and "experiment_name" not in run_spec:
        config.experiment_name = _slug(suite_name)
    if "method_name" not in run_spec and run_name:
        config.method_name = _slug(run_name)
    if "viewer" not in run_spec or "quit_on_train_completion" not in run_spec.get("viewer", {}):
        config.viewer.quit_on_train_completion = True
    return config


def _prepare_training_config(config: TrainerConfig) -> Path:
    available_device_types = get_available_devices()
    if config.machine.device_type not in available_device_types:
        raise RuntimeError(
            f"Specified device type '{config.machine.device_type}' is not available. "
            f"Available device types: {available_device_types}."
        )
    if config.data:
        CONSOLE.log("Using data alias for pipeline.datamanager.data")
        _set_datamanager_data(config, config.data)
    elif config.pipeline.datamanager.data is not None:
        _set_datamanager_data(config, config.pipeline.datamanager.data)
    if config.prompt:
        CONSOLE.log("Using prompt alias for pipeline.model.prompt")
        config.pipeline.model.prompt = config.prompt
    config.set_timestamp()
    config.print_to_terminal()
    config.save_config()
    return config.get_base_dir() / "config.yml"


def _run_training(config: TrainerConfig) -> Path:
    config_path = _prepare_training_config(config)
    launch(
        main_func=train_loop,
        num_devices_per_machine=config.machine.num_devices,
        device_type=config.machine.device_type,
        num_machines=config.machine.num_machines,
        machine_rank=config.machine.machine_rank,
        dist_url=config.machine.dist_url,
        config=config,
    )
    return config_path


def _render_output_path(
    render_spec: Mapping[str, Any], config: TrainerConfig, suite_name: Optional[str], run_name: str
) -> Path:
    if "output_path" in render_spec:
        return _as_path(render_spec["output_path"])
    root = _as_path(render_spec.get("output_dir", "renders"))
    experiment = _slug(str(config.experiment_name or suite_name or "experiments"))
    method = _slug(str(config.method_name or run_name or "run"))
    render_name = _slug(str(render_spec.get("name", "camera-path")))
    output_format = str(render_spec.get("output_format", "video"))
    suffix = ".mp4" if output_format == "video" else ""
    return root / experiment / method / config.timestamp / f"{render_name}{suffix}"


def _run_render(
    render_spec: Mapping[str, Any], config_path: Path, config: TrainerConfig, suite_name: Optional[str], run_name: str
) -> Optional[Path]:
    if not render_spec or not bool(render_spec.get("enabled", True)):
        return None
    camera_path = render_spec.get("camera_path_filename", render_spec.get("camera_path"))
    if camera_path is None:
        raise ValueError("render.camera_path_filename is required when render is enabled")
    output_path = _render_output_path(render_spec, config, suite_name, run_name)
    command = RenderCameraPath(
        load_config=config_path,
        output_path=output_path,
        camera_path_filename=_as_path(camera_path),
        output_format=render_spec.get("output_format", "video"),
        image_format=render_spec.get("image_format", "jpeg"),
        jpeg_quality=int(render_spec.get("jpeg_quality", 100)),
        downscale_factor=float(render_spec.get("downscale_factor", 1.0)),
        eval_num_rays_per_chunk=render_spec.get("eval_num_rays_per_chunk"),
        rendered_output_names=list(render_spec.get("rendered_output_names", ["rgb"])),
        render_nearest_camera=bool(render_spec.get("render_nearest_camera", False)),
        check_occlusions=bool(render_spec.get("check_occlusions", False)),
        camera_idx=render_spec.get("camera_idx"),
    )
    CONSOLE.rule(f"Render {run_name}")
    command.main()
    return output_path


def _write_run_manifest(base_dir: Path, run_spec: Mapping[str, Any], render_path: Optional[Path]) -> None:
    manifest = {"run": dict(run_spec)}
    if render_path is not None:
        manifest["render_path"] = str(render_path)
    (base_dir / "experiment_run.yml").write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")


@dataclass
class ExperimentCommand:
    """Run a YAML-defined experiment suite."""

    config: Path
    """Experiment YAML file."""
    dry_run: bool = False
    """Prepare and print configs, but do not train or render."""
    skip_train: bool = False
    """Skip training and only run enabled renders against the prepared config."""
    skip_render: bool = False
    """Run training but skip post-run renders."""
    only: Optional[Tuple[str, ...]] = None
    """Optional run names to execute from the YAML file."""

    def main(self) -> None:
        suite = yaml.safe_load(self.config.read_text()) or {}
        defaults = suite.get("defaults", {}) or {}
        raw_runs = suite.get("runs")
        suite_name = suite.get("suite_name", suite.get("name", defaults.get("suite_name")))
        if raw_runs is None:
            raw_runs = [{key: value for key, value in suite.items() if key not in {"defaults", "runs"}}]
        selected = set(self.only or ())

        for index, raw_run in enumerate(raw_runs):
            run_spec = _deep_merge(defaults, raw_run or {})
            run_name = str(run_spec.get("name", f"run-{index + 1}"))
            if selected and run_name not in selected:
                continue

            CONSOLE.rule(f"Experiment {run_name}")
            config = _build_trainer_config(run_spec, suite_name, run_name)
            config_path = _prepare_training_config(config) if self.dry_run or self.skip_train else _run_training(config)
            render_path = None
            if not self.skip_render and not self.dry_run:
                render_path = _run_render(run_spec.get("render", {}) or {}, config_path, config, suite_name, run_name)
            _write_run_manifest(config.get_base_dir(), run_spec, render_path)
            if self.dry_run:
                CONSOLE.log(f"Dry run prepared config: {config_path}")


def entrypoint() -> None:
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(ExperimentCommand).main()


if __name__ == "__main__":
    entrypoint()
