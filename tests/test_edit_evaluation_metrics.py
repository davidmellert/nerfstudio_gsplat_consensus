from pathlib import Path

import pytest
import torch

from nerfstudio.configs.evaluation_config import (
    EditEvaluationConfig,
    EvaluationClipConfig,
    EvaluationReferenceConfig,
)
from nerfstudio.evaluation.edit_metrics import (
    compute_gaussian_delta_stats,
    summarize_per_image_rows,
    validate_metric_selection,
)
from nerfstudio.scripts.experiment import _build_trainer_config


def test_experiment_config_maps_evaluation_block():
    config = _build_trainer_config(
        {
            "method": "splatfacto",
            "name": "eval-test",
            "load_dir": "outputs/source/nerfstudio_models",
            "load_step": 123,
            "evaluation": {
                "enabled": True,
                "output_dir": "evals",
                "metrics": ["clipdir", "gaussian_scene"],
                "save_rendered_images": True,
                "clip": {
                    "prompt": "a photo of a bicycle at sunset",
                    "source_prompt": "a photo of a bicycle",
                    "model": "ViT-B/32",
                    "neighbor_count": 2,
                },
            },
        },
        suite_name=None,
        run_name="eval-test",
    )

    assert config.evaluation.enabled is True
    assert config.evaluation.output_dir == Path("evals")
    assert config.evaluation.metrics == ("clip_direction", "gaussian_scene")
    assert config.evaluation.save_rendered_images is True
    assert config.evaluation.clip.prompt == "a photo of a bicycle at sunset"
    assert config.evaluation.clip.source_prompt == "a photo of a bicycle"
    assert config.evaluation.clip.model == "ViT-B/32"
    assert config.evaluation.clip.neighbor_count == 2
    assert config.evaluation.reference.load_dir == Path("outputs/source/nerfstudio_models")
    assert config.evaluation.reference.load_step == 123


def test_metric_selection_validation_requires_prompt():
    config = EditEvaluationConfig(enabled=True, metrics=("clip_text",))

    with pytest.raises(ValueError, match="prompt"):
        validate_metric_selection(config)


def test_metric_selection_validation_requires_source_prompt():
    config = EditEvaluationConfig(
        enabled=True,
        metrics=("clip_direction",),
        clip=EvaluationClipConfig(prompt="target"),
        reference=EvaluationReferenceConfig(load_dir=Path("source")),
    )

    with pytest.raises(ValueError, match="source_prompt"):
        validate_metric_selection(config)


def test_metric_selection_validation_requires_reference():
    config = EditEvaluationConfig(enabled=True, metrics=("clip_image",))

    with pytest.raises(ValueError, match="reference checkpoint"):
        validate_metric_selection(config)


def test_metric_selection_validation_reports_missing_clip(monkeypatch):
    import builtins

    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "clip":
            raise ImportError("no clip")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    config = EditEvaluationConfig(
        enabled=True,
        metrics=("clip_text",),
        clip=EvaluationClipConfig(prompt="target"),
    )

    with pytest.raises(RuntimeError, match="OpenAI CLIP"):
        validate_metric_selection(config, require_clip_dependency=True)


def test_summarize_per_image_rows_aggregates_numeric_values():
    summary = summarize_per_image_rows(
        [
            {"image_name": "a.png", "psnr": 10.0, "l1": 0.2},
            {"image_name": "b.png", "psnr": 20.0, "l1": 0.4},
        ]
    )

    assert summary["psnr"]["mean"] == pytest.approx(15.0)
    assert summary["psnr"]["std"] == pytest.approx(5.0)
    assert summary["psnr"]["count"] == pytest.approx(2.0)
    assert summary["l1"]["max"] == pytest.approx(0.4)


def test_gaussian_delta_matching_counts():
    final = {"means": torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])}
    reference = {"means": torch.zeros((2, 3))}

    delta = compute_gaussian_delta_stats(final, reference)

    assert delta["final_count"] == 2
    assert delta["reference_count"] == 2
    assert delta["count_delta"] == 0
    assert delta["groups"]["means"]["delta_norm_mean"] == pytest.approx(1.5)


def test_gaussian_delta_mismatched_counts_skips_pairwise_delta():
    final = {"means": torch.zeros((2, 3))}
    reference = {"means": torch.zeros((3, 3))}

    delta = compute_gaussian_delta_stats(final, reference)

    assert delta["count_delta"] == -1
    assert "means" not in delta["groups"]
    assert "count mismatch" in delta["skipped_groups"]["means"]
