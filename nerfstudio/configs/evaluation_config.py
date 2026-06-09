"""Configuration for post-training edit evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class EvaluationClipConfig:
    """CLIP model and text prompt settings for edit metrics."""

    prompt: Optional[str] = None
    """Target edit prompt for CLIP text and direction metrics."""
    source_prompt: Optional[str] = None
    """Source prompt before editing, required for CLIP direction metrics."""
    model: str = "ViT-L/14"
    """OpenAI CLIP model name."""
    neighbor_count: int = 1
    """Number of pose-nearest neighboring views for CLIP direction consistency."""


@dataclass
class EvaluationReferenceConfig:
    """Reference checkpoint used for preservation and edit-direction metrics."""

    load_dir: Optional[Path] = None
    """Directory containing reference checkpoints."""
    load_step: Optional[int] = None
    """Reference checkpoint step; latest checkpoint is used when unset."""
    load_checkpoint: Optional[Path] = None
    """Optional direct reference checkpoint path."""


@dataclass
class EditEvaluationConfig:
    """Post-training edit evaluation settings."""

    enabled: bool = False
    """Whether to run post-training edit evaluation."""
    output_dir: Path = Path("evaluations")
    """Directory for evaluation outputs, relative to the experiment run directory unless absolute."""
    metrics: Tuple[str, ...] = ("reconstruction",)
    """Metrics to compute."""
    save_per_image: bool = True
    """Whether to write per-image metric rows."""
    save_rendered_images: bool = False
    """Whether to save rendered final/reference images for inspected rows."""
    clip: EvaluationClipConfig = field(default_factory=EvaluationClipConfig)
    """CLIP metric settings."""
    reference: EvaluationReferenceConfig = field(default_factory=EvaluationReferenceConfig)
    """Reference checkpoint settings."""
