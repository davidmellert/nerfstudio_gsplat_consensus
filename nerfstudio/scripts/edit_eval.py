#!/usr/bin/env python
"""CLI for configurable edit evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import tyro

from nerfstudio.evaluation.edit_metrics import evaluate_edit_metrics


@dataclass
class ComputeEditMetrics:
    """Load a saved config and compute its configured edit evaluation metrics."""

    load_config: Path
    """Path to a saved Nerfstudio config.yml."""
    output_dir: Optional[Path] = None
    """Optional output directory override."""
    eval_num_rays_per_chunk: Optional[int] = None
    """Optional ray chunk override for rendering eval images."""

    def main(self) -> None:
        evaluate_edit_metrics(
            self.load_config,
            output_dir=self.output_dir,
            eval_num_rays_per_chunk=self.eval_num_rays_per_chunk,
        )


def entrypoint() -> None:
    """Entrypoint for use with pyproject scripts."""

    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(ComputeEditMetrics).main()


if __name__ == "__main__":
    entrypoint()


get_parser_fn = lambda: tyro.extras.get_parser(ComputeEditMetrics)  # noqa
