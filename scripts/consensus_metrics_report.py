#!/usr/bin/env python
"""Summarize and plot Gaussian consensus diagnostics.

This reads Nerfstudio TensorBoard event files or a CSV export and extracts the
consensus metrics emitted by ``Trainer.gaussian_consensus_train_iteration``.

Examples:
    python nerfstudio_gsplat_consensus/scripts/consensus_metrics_report.py \
        outputs/splatfacto-gaussian-consensus-colmap/2026-05-28_090739 --out-dir /tmp/consensus_report

    python nerfstudio_gsplat_consensus/scripts/consensus_metrics_report.py wandb_export.csv --num-views 4

For true per-iteration curves, train with ``--logging.steps-per-log 1`` and a
persistent writer such as ``--vis tensorboard`` or ``--vis wandb``.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


CONSENSUS_METRICS = (
    "Consensus/features_dc_avg_visible_views",
    "Consensus/features_dc_active_fraction",
    "Consensus/features_dc_updated_fraction",
    "Consensus/features_rest_avg_visible_views",
    "Consensus/features_rest_active_fraction",
    "Consensus/features_rest_updated_fraction",
)
NUM_VIEWS_METRIC = "Consensus/num_views"

ALL_METRICS = CONSENSUS_METRICS + (NUM_VIEWS_METRIC,)

MetricSeries = Dict[str, Dict[int, float]]


def _parse_float(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        parsed = float(value)
    else:
        value_str = str(value).strip()
        if not value_str:
            return None
        try:
            parsed = float(value_str)
        except ValueError:
            return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def _match_metric(name: str, metrics: Sequence[str] = ALL_METRICS) -> Optional[str]:
    normalized = name.strip()
    for metric in metrics:
        if normalized == metric or normalized.endswith("/" + metric) or normalized.endswith(metric):
            return metric
    return None


def _find_step_column(fieldnames: Sequence[str]) -> Optional[str]:
    candidates = {"step", "global_step", "global step", "_step", "trainer/global_step"}
    for fieldname in fieldnames:
        if fieldname.strip().lower() in candidates:
            return fieldname
    return None


def _read_csv_metrics(path: Path) -> MetricSeries:
    series: MetricSeries = {metric: {} for metric in ALL_METRICS}
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise RuntimeError(f"CSV file has no header: {path}")

        step_column = _find_step_column(reader.fieldnames)
        metric_columns = {
            column: matched
            for column in reader.fieldnames
            if (matched := _match_metric(column)) is not None
        }
        if not metric_columns:
            expected = ", ".join(CONSENSUS_METRICS)
            raise RuntimeError(f"No consensus metric columns found in {path}. Expected columns ending with: {expected}")

        for row_idx, row in enumerate(reader):
            if step_column is None:
                step = row_idx
            else:
                step_value = _parse_float(row.get(step_column))
                if step_value is None:
                    continue
                step = int(step_value)

            for column, metric in metric_columns.items():
                value = _parse_float(row.get(column))
                if value is not None:
                    series[metric][step] = value
    return series


def _event_files_from_input(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    return sorted(candidate for candidate in path.rglob("events.out.tfevents*") if candidate.is_file())


def _read_tensorboard_metrics(path: Path) -> MetricSeries:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError as exc:
        raise RuntimeError(
            "TensorBoard is required to read event files. Install tensorboard or pass a CSV export instead."
        ) from exc

    event_files = _event_files_from_input(path)
    if not event_files:
        raise RuntimeError(f"No TensorBoard event files found under: {path}")

    series: MetricSeries = {metric: {} for metric in ALL_METRICS}
    found_tags: List[str] = []

    for event_file in event_files:
        accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
        accumulator.Reload()
        for tag in accumulator.Tags().get("scalars", []):
            metric = _match_metric(tag)
            if metric is None:
                continue
            found_tags.append(tag)
            for scalar_event in accumulator.Scalars(tag):
                series[metric][int(scalar_event.step)] = float(scalar_event.value)

    if not found_tags:
        expected = ", ".join(CONSENSUS_METRICS)
        raise RuntimeError(f"No consensus scalar tags found in event files. Expected tags ending with: {expected}")
    return series


def read_metrics(input_path: Path) -> MetricSeries:
    if input_path.suffix.lower() == ".csv":
        return _read_csv_metrics(input_path)
    return _read_tensorboard_metrics(input_path)


def _all_steps(series: Mapping[str, Mapping[int, float]], metrics: Sequence[str] = CONSENSUS_METRICS) -> List[int]:
    steps = set()
    for metric in metrics:
        steps.update(series.get(metric, {}).keys())
    return sorted(steps)


def _values(series: Mapping[str, Mapping[int, float]], metric: str) -> List[float]:
    return [value for _, value in sorted(series.get(metric, {}).items())]


def _latest(series: Mapping[str, Mapping[int, float]], metric: str) -> Optional[Tuple[int, float]]:
    values = series.get(metric, {})
    if not values:
        return None
    step = max(values)
    return step, values[step]


def _series_at_steps(series: Mapping[str, Mapping[int, float]], metric: str, steps: Sequence[int]) -> List[float]:
    metric_values = series.get(metric, {})
    return [metric_values.get(step, math.nan) for step in steps]


def _median_or_none(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return statistics.median(values)


def _mean_or_none(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return statistics.fmean(values)


def _format_optional(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.6g}"


def infer_num_views(series: Mapping[str, Mapping[int, float]], override: Optional[float]) -> Optional[float]:
    if override is not None:
        return override
    values = _values(series, NUM_VIEWS_METRIC)
    if values:
        return values[-1]
    return None


def write_csv(series: Mapping[str, Mapping[int, float]], out_path: Path) -> None:
    steps = _all_steps(series)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", *CONSENSUS_METRICS])
        for step in steps:
            row = [step]
            for metric in CONSENSUS_METRICS:
                value = series.get(metric, {}).get(step)
                row.append("" if value is None else f"{value:.10g}")
            writer.writerow(row)


def build_report(
    series: Mapping[str, Mapping[int, float]],
    num_views: Optional[float],
    batch_like_fraction: float,
    low_updated_fraction: float,
    low_active_fraction: float,
) -> str:
    lines: List[str] = []
    steps = _all_steps(series)
    lines.append("Gaussian Consensus Diagnostic Report")
    lines.append("=" * 37)
    lines.append(f"Logged steps: {len(steps)}")
    if steps:
        lines.append(f"Step range: {steps[0]}..{steps[-1]}")
    lines.append(f"Configured/inferred num_views: {_format_optional(num_views)}")
    lines.append("")
    lines.append("Metric summary")
    lines.append("--------------")

    for metric in CONSENSUS_METRICS:
        values = _values(series, metric)
        latest = _latest(series, metric)
        latest_text = "n/a" if latest is None else f"{latest[1]:.6g} at step {latest[0]}"
        lines.append(
            f"{metric}: latest={latest_text}, "
            f"mean={_format_optional(_mean_or_none(values))}, "
            f"median={_format_optional(_median_or_none(values))}, "
            f"min={_format_optional(min(values) if values else None)}, "
            f"max={_format_optional(max(values) if values else None)}"
        )

    lines.append("")
    lines.append("Checks")
    lines.append("------")
    for group in ("features_dc", "features_rest"):
        avg_metric = f"Consensus/{group}_avg_visible_views"
        active_metric = f"Consensus/{group}_active_fraction"
        updated_metric = f"Consensus/{group}_updated_fraction"
        avg_latest = _latest(series, avg_metric)
        active_latest = _latest(series, active_metric)
        updated_latest = _latest(series, updated_metric)

        if avg_latest is None or active_latest is None or updated_latest is None:
            lines.append(f"{group}: missing one or more diagnostics.")
            continue

        avg_visible = avg_latest[1]
        active_fraction = active_latest[1]
        updated_fraction = updated_latest[1]
        visible_ratio = None if not num_views or num_views <= 0 else avg_visible / num_views

        status_parts = []
        if visible_ratio is not None and visible_ratio >= batch_like_fraction and updated_fraction >= batch_like_fraction:
            status_parts.append("batch-like: most Gaussians see nearly all views and survive the cosine filter")
        if updated_fraction <= low_updated_fraction:
            status_parts.append("over-filtered: cosine weighting is suppressing many Gaussian updates")
        if active_fraction <= low_active_fraction:
            status_parts.append("inactive: few Gaussians have nonzero gradients")
        if not status_parts:
            status_parts.append("mixed: not clearly batch-like or over-filtered by the configured thresholds")

        ratio_text = "n/a" if visible_ratio is None else f"{visible_ratio:.3f}"
        lines.append(
            f"{group}: avg_visible={avg_visible:.6g}, visible/num_views={ratio_text}, "
            f"active={active_fraction:.6g}, updated={updated_fraction:.6g} -> {'; '.join(status_parts)}"
        )

    rest_active = _latest(series, "Consensus/features_rest_active_fraction")
    if rest_active is not None and rest_active[1] <= low_active_fraction:
        lines.append(
            "features_rest warning: active_fraction is near zero, so those stored gradients are likely buying little."
        )

    lines.append("")
    lines.append("Interpretation note")
    lines.append("-------------------")
    lines.append(
        "These diagnostics test the current implementation, which infers per-Gaussian visibility from raw parameter "
        "gradient norm. They do not prove real rasterizer contribution visibility. If the curves look batch-like, "
        "the cosine consensus is probably behaving like ordinary multi-view batch training for these groups."
    )
    return "\n".join(lines) + "\n"


def write_report(report: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)


def plot_metrics(
    series: Mapping[str, Mapping[int, float]],
    out_path: Path,
    num_views: Optional[float],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to write the PNG graph.") from exc

    steps = _all_steps(series)
    if not steps:
        raise RuntimeError("No consensus metric steps available to plot.")

    colors = {"features_dc": "#1f77b4", "features_rest": "#d62728"}
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, constrained_layout=True)
    fig.suptitle("Gaussian Consensus Diagnostics")

    plot_specs = (
        ("avg_visible_views", "Average visible views", None),
        ("active_fraction", "Active fraction", (0.0, 1.05)),
        ("updated_fraction", "Updated fraction after cosine filter", (0.0, 1.05)),
    )

    for axis, (suffix, title, ylim) in zip(axes, plot_specs):
        for group in ("features_dc", "features_rest"):
            metric = f"Consensus/{group}_{suffix}"
            values = _series_at_steps(series, metric, steps)
            if all(math.isnan(value) for value in values):
                continue
            axis.plot(steps, values, label=group, color=colors[group], linewidth=1.8)

        if suffix == "avg_visible_views" and num_views is not None:
            axis.axhline(num_views, color="#444444", linestyle="--", linewidth=1.2, label="num_views")
        axis.set_title(title)
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best")
        if ylim is not None:
            axis.set_ylim(*ylim)

    axes[-1].set_xlabel("Training step")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def default_out_dir(input_path: Path) -> Path:
    if input_path.is_dir():
        return input_path / "consensus_metrics_report"
    return input_path.parent / f"{input_path.stem}_consensus_metrics_report"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "input",
        type=Path,
        help="Run directory containing TensorBoard events, a single event file, or a CSV export.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for consensus_metrics.csv, consensus_metrics_summary.txt, and consensus_metrics.png.",
    )
    parser.add_argument(
        "--num-views",
        type=float,
        default=None,
        help="Override/inject batch size if Consensus/num_views is not present in the logs.",
    )
    parser.add_argument(
        "--batch-like-fraction",
        type=float,
        default=0.9,
        help="Threshold for calling visible/update fractions batch-like.",
    )
    parser.add_argument(
        "--low-updated-fraction",
        type=float,
        default=0.25,
        help="Threshold for warning that cosine filtering suppresses many updates.",
    )
    parser.add_argument(
        "--low-active-fraction",
        type=float,
        default=0.05,
        help="Threshold for warning that a parameter group is mostly inactive.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    input_path = args.input.expanduser().resolve()
    if not input_path.exists():
        print(f"Input does not exist: {input_path}", file=sys.stderr)
        return 2

    out_dir = (args.out_dir or default_out_dir(input_path)).expanduser().resolve()
    try:
        series = read_metrics(input_path)
        num_views = infer_num_views(series, args.num_views)

        csv_path = out_dir / "consensus_metrics.csv"
        report_path = out_dir / "consensus_metrics_summary.txt"
        plot_path = out_dir / "consensus_metrics.png"

        write_csv(series, csv_path)
        report = build_report(
            series=series,
            num_views=num_views,
            batch_like_fraction=args.batch_like_fraction,
            low_updated_fraction=args.low_updated_fraction,
            low_active_fraction=args.low_active_fraction,
        )
        write_report(report, report_path)
        plot_metrics(series, plot_path, num_views)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(report)
    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote report: {report_path}")
    print(f"Wrote graph: {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
