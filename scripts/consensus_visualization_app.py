#!/usr/bin/env python
"""Standalone Streamlit app for Gaussian consensus visualization snapshots.

Run from a separate viewer environment, for example:

    streamlit run scripts/consensus_visualization_app.py -- outputs

The argument may be a broad outputs directory, one experiment/run directory, one timestamped run directory,
or a consensus_visualizations directory.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image


MAX_SCATTER_POINTS = 100_000

MAP_DESCRIPTIONS = {
    "current_rgb": "The model render from the anchor camera before the optimizer step is applied.",
    "target_rgb_after_step": "The model render from the same anchor/target camera after the optimizer step is applied.",
    "view_update_norm": "Magnitude of the update proxy from one sampled view. This answers which image is pulling on which rendered regions.",
    "visible_view_count": "How many sampled views have a nonzero update for each rendered region. A value of 1 means only one sampled view contributed there.",
    "mean_update_norm": "Magnitude of the visible-view mean update before the consensus weighting is applied.",
    "final_update_norm": "Magnitude of the final consensus update that will be assigned to the Gaussian gradients for the optimizer step.",
    "agreement": "How much the sampled views agree on update direction. 1 means directions are highly aligned; 0 means no active/consistent update.",
    "disagreement": "The inverse of agreement. Bright regions are places where views conflict or cancel each other.",
    "dominant_view": "Categorical view id with the largest update norm for each rendered Gaussian region. 0 is the anchor, 1+ are references.",
    "dominance_strength": "Largest per-view update norm divided by the sum over views. Near 1 means one view dominates; lower means views contribute more evenly.",
    "suppression_ratio": "Final consensus update norm divided by mean update norm. Lower values indicate stronger consensus suppression.",
    "rgb_update_mean": "Signed RGB update proxy from features_dc before consensus weighting. Neutral gray means little color change.",
    "rgb_update_final": "Signed RGB update proxy from features_dc after consensus weighting. Compare to rgb_update_mean to see what consensus kept or suppressed.",
    "opacity_update_mean": "Signed opacity update proxy before consensus weighting. Warm/red means opacity increase; cool/blue means opacity decrease.",
    "opacity_update_final": "Signed opacity update proxy after consensus weighting. Useful for seeing if consensus suppresses holes, floaters, or ghosting updates.",
}


def _initial_path() -> str:
    for arg in sys.argv[1:]:
        if not arg.startswith("-"):
            return arg
    return ""


def _strip_prefix(value: str, prefix: str) -> str:
    return value[len(prefix) :] if value.startswith(prefix) else value


def _find_visualization_roots(path: Path) -> List[Path]:
    if path.is_file() and path.name == "index.json":
        path = path.parent
    if not path.exists():
        return []

    roots: List[Path] = []
    if (path / "index.json").exists():
        roots.append(path)

    direct_child = path / "consensus_visualizations"
    if (direct_child / "index.json").exists():
        roots.append(direct_child)

    if path.is_dir():
        roots.extend(index_path.parent for index_path in path.rglob("consensus_visualizations/index.json"))

    unique_roots = {root.resolve(): root for root in roots}
    return sorted(unique_roots.values(), key=lambda root: str(root))


def _visualization_root_label(root: Path, search_root: Path) -> str:
    try:
        label_path = root.relative_to(search_root)
    except ValueError:
        label_path = root
    if label_path.name == "consensus_visualizations":
        label_path = label_path.parent
    label = str(label_path)
    if label in {"", "."}:
        label = root.parent.name if root.name == "consensus_visualizations" else root.name
    return label


@st.cache_data(show_spinner=False)
def _load_json(path: str, mtime: float) -> Dict:
    del mtime
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def _load_npz(path: str, mtime: float) -> Dict[str, np.ndarray]:
    del mtime
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def _capture_window_start(capture: Dict) -> int:
    return int(capture.get("window_start", capture.get("step", 0)))


def _capture_step(capture: Dict) -> int:
    return int(capture.get("step", 0))


def _capture_step_dir(root: Path, capture: Dict) -> Path:
    return root / str(capture["path"])


def _load_manifest_for_capture(root: Path, capture: Dict) -> Dict:
    manifest_path = _capture_step_dir(root, capture) / "manifest.json"
    return _load_json(str(manifest_path), manifest_path.stat().st_mtime)


def _load_snapshot_for_capture(root: Path, capture: Dict) -> Dict[str, np.ndarray]:
    step_dir = _capture_step_dir(root, capture)
    manifest = _load_manifest_for_capture(root, capture)
    snapshot_file = manifest.get("snapshot_file")
    if not snapshot_file:
        return {}
    snapshot_path = step_dir / snapshot_file
    if not snapshot_path.exists():
        return {}
    return _load_npz(str(snapshot_path), snapshot_path.stat().st_mtime)


def _window_label(start: int, captures: Sequence[Dict]) -> str:
    steps = [_capture_step(capture) for capture in captures if _capture_window_start(capture) == start]
    if len(steps) <= 1:
        return f"{start:09d}"
    return f"{min(steps):09d}-{max(steps):09d}"


def _arrays_for_map(root: Path, captures: Sequence[Dict], map_key: str) -> List[np.ndarray]:
    arrays = []
    for capture in captures:
        snapshot = _load_snapshot_for_capture(root, capture)
        if map_key in snapshot:
            arrays.append(np.squeeze(snapshot[map_key]))
    return arrays


def _map_description(name: str) -> str:
    if name.startswith("view_update_norm_"):
        return MAP_DESCRIPTIONS["view_update_norm"]
    return MAP_DESCRIPTIONS.get(name, "No description available for this map yet.")


def _is_signed_map(name: str) -> bool:
    return "rgb_update" in name or "opacity_update" in name


def _is_update_norm_map(name: str) -> bool:
    return name.startswith("view_update_norm_") or name in {"mean_update_norm", "final_update_norm"}


def _static_scale_for_map(name: str, arrays: Sequence[np.ndarray]) -> Tuple[Optional[Tuple[float, float]], Optional[float]]:
    if not arrays:
        return None, None
    finite_values = []
    for array in arrays:
        values = np.asarray(array, dtype=np.float32)
        finite = values[np.isfinite(values)]
        if finite.size:
            finite_values.append(finite.reshape(-1))
    if not finite_values:
        return None, None
    values = np.concatenate(finite_values)

    if name in {"agreement", "disagreement"}:
        lower = float(np.min(values)) if values.size else 0.0
        upper = float(np.max(values)) if values.size else 1.0
        upper = max(upper, lower + 1e-8)
        return (lower, upper), None
    if name in {"dominance_strength", "suppression_ratio"}:
        return (0.0, 1.0), None
    if name in {"dominant_view", "visible_view_count"}:
        return (0.0, max(1.0, float(np.max(values)))), None
    if _is_signed_map(name):
        scale = float(np.max(np.abs(values))) if values.size else 1.0
        scale = max(scale, 1e-8)
        return (-scale, scale), scale
    if _is_update_norm_map(name):
        upper = float(np.max(values)) if values.size else 1.0
        return (0.0, max(upper, 1e-8)), None

    upper = float(np.percentile(values, 99.0))
    if upper <= 0.0:
        upper = float(np.max(values)) if values.size else 1.0
    return (0.0, max(upper, 1e-8)), None


def _signed_to_rgb(array: np.ndarray, scale: Optional[float] = None) -> np.ndarray:
    array = np.nan_to_num(array.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if scale is None:
        scale = float(np.percentile(np.abs(array), 99.0)) if array.size else 1.0
        if scale <= 0.0:
            scale = float(np.max(np.abs(array))) if array.size else 1.0
    scale = max(float(scale), 1e-8)
    if array.ndim == 2:
        signed = np.clip(array / scale, -1.0, 1.0)
        magnitude = np.abs(signed)
        return np.stack(
            [
                np.where(signed > 0, magnitude, 0.0),
                1.0 - magnitude,
                np.where(signed < 0, magnitude, 0.0),
            ],
            axis=-1,
        )
    return np.clip(array / (2.0 * scale) + 0.5, 0.0, 1.0)


def _display_map(
    name: str,
    array: np.ndarray,
    default_range: Optional[Tuple[float, float]] = None,
    signed_scale: Optional[float] = None,
) -> None:
    array = np.squeeze(array)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        st.warning("Selected map has no finite values.")
        return

    if array.ndim == 3 and array.shape[-1] == 3:
        signed_default = bool(np.nanmin(array) < 0.0 or np.nanmax(array) > 1.0 or _is_signed_map(name))
        signed = st.checkbox("Signed RGB normalization", value=signed_default)
        image = _signed_to_rgb(array, signed_scale) if signed else np.clip(array, 0.0, 1.0)
        st.image(image, caption=name, use_container_width=True)
        if signed and signed_scale is not None:
            st.caption(f"Static signed scale: +/- {signed_scale:.4g}")
        return

    if array.ndim != 2:
        st.write(f"Array shape: {array.shape}")
        st.dataframe(array.reshape(-1)[:1000])
        return

    full_min = float(np.min(finite))
    full_max = float(np.max(finite))
    if default_range is None:
        default_min = float(np.percentile(finite, 1.0))
        default_max = float(np.percentile(finite, 99.0))
    else:
        default_min, default_max = default_range
        full_min = min(full_min, default_min)
        full_max = max(full_max, default_max)
    if default_min == default_max:
        default_min, default_max = full_min, full_max
    if full_min == full_max:
        full_min -= 1.0
        full_max += 1.0
        default_min, default_max = full_min, full_max

    zmin, zmax = st.slider(
        "Color range",
        min_value=full_min,
        max_value=full_max,
        value=(default_min, default_max),
    )
    colorscale = "RdBu" if "opacity_update" in name else "Turbo"
    fig = px.imshow(array, color_continuous_scale=colorscale, zmin=zmin, zmax=zmax, origin="upper")
    fig.update_layout(margin=dict(l=0, r=0, t=24, b=0), height=650)
    st.plotly_chart(fig, use_container_width=True)


def _scatter_data(snapshot: Dict[str, np.ndarray]) -> Optional[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]:
    x = snapshot.get("gaussian__mean_update_norm")
    y = snapshot.get("gaussian__disagreement")
    if x is None or y is None:
        return None
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    color = snapshot.get("gaussian__dominant_view")
    color = np.asarray(color).reshape(-1) if color is not None else None
    valid = np.isfinite(x) & np.isfinite(y)
    if color is not None:
        valid = valid & np.isfinite(color)
    x = x[valid]
    y = y[valid]
    color = color[valid] if color is not None else None
    if x.size > MAX_SCATTER_POINTS:
        rng = np.random.default_rng(0)
        indices = rng.choice(x.size, size=MAX_SCATTER_POINTS, replace=False)
        x = x[indices]
        y = y[indices]
        color = color[indices] if color is not None else None
    return x, y, color


def main() -> None:
    st.set_page_config(page_title="Consensus Visualizations", layout="wide")
    st.title("Gaussian Consensus Visualizations")

    with st.sidebar:
        run_dir_text = st.text_input("Output, run, or visualization directory", value=_initial_path())
        if not run_dir_text:
            st.info("Pass an outputs/run directory after `--`, or enter one here.")
            st.stop()
        run_dir = Path(run_dir_text).expanduser()
        visualization_roots = _find_visualization_roots(run_dir)
        if not visualization_roots:
            st.error("No consensus_visualizations/index.json found under that path.")
            st.stop()

        if len(visualization_roots) == 1:
            visualization_root = visualization_roots[0]
        else:
            root_labels = [
                f"{idx + 1}. {_visualization_root_label(root, run_dir)}"
                for idx, root in enumerate(visualization_roots)
            ]
            selected_root_label = st.selectbox("Visualization folder", root_labels, index=len(root_labels) - 1)
            visualization_root = visualization_roots[root_labels.index(selected_root_label)]
        st.caption(str(visualization_root))

    index_path = visualization_root / "index.json"
    index = _load_json(str(index_path), index_path.stat().st_mtime)
    captures = index.get("captures", [])
    if not captures:
        st.warning("No captured steps found in index.json.")
        st.stop()

    window_starts = sorted({_capture_window_start(capture) for capture in captures})
    window_labels = [_window_label(start, captures) for start in window_starts]
    selected_window_label = st.sidebar.selectbox("Capture window", window_labels, index=len(window_labels) - 1)
    selected_window_start = window_starts[window_labels.index(selected_window_label)]
    window_captures = [capture for capture in captures if _capture_window_start(capture) == selected_window_start]
    window_captures.sort(key=_capture_step)

    step_labels = [f"{_capture_step(capture):09d}" for capture in window_captures]
    selected_label = st.sidebar.selectbox("Step in window", step_labels, index=len(step_labels) - 1)
    selected_capture = window_captures[step_labels.index(selected_label)]
    scale_scope = st.sidebar.selectbox("Default map scale", ["Selected window", "All captures", "Current step"], index=0)
    scale_captures = {
        "Selected window": window_captures,
        "All captures": captures,
        "Current step": [selected_capture],
    }[scale_scope]

    step_dir = _capture_step_dir(visualization_root, selected_capture)
    manifest = _load_manifest_for_capture(visualization_root, selected_capture)

    st.subheader(f"Step {manifest['step']}")
    summary = manifest.get("summary", {})
    metric_names = ["mean_update_norm", "final_update_norm", "disagreement", "dominance_strength", "suppression_ratio"]
    cols = st.columns(len(metric_names))
    for col, name in zip(cols, metric_names):
        value = summary.get(name)
        col.metric(name, "n/a" if value is None else f"{value:.4g}")

    with st.expander("Capture Metadata", expanded=False):
        st.json({key: value for key, value in manifest.items() if key != "panels"})

    preview_path = step_dir / manifest.get("preview_dashboard", "preview_dashboard.png")
    if preview_path.exists():
        st.image(_load_image(preview_path), caption="Preview dashboard", use_container_width=True)
        st.download_button(
            "Download preview dashboard",
            data=preview_path.read_bytes(),
            file_name=preview_path.name,
            mime="image/png",
        )

    snapshot = _load_snapshot_for_capture(visualization_root, selected_capture)

    tab_maps, tab_inputs, tab_scatter, tab_panels, tab_guide = st.tabs(
        ["Maps", "Inputs", "Scatter", "PNG Panels", "Map Guide"]
    )

    with tab_maps:
        map_keys = sorted(key for key in snapshot.keys() if key.startswith("map__"))
        if not map_keys:
            st.info("No raw map arrays found in snapshot.npz.")
        else:
            map_name = st.selectbox("Map", [_strip_prefix(key, "map__") for key in map_keys])
            st.info(_map_description(map_name))
            map_key = f"map__{map_name}"
            comparison_arrays = _arrays_for_map(visualization_root, scale_captures, map_key)
            default_range, signed_scale = _static_scale_for_map(map_name, comparison_arrays)
            st.caption(f"Default scale scope: {scale_scope}")
            _display_map(map_name, snapshot[map_key], default_range=default_range, signed_scale=signed_scale)

    with tab_inputs:
        input_keys = sorted(key for key in snapshot.keys() if key.startswith("input__"))
        if input_keys:
            cols = st.columns(min(len(input_keys), 4))
            for idx, key in enumerate(input_keys):
                cols[idx % len(cols)].image(snapshot[key], caption=_strip_prefix(key, "input__"), use_container_width=True)
        else:
            panels = manifest.get("panels", {})
            input_panels = [(name, rel_path) for name, rel_path in panels.items() if name.startswith("input_")]
            if not input_panels:
                st.info("No input images found.")
            cols = st.columns(max(1, min(len(input_panels), 4)))
            for idx, (name, rel_path) in enumerate(input_panels):
                cols[idx % len(cols)].image(_load_image(step_dir / rel_path), caption=name, use_container_width=True)

    with tab_scatter:
        scatter = _scatter_data(snapshot)
        if scatter is None:
            st.info("Scatter data requires gaussian__mean_update_norm and gaussian__disagreement in snapshot.npz.")
        else:
            x, y, color = scatter
            marker = dict(size=3, opacity=0.35)
            if color is not None:
                marker.update(color=color, colorscale="Turbo", showscale=True)
            fig = go.Figure(data=go.Scattergl(x=x, y=y, mode="markers", marker=marker))
            fig.update_layout(
                xaxis_title="mean_update_norm",
                yaxis_title="disagreement",
                margin=dict(l=0, r=0, t=20, b=0),
                height=650,
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab_panels:
        panels = manifest.get("panels", {})
        if not panels:
            st.info("No PNG panels listed in manifest.json.")
        else:
            panel_name = st.selectbox("Panel", sorted(panels.keys()))
            panel_path = step_dir / panels[panel_name]
            st.image(_load_image(panel_path), caption=panel_name, use_container_width=True)
            st.download_button(
                "Download selected panel",
                data=panel_path.read_bytes(),
                file_name=panel_path.name,
                mime="image/png",
            )

    with tab_guide:
        st.markdown(
            """
### How To Read These Maps

All update maps use an update proxy, `-learning_rate * gradient`. This is the direction the parameter would move under a simple SGD-style step. It is not an exact Adam update, but it gives the right sign and relative magnitude for debugging consensus behavior.

- `input_anchor`, `input_ref_*`: the edited training images sampled for this consensus step. The anchor is the camera used for rendering the maps.
- `view_update_norm_XX`: how strongly view `XX` wants to update each rendered Gaussian region. All update-norm maps default to the same `0..max` scale for the selected comparison scope.
- `visible_view_count`: how many sampled views had nonzero update norm for the rendered region. Use this to check whether a region is driven by only one view.
- `mean_update_norm`: the average visible-view update magnitude before consensus suppresses or reweights it.
- `final_update_norm`: the actual consensus update magnitude assigned back to the Gaussian parameter gradients before the optimizer step.
- `agreement`: directional consistency across sampled views. High agreement means the views want similar updates.
- `disagreement`: `1 - agreement`. High disagreement means the views are pulling in conflicting or canceling directions.
- `dominant_view`: the sampled view with the largest update norm. `0` is the anchor, `1+` are reference views.
- `dominance_strength`: how much the dominant view controls the update, computed as max view norm divided by sum of view norms.
- `suppression_ratio`: final update norm divided by mean update norm. Low values show where consensus suppressed an otherwise strong update.
- `rgb_update_mean`: signed color update from `features_dc` before final consensus weighting. Gray means near zero; color shifts show the intended RGB direction.
- `rgb_update_final`: signed color update after consensus weighting. Compare this with `rgb_update_mean` to see which color changes survived.
- `opacity_update_mean`: signed opacity update before final consensus weighting. Warm/red increases opacity; cool/blue decreases opacity.
- `opacity_update_final`: signed opacity update after consensus weighting.
- `current_rgb`: model render from the anchor camera before the optimizer step.
- `target_rgb_after_step`: model render from the same anchor/target camera after the optimizer step.

`features_rest` contributes to norm and agreement maps, but the first implementation does not project higher-order SH gradients into a signed color-effect map.
            """
        )


if __name__ == "__main__":
    main()
