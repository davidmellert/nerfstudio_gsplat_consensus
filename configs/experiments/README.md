# Experiment YAML Reference

This folder contains YAML files for `ns-experiment`. The goal is to keep the command short and move the experiment choices into a readable config file.

```bash
ns-experiment --config configs/experiments/gaussian_consensus_example.yml
```

Useful checks before a long run:

```bash
ns-experiment --config configs/experiments/gaussian_consensus_example.yml --dry-run
ns-experiment --config configs/experiments/gaussian_consensus_example.yml --dry-run --only online-8views-all-no-densify-sunset10
```

## Overall Structure

A config file has optional shared `defaults` and one or more entries under `runs`. Each run is merged on top of `defaults`, so only the changed values need to be written per run.

```yaml
suite_name: bicycle-consensus-sweep

defaults:
  method: splatfacto-colmap
  data: data/bicycle_ns
  experiment_name: bicycle_ns_metrics
  consensus:
    mode: standard
    densification: false
    trainable: all

runs:
  - name: standard-1view-all-no-densify-sunset10
    consensus:
      mode: standard
      num_views: 1

  - name: online-8views-all-no-densify-sunset10
    consensus:
      mode: online
      num_views: 8
```

Training outputs are written by Nerfstudio as:

```text
outputs/<experiment_name>/<run_name>/<timestamp>/
```

Render outputs are written as:

```text
renders/<experiment_name>/<run_name>/<timestamp>/<render-name>.mp4
```

If `experiment_name` is omitted, `suite_name` is used. If `method_name` is omitted, the run `name` is used as the Nerfstudio method/output folder name.

## Choosing What Gets Trained

Use `consensus.trainable` to choose which Gaussian optimizer groups are updated. This works in `standard`, `consensus`, and `online` modes.

```yaml
consensus:
  trainable: all
```

Available Gaussian parameter groups:

| Group | What it controls |
| --- | --- |
| `means` | Gaussian center positions |
| `scales` | Gaussian sizes |
| `quats` | Gaussian rotations |
| `features_dc` | DC/color base spherical-harmonic coefficients |
| `features_rest` | Higher-order spherical-harmonic appearance coefficients |
| `opacities` | Gaussian opacity |

Supported shorthand values:

| Value | Expands to |
| --- | --- |
| `all` or `true` | `means`, `scales`, `quats`, `features_dc`, `features_rest`, `opacities` |
| `appearance`, `color`, or `colors` | `features_dc`, `features_rest` |
| A single group name | Only that group |

Supported aliases:

| Alias | Group |
| --- | --- |
| `mean` | `means` |
| `scale` | `scales` |
| `quat`, `rotation`, `rotations` | `quats` |
| `feature_dc`, `dc` | `features_dc` |
| `feature_rest`, `rest` | `features_rest` |
| `opacity` | `opacities` |

You can also use a list:

```yaml
consensus:
  trainable: [features_dc, features_rest, opacities]
```

Or a true/false mapping:

```yaml
consensus:
  trainable:
    means: false
    scales: false
    quats: false
    features_dc: true
    features_rest: true
    opacities: true
```

At least one group must be enabled. If `densification: true`, all Gaussian groups must be trainable, because densification/culling/reset changes geometry and optimizer state.

## Training Modes

Set the mode with `consensus.mode`.

| Mode | Behavior |
| --- | --- |
| `standard` | Usual Nerfstudio/Splatfacto training: one image, render, loss, backpropagation, optimizer step. Consensus is disabled. |
| `consensus` | Multi-view consensus training with stored per-view gradients. This is the normal/non-online consensus path. |
| `online` | Multi-view consensus training with online gradient accumulation. This avoids storing all per-view gradients at once. |
| `batch` | Sequential multi-view batch averaging path. Keep this for experiments only. |

Aliases accepted by the runner:

| Alias | Mode |
| --- | --- |
| `default`, `off`, `none` | `standard` |
| `gaussian_consensus`, `stored`, `offline`, `normal` | `consensus` |
| `online_consensus` | `online` |
| `sequential_batch` | `batch` |

In `standard` mode, `num_views` is not used by the trainer, but it is still useful to set it to `1` in the run name/config for clarity.

## Densification

Densification is off by default in `ns-experiment`.

```yaml
consensus:
  densification: false
```

Options:

| Key | Values | Meaning |
| --- | --- | --- |
| `densification` | `true` or `false` | Enables/disables Splatfacto refinement callbacks such as densification, culling, and opacity reset. |
| `densify` | `true` or `false` | Alias for `densification`. |
| `densify_min_view_support` | integer | In consensus modes with densification, minimum accumulated visible-view support before a Gaussian can be split/cloned. |

When densification is enabled, use:

```yaml
consensus:
  densification: true
  trainable: all
```

## Consensus Options

These keys live under `consensus`.

| Key | Values | Meaning |
| --- | --- | --- |
| `mode` | `standard`, `consensus`, `online`, `batch` | Chooses the training path. |
| `trainable` | `all`, `appearance`, a group name, a list, or a mapping | Chooses which Gaussian parameter groups receive optimizer updates. |
| `densification` | bool | Enables/disables refinement callbacks. Default in the runner is `false`. |
| `densify` | bool | Alias for `densification`. |
| `aggregator` | `mean`, `cosine`, or `cos` | Chooses how visible per-view Gaussian gradients are combined. `cos` is normalized to `cosine`. |
| `aggregation` | `mean`, `cosine`, or `cos` | Alias for `aggregator`. |
| `accumulation` | `online` or `stored` | Advanced override for how gradients are accumulated. Normally set this by choosing `mode: online` or `mode: consensus`. |
| `implementation` | `online` or `stored` | Alias for `accumulation`. |
| `online` | bool | Backward-compatible shortcut. `true` means `accumulation: online`; `false` means `stored`. |
| `num_views` | integer >= 1 | Number of edited training views used for one consensus optimizer step. |
| `max_views_per_gaussian` | integer | Maximum visible views that can influence one Gaussian per step. `0` means all visible views. |
| `view_sampling` | `global` or `pose_neighborhood` | How extra consensus views are chosen. |
| `neighbor_pool_size` | integer | Candidate pool size for nearest-view sampling when `view_sampling: pose_neighborhood`. |
| `position_weight` | float | Weight for camera-center distance during pose-neighborhood ranking. |
| `direction_weight` | float | Weight for viewing-direction distance during pose-neighborhood ranking. |
| `min_alignment` | float | Minimum cosine agreement used by the `cosine` aggregator. `0` ignores opposing gradients. |
| `gaussian_chunk_size` | integer | Number of Gaussians aggregated at once per optimizer group. Lower values use less memory and may be slower. |
| `store_grads_on_cpu` | bool | Compatibility option for stored accumulation. Usually leave this alone. |
| `densify_min_view_support` | integer | Minimum consensus visibility support for densification when densification is enabled. |

## Common Top-Level Keys

These keys can be placed in `defaults` or in an individual run.

| Key | Meaning |
| --- | --- |
| `name` | Run name. Also becomes `method_name` unless `method_name` is set explicitly. |
| `suite_name` | Human-readable suite name. Used as fallback experiment name. |
| `method` | Nerfstudio method config key, for example `splatfacto-colmap`. |
| `base_config` | Path to an existing Nerfstudio `config.yml` to use as the starting config instead of `method`. |
| `data` | Dataset path. If a bare name does not exist but `data/<name>` exists, the runner uses `data/<name>`. |
| `experiment_name` | Top-level Nerfstudio output folder under `outputs/`. |
| `method_name` | Run/output folder under `outputs/<experiment_name>/`. Usually left unset so the run `name` is used. |
| `output_dir` | Root training output directory. Usually `outputs`. |
| `vis` | Nerfstudio visualization backend, for example `tensorboard`, `viewer`, or `viewer+tensorboard`. |
| `downscale_factor` | Image downscale factor passed to Nerfstudio. |
| `load_dir` | Checkpoint directory to resume from, for example `outputs/.../nerfstudio_models`. |
| `load_step` | Specific checkpoint step to load from `load_dir`. Leave unset for latest. |
| `load_checkpoint` | Exact checkpoint file path. |
| `relative_model_dir` | Model checkpoint subfolder name inside a run output. |
| `trainable` | Top-level fallback for `consensus.trainable`. Prefer putting it under `consensus`. |

Other native `TrainerConfig` fields can also be set at the top level if their names are not one of the special runner sections.

## Native Nerfstudio Sections

These sections pass values directly into the corresponding Nerfstudio config object. The runner checks that each field exists and raises an error for unknown field names.

| Section | Target |
| --- | --- |
| `trainer` | Top-level `TrainerConfig` fields, such as `max_num_iterations`, `steps_per_eval_batch`, `steps_per_eval_image`, `steps_per_eval_all_images`, `steps_per_save`, `save_only_latest_checkpoint`. |
| `machine` | `config.machine`, such as `num_devices`, `device_type`, `seed`. |
| `viewer` | `config.viewer`, such as `quit_on_train_completion`, `websocket_port`, `num_rays_per_chunk`. |
| `logging` | `config.logging`, such as `steps_per_log`, `profiler`. |
| `datamanager` | `config.pipeline.datamanager`, such as `cache_images`, `cache_images_type`, `dataloader_num_workers`. |
| `dataparser` | `config.pipeline.datamanager.dataparser`, usually COLMAP fields. Can also be the string `colmap`. |
| `model` | `config.pipeline.model`, such as `stop_split_at`, `background_color`, `rasterize_mode`, or any lower-level Gaussian consensus model field. |
| `overrides` | Dotted-path assignments for fields not convenient to express through the sections above. |

Example:

```yaml
trainer:
  max_num_iterations: 17000
  steps_per_eval_batch: 0
  steps_per_eval_image: 0
  steps_per_eval_all_images: 0

datamanager:
  cache_images: cpu

dataparser:
  images_path: images_edited_sunset10
  skip_missing_images: true
  auto_downscale_missing_images: false
  eval_mode: all

model:
  stop_split_at: 17000
```

Dotted overrides are useful for rare nested fields:

```yaml
overrides:
  pipeline.model.background_color: white
  pipeline.datamanager.dataparser.eval_mode: all
```

## Render Section

If `render.enabled: true`, `ns-experiment` runs `ns-render camera-path` after training finishes.

```yaml
render:
  enabled: true
  name: circle-path
  camera_path_filename: outputs/bicycle_ns/splatfacto-colmap/2026-05-27_221222/camera_paths/circle_path.json
  output_dir: renders
  output_format: video
  rendered_output_names: [rgb]
```

| Key | Values | Meaning |
| --- | --- | --- |
| `enabled` | bool | If `false`, skip rendering. If the render block exists and `enabled` is omitted, rendering is enabled. |
| `name` | string | Render file stem when `output_path` is not given. |
| `camera_path_filename` | path | Camera path JSON. Required when rendering is enabled. |
| `camera_path` | path | Alias for `camera_path_filename`. |
| `output_dir` | path | Root render folder. Default is `renders`. |
| `output_path` | path | Exact output path. Overrides the structured render folder. |
| `output_format` | string | Passed to `RenderCameraPath`; use `video` for mp4 output. |
| `image_format` | string | Image format used by the renderer, for example `jpeg`. |
| `jpeg_quality` | integer | JPEG quality. Default is `100`. |
| `downscale_factor` | float | Render downscale factor. Default is `1.0`. |
| `eval_num_rays_per_chunk` | integer or null | Render ray chunk size override. |
| `rendered_output_names` | list | Outputs to render, usually `[rgb]`. |
| `render_nearest_camera` | bool | Passed through to `ns-render camera-path`. |
| `check_occlusions` | bool | Passed through to `ns-render camera-path`. |
| `camera_idx` | integer or null | Optional camera index. |

Without `output_path`, renders are stored as:

```text
<output_dir>/<experiment_name>/<run_name>/<timestamp>/<name>.mp4
```

## Running Multiple Experiments

Run every entry in `runs`:

```bash
ns-experiment --config configs/experiments/gaussian_consensus_example.yml
```

Run only selected names:

```bash
ns-experiment --config configs/experiments/gaussian_consensus_example.yml --only standard-1view-all-no-densify-sunset10
```

Skip rendering even if a run has `render.enabled: true`:

```bash
ns-experiment --config configs/experiments/gaussian_consensus_example.yml --skip-render
```

Render only from the prepared config path without training:

```bash
ns-experiment --config configs/experiments/gaussian_consensus_example.yml --skip-train
```

Dry runs prepare and print configs but do not train or render:

```bash
ns-experiment --config configs/experiments/gaussian_consensus_example.yml --dry-run
```

## Practical Examples

Appearance-only standard edit, no densification:

```yaml
- name: standard-1view-appearance-no-densify-sunset10
  consensus:
    mode: standard
    num_views: 1
    densification: false
    trainable: appearance
```

Update opacity and color only:

```yaml
- name: online-8views-color-opacity-no-densify-sunset10
  consensus:
    mode: online
    num_views: 8
    densification: false
    trainable: [features_dc, features_rest, opacities]
```

Full online update with densification:

```yaml
- name: online-8views-all-densify-sunset10
  consensus:
    mode: online
    num_views: 8
    densification: true
    trainable: all
```

Stored consensus with 16 images:

```yaml
- name: consensus-16views-all-no-densify-sunset10
  consensus:
    mode: consensus
    num_views: 16
    densification: false
    trainable: all
```
