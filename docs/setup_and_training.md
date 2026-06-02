# Setup and Training Guide

## Environment Setup (Remote GPU Machine)

### Prerequisites

- NVIDIA GPU with CUDA support
- Check your CUDA version: `nvidia-smi`
- Conda installed

### Create Environment

```bash
conda create -n ns-gsplat-consensus python=3.10 -y
conda activate ns-gsplat-consensus
```

### Install PyTorch with CUDA

For CUDA 11.8:

```bash
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
```

For CUDA 12.1 instead:

```bash
pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit
```

### Install Nerfstudio (from this repo)

```bash
git clone git@github.com:davidmellert/nerfstudio_gsplat_consensus.git
cd nerfstudio_gsplat_consensus
pip install -e .
```

### Optional: tiny-cuda-nn (not needed for splatfacto methods)

```bash
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

## Training

### Step 1: Train a Baseline Scene from Scratch

```bash
DATA=/path/to/your/data  # e.g., data/bicycle_ns

ns-train splatfacto-colmap \
  --data "$DATA" \
  --downscale-factor 8 \
  --max-num-iterations 15000
```

This produces a checkpoint directory under `outputs/`. Set it for subsequent steps:

```bash
CKPT_DIR=outputs/<your_scene>/splatfacto-colmap/<timestamp>/nerfstudio_models
```

### Step 2: Fine-tune on Edited Images

All fine-tuning commands load from the baseline checkpoint (`--load-dir`) and train on edited images (`--images-path`).

#### Single-View Batch Baseline (1 view, no densification)

```bash
ns-train splatfacto-gaussian-batch-colmap \
  --data "$DATA" \
  --downscale-factor 8 \
  --load-dir "$CKPT_DIR" \
  --max-num-iterations 17000 \
  --pipeline.datamanager.cache-images cpu \
  --pipeline.model.gaussian-consensus-num-views 1 \
  colmap --images-path images_edited \
  --skip-missing-images True \
  --auto-downscale-missing-images False \
  --eval-mode all
```

#### Consensus (4 views, no densification)

```bash
ns-train splatfacto-gaussian-consensus-colmap \
  --data "$DATA" \
  --downscale-factor 8 \
  --load-dir "$CKPT_DIR" \
  --max-num-iterations 17000 \
  --pipeline.datamanager.cache-images cpu \
  --pipeline.model.gaussian-consensus-num-views 4 \
  --pipeline.model.gaussian-consensus-max-views-per-gaussian 4 \
  colmap --images-path images_edited \
  --skip-missing-images True \
  --auto-downscale-missing-images False \
  --eval-mode all
```

#### Consensus (8 views, no densification)

```bash
ns-train splatfacto-gaussian-consensus-colmap \
  --data "$DATA" \
  --downscale-factor 8 \
  --load-dir "$CKPT_DIR" \
  --max-num-iterations 16000 \
  --pipeline.datamanager.cache-images cpu \
  --pipeline.model.gaussian-consensus-num-views 8 \
  --pipeline.model.gaussian-consensus-max-views-per-gaussian 8 \
  colmap --images-path images_edited \
  --skip-missing-images True \
  --auto-downscale-missing-images False \
  --eval-mode all
```

#### Batch with Densification (4 views)

```bash
ns-train splatfacto-gaussian-batch-densify-colmap \
  --data "$DATA" \
  --downscale-factor 8 \
  --load-dir "$CKPT_DIR" \
  --max-num-iterations 17000 \
  --pipeline.datamanager.cache-images cpu \
  --pipeline.model.gaussian-consensus-num-views 4 \
  --pipeline.model.stop-split-at 20000 \
  colmap --images-path images_edited \
  --skip-missing-images True \
  --auto-downscale-missing-images False \
  --eval-mode all
```

### Step 3: Render Videos

Render a camera flyaround from any trained model:

```bash
ns-render camera-path \
  --load-config outputs/<your_scene>/<method>/<timestamp>/config.yml \
  --camera-path-filename docs/assets/circle_path.json \
  --output-path renders/my_render.mp4
```

A default `circle_path.json` is included in the repo at `docs/assets/`. You can also create custom camera paths in the nerfstudio web viewer.

## Optional: DC Regularization

Add an L2 anchor loss on `features_dc` to resist inconsistent color changes during fine-tuning. Append these flags to any fine-tuning command:

```bash
--pipeline.model.dc-regularization-enabled True \
--pipeline.model.dc-regularization-weight 0.01
```

Example with consensus:

```bash
ns-train splatfacto-gaussian-consensus-colmap \
  --data "$DATA" \
  --downscale-factor 8 \
  --load-dir "$CKPT_DIR" \
  --max-num-iterations 17000 \
  --pipeline.datamanager.cache-images cpu \
  --pipeline.model.gaussian-consensus-num-views 4 \
  --pipeline.model.gaussian-consensus-max-views-per-gaussian 4 \
  --pipeline.model.dc-regularization-enabled True \
  --pipeline.model.dc-regularization-weight 0.01 \
  colmap --images-path images_edited \
  --skip-missing-images True \
  --auto-downscale-missing-images False \
  --eval-mode all
```

## Common Flags Reference

| Flag | Description |
|------|-------------|
| `--data` | Path to dataset root |
| `--downscale-factor N` | Downscale images by factor N |
| `--load-dir` | Checkpoint directory to resume/fine-tune from |
| `--max-num-iterations` | Total training steps (including pre-training steps from checkpoint) |
| `--pipeline.datamanager.cache-images cpu` | Cache images in CPU RAM instead of GPU VRAM |
| `--pipeline.model.gaussian-consensus-num-views N` | Number of views per consensus step |
| `--pipeline.model.gaussian-consensus-max-views-per-gaussian N` | Cap visible views per Gaussian |
| `colmap --images-path <folder>` | Override image folder (for edited images) |
| `--skip-missing-images True` | Skip frames with no edited image |
| `--auto-downscale-missing-images False` | Don't auto-create downscaled images |
| `--eval-mode all` | Use all images for training, no eval split |

## Tips

- Use `caffeinate -dims` (macOS) or `systemd-inhibit` (Linux) to prevent the machine from sleeping during long training runs.
- Monitor training with TensorBoard: `tensorboard --logdir outputs/`
- Use `ns-train <method> --help` to see all available options.
