# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Code to train model.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import os
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, DefaultDict, Dict, List, Literal, Optional, Tuple, Type, cast

import mediapy as media
import numpy as np
import torch
import viser
from PIL import Image, ImageDraw
from rich import box, style
from rich.panel import Panel
from rich.table import Table
from torch.cuda.amp.grad_scaler import GradScaler

from nerfstudio.configs.experiment_config import ExperimentConfig
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.pipelines.base_pipeline import VanillaPipeline
from nerfstudio.utils import colormaps, profiler, writer
from nerfstudio.utils.consensus_visualization import compute_consensus_visualization_data
from nerfstudio.utils.decorators import check_eval_enabled, check_main_thread, check_viewer_enabled
from nerfstudio.utils.misc import step_check
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils.writer import EventName, TimeWriter
from nerfstudio.viewer.viewer import Viewer as ViewerState
from nerfstudio.viewer_legacy.server.viewer_state import ViewerLegacyState

TRAIN_INTERATION_OUTPUT = Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
TORCH_DEVICE = str


@dataclass
class TrainerConfig(ExperimentConfig):
    """Configuration for training regimen"""

    _target: Type = field(default_factory=lambda: Trainer)
    """target class to instantiate"""
    steps_per_save: int = 1000
    """Number of steps between saves."""
    steps_per_eval_batch: int = 500
    """Number of steps between randomly sampled batches of rays."""
    steps_per_eval_image: int = 500
    """Number of steps between single eval images."""
    steps_per_eval_all_images: int = 25000
    """Number of steps between eval all images."""
    max_num_iterations: int = 1000000
    """Maximum number of iterations to run."""
    mixed_precision: bool = False
    """Whether or not to use mixed precision for training."""
    use_grad_scaler: bool = False
    """Use gradient scaler even if the automatic mixed precision is disabled."""
    save_only_latest_checkpoint: bool = True
    """Whether to only save the latest checkpoint or all checkpoints."""
    # optional parameters if we want to resume training
    load_dir: Optional[Path] = None
    """Optionally specify a pre-trained model directory to load from."""
    load_step: Optional[int] = None
    """Optionally specify model step to load from; if none, will find most recent model in load_dir."""
    load_config: Optional[Path] = None
    """Path to config YAML file."""
    load_checkpoint: Optional[Path] = None
    """Path to checkpoint file."""
    log_gradients: bool = False
    """Optionally log gradients during training"""
    gradient_accumulation_steps: Dict[str, int] = field(default_factory=lambda: {})
    """Number of steps to accumulate gradients over. Contains a mapping of {param_group:num}"""
    start_paused: bool = False
    """Whether to start the training in a paused state."""
    downscale_factor: Optional[int] = None
    """Override the dataparser image downscale factor, e.g. 8 to use/create images_8."""


class Trainer:
    """Trainer class

    Args:
        config: The configuration object.
        local_rank: Local rank of the process.
        world_size: World size of the process.

    Attributes:
        config: The configuration object.
        local_rank: Local rank of the process.
        world_size: World size of the process.
        device: The device to run the training on.
        pipeline: The pipeline object.
        optimizers: The optimizers object.
        callbacks: The callbacks object.
        training_state: Current model training state.
    """

    pipeline: VanillaPipeline
    optimizers: Optimizers
    callbacks: List[TrainingCallback]

    def __init__(self, config: TrainerConfig, local_rank: int = 0, world_size: int = 1) -> None:
        self.train_lock = Lock()
        self.config = config
        self.local_rank = local_rank
        self.world_size = world_size
        self.device: TORCH_DEVICE = config.machine.device_type
        if self.device == "cuda":
            self.device += f":{local_rank}"
        self.mixed_precision: bool = self.config.mixed_precision
        self.use_grad_scaler: bool = self.mixed_precision or self.config.use_grad_scaler
        self.training_state: Literal["training", "paused", "completed"] = (
            "paused" if self.config.start_paused else "training"
        )
        self.gradient_accumulation_steps: DefaultDict = defaultdict(lambda: 1)
        self.gradient_accumulation_steps.update(self.config.gradient_accumulation_steps)

        if self.device == "cpu":
            self.mixed_precision = False
            CONSOLE.print("Mixed precision is disabled for CPU training.")
        self._start_step: int = 0
        # optimizers
        self.grad_scaler = GradScaler(enabled=self.use_grad_scaler)

        self.base_dir: Path = config.get_base_dir()
        # directory to save checkpoints
        self.checkpoint_dir: Path = config.get_checkpoint_dir()
        CONSOLE.log(f"Saving checkpoints to: {self.checkpoint_dir}")

        self.viewer_state = None

        # used to keep track of the current step
        self.step = 0

    def setup(self, test_mode: Literal["test", "val", "inference"] = "val") -> None:
        """Setup the Trainer by calling other setup functions.

        Args:
            test_mode:
                'val': loads train/val datasets into memory
                'test': loads train/test datasets into memory
                'inference': does not load any dataset into memory
        """
        if self.config.downscale_factor is not None:
            dataparser = self.config.pipeline.datamanager.dataparser
            if not hasattr(dataparser, "downscale_factor"):
                raise ValueError("The configured dataparser does not support downscale_factor.")
            dataparser.downscale_factor = self.config.downscale_factor

        self.pipeline = self.config.pipeline.setup(
            device=self.device,
            test_mode=test_mode,
            world_size=self.world_size,
            local_rank=self.local_rank,
            grad_scaler=self.grad_scaler,
        )
        self.optimizers = self.setup_optimizers()

        # set up viewer if enabled
        viewer_log_path = self.base_dir / self.config.viewer.relative_log_filename
        self.viewer_state, banner_messages = None, None
        if self.config.is_viewer_legacy_enabled() and self.local_rank == 0:
            datapath = self.config.data
            if datapath is None:
                datapath = self.base_dir
            self.viewer_state = ViewerLegacyState(
                self.config.viewer,
                log_filename=viewer_log_path,
                datapath=datapath,
                pipeline=self.pipeline,
                trainer=self,
                train_lock=self.train_lock,
            )
            banner_messages = [f"Legacy viewer at: {self.viewer_state.viewer_url}"]
        if self.config.is_viewer_enabled() and self.local_rank == 0:
            datapath = self.config.data
            if datapath is None:
                datapath = self.base_dir
            self.viewer_state = ViewerState(
                self.config.viewer,
                log_filename=viewer_log_path,
                datapath=datapath,
                pipeline=self.pipeline,
                trainer=self,
                train_lock=self.train_lock,
                share=self.config.viewer.make_share_url,
            )
            banner_messages = self.viewer_state.viewer_info
        self._check_viewer_warnings()

        self._load_checkpoint()

        self.callbacks = self.pipeline.get_training_callbacks(
            TrainingCallbackAttributes(
                optimizers=self.optimizers, grad_scaler=self.grad_scaler, pipeline=self.pipeline, trainer=self
            )
        )

        # set up writers/profilers if enabled
        writer_log_path = self.base_dir / self.config.logging.relative_log_dir
        writer.setup_event_writer(
            self.config.is_wandb_enabled(),
            self.config.is_tensorboard_enabled(),
            self.config.is_comet_enabled(),
            log_dir=writer_log_path,
            experiment_name=self.config.experiment_name,
            project_name=self.config.project_name,
        )
        writer.setup_local_writer(
            self.config.logging, max_iter=self.config.max_num_iterations, banner_messages=banner_messages
        )
        writer.put_config(name="config", config_dict=dataclasses.asdict(self.config), step=0)
        profiler.setup_profiler(self.config.logging, writer_log_path)

    def setup_optimizers(self) -> Optimizers:
        """Helper to set up the optimizers

        Returns:
            The optimizers object given the trainer config.
        """
        optimizer_config = self.config.optimizers.copy()
        param_groups = self.pipeline.get_param_groups()
        return Optimizers(optimizer_config, param_groups)

    def train(self) -> None:
        """Train the model."""
        assert self.pipeline.datamanager.train_dataset is not None, "Missing DatsetInputs"
        if hasattr(self.pipeline.datamanager, "train_dataparser_outputs"):
            self.pipeline.datamanager.train_dataparser_outputs.save_dataparser_transform(  # type: ignore
                self.base_dir / "dataparser_transforms.json"
            )

        self._init_viewer_state()
        with TimeWriter(writer, EventName.TOTAL_TRAIN_TIME):
            num_iterations = self.config.max_num_iterations - self._start_step
            step = 0
            self.stop_training = False
            for step in range(self._start_step, self._start_step + num_iterations):
                self.step = step
                if self.stop_training:
                    break
                while self.training_state == "paused":
                    if self.stop_training:
                        self._after_train()
                        return
                    time.sleep(0.01)
                with self.train_lock:
                    with TimeWriter(writer, EventName.ITER_TRAIN_TIME, step=step) as train_t:
                        self.pipeline.train()

                        # training callbacks before the training iteration
                        for callback in self.callbacks:
                            callback.run_callback_at_location(
                                step, location=TrainingCallbackLocation.BEFORE_TRAIN_ITERATION
                            )

                        # time the forward pass
                        loss, loss_dict, metrics_dict = self.train_iteration(step)

                        # training callbacks after the training iteration
                        for callback in self.callbacks:
                            callback.run_callback_at_location(
                                step, location=TrainingCallbackLocation.AFTER_TRAIN_ITERATION
                            )

                # Skip the first two steps to avoid skewed timings that break the viewer rendering speed estimate.
                if step > 1:
                    writer.put_time(
                        name=EventName.TRAIN_RAYS_PER_SEC,
                        duration=self.world_size
                        * self.pipeline.datamanager.get_train_rays_per_batch()
                        / max(0.001, train_t.duration),
                        step=step,
                        avg_over_steps=True,
                    )

                self._update_viewer_state(step)

                # a batch of train rays
                if step_check(step, self.config.logging.steps_per_log, run_at_zero=True):
                    writer.put_scalar(name="Train Loss", scalar=loss, step=step)
                    writer.put_dict(name="Train Loss Dict", scalar_dict=loss_dict, step=step)
                    writer.put_dict(name="Train Metrics Dict", scalar_dict=metrics_dict, step=step)
                    # The actual memory allocated by Pytorch. This is likely less than the amount
                    # shown in nvidia-smi since some unused memory can be held by the caching
                    # allocator and some context needs to be created on GPU. See Memory management
                    # (https://pytorch.org/docs/stable/notes/cuda.html#cuda-memory-management)
                    # for more details about GPU memory management.
                    writer.put_scalar(
                        name="GPU Memory (MB)", scalar=torch.cuda.max_memory_allocated() / (1024**2), step=step
                    )

                # Do not perform evaluation if there are no validation images
                if self.pipeline.datamanager.eval_dataset and len(self.pipeline.datamanager.eval_dataset) > 0:
                    with self.train_lock:
                        self.eval_iteration(step)

                if step_check(step, self.config.steps_per_save):
                    self.save_checkpoint(step)

                writer.write_out_storage()

        # save checkpoint at the end of training, and write out any remaining events
        self._after_train()

    def shutdown(self) -> None:
        """Stop the trainer and stop all associated threads/processes (such as the viewer)."""
        self.stop_training = True  # tell the training loop to stop
        if self.viewer_state is not None:
            # stop the viewer
            # this condition excludes the case where `viser_server` is either `None` or an
            # instance of `viewer_legacy`'s `ViserServer` instead of the upstream one.
            if isinstance(self.viewer_state.viser_server, viser.ViserServer):
                self.viewer_state.viser_server.stop()

    def _after_train(self) -> None:
        """Function to run after training is complete"""
        self.training_state = "completed"  # used to update the webui state
        # save checkpoint at the end of training
        self.save_checkpoint(self.step)
        # write out any remaining events (e.g., total train time)
        writer.write_out_storage()
        table = Table(
            title=None,
            show_header=False,
            box=box.MINIMAL,
            title_style=style.Style(bold=True),
        )
        table.add_row("Config File", str(self.config.get_base_dir() / "config.yml"))
        table.add_row("Checkpoint Directory", str(self.checkpoint_dir))
        CONSOLE.print(Panel(table, title="[bold][green]:tada: Training Finished :tada:[/bold]", expand=False))

        # after train end callbacks
        for callback in self.callbacks:
            callback.run_callback_at_location(step=self.step, location=TrainingCallbackLocation.AFTER_TRAIN)

        if not self.config.viewer.quit_on_train_completion:
            self._train_complete_viewer()

    @check_main_thread
    def _check_viewer_warnings(self) -> None:
        """Helper to print out any warnings regarding the way the viewer/loggers are enabled"""
        if (
            (self.config.is_viewer_legacy_enabled() or self.config.is_viewer_enabled())
            and not self.config.is_tensorboard_enabled()
            and not self.config.is_wandb_enabled()
            and not self.config.is_comet_enabled()
        ):
            string: str = (
                "[NOTE] Not running eval iterations since only viewer is enabled.\n"
                "Use [yellow]--vis {wandb, tensorboard, viewer+wandb, viewer+tensorboard}[/yellow] to run with eval."
            )
            CONSOLE.print(f"{string}")

    @check_viewer_enabled
    def _init_viewer_state(self) -> None:
        """Initializes viewer scene with given train dataset"""
        assert self.viewer_state and self.pipeline.datamanager.train_dataset
        self.viewer_state.init_scene(
            train_dataset=self.pipeline.datamanager.train_dataset,
            train_state=self.training_state,
            eval_dataset=self.pipeline.datamanager.eval_dataset,
        )

    @check_viewer_enabled
    def _update_viewer_state(self, step: int) -> None:
        """Updates the viewer state by rendering out scene with current pipeline
        Returns the time taken to render scene.

        Args:
            step: current train step
        """
        assert self.viewer_state is not None
        num_rays_per_batch: int = self.pipeline.datamanager.get_train_rays_per_batch()
        try:
            self.viewer_state.update_scene(step, num_rays_per_batch)
        except RuntimeError:
            time.sleep(0.03)  # sleep to allow buffer to reset
            CONSOLE.log("Viewer failed. Continuing training.")

    @check_viewer_enabled
    def _train_complete_viewer(self) -> None:
        """Let the viewer know that the training is complete"""
        assert self.viewer_state is not None
        self.training_state = "completed"
        try:
            self.viewer_state.training_complete()
        except RuntimeError:
            time.sleep(0.03)  # sleep to allow buffer to reset
            CONSOLE.log("Viewer failed. Continuing training.")
        CONSOLE.print("Use ctrl+c to quit", justify="center")
        while True:
            time.sleep(0.01)

    @check_viewer_enabled
    def _update_viewer_rays_per_sec(self, train_t: TimeWriter, vis_t: TimeWriter, step: int) -> None:
        """Performs update on rays/sec calculation for training

        Args:
            train_t: timer object carrying time to execute total training iteration
            vis_t: timer object carrying time to execute visualization step
            step: current step
        """
        train_num_rays_per_batch: int = self.pipeline.datamanager.get_train_rays_per_batch()
        writer.put_time(
            name=EventName.TRAIN_RAYS_PER_SEC,
            duration=self.world_size * train_num_rays_per_batch / (train_t.duration - vis_t.duration),
            step=step,
            avg_over_steps=True,
        )

    def _load_checkpoint(self) -> None:
        """Helper function to load pipeline and optimizer from prespecified checkpoint"""
        load_dir = self.config.load_dir
        load_checkpoint = self.config.load_checkpoint
        if load_dir is not None:
            load_step = self.config.load_step
            if load_step is None:
                print("Loading latest Nerfstudio checkpoint from load_dir...")
                # NOTE: this is specific to the checkpoint name format
                load_step = sorted(int(x[x.find("-") + 1 : x.find(".")]) for x in os.listdir(load_dir))[-1]
            load_path: Path = load_dir / f"step-{load_step:09d}.ckpt"
            assert load_path.exists(), f"Checkpoint {load_path} does not exist"
            loaded_state = torch.load(load_path, map_location="cpu")
            self._start_step = loaded_state["step"] + 1
            # load the checkpoints for pipeline, optimizers, and gradient scalar
            self.pipeline.load_pipeline(loaded_state["pipeline"], loaded_state["step"])
            self.optimizers.update_parameters(self.pipeline.get_param_groups())
            self.optimizers.load_optimizers(loaded_state["optimizers"])
            if "schedulers" in loaded_state and self.config.load_scheduler:
                self.optimizers.load_schedulers(loaded_state["schedulers"])
            self.grad_scaler.load_state_dict(loaded_state["scalers"])
            CONSOLE.print(f"Done loading Nerfstudio checkpoint from {load_path}")
        elif load_checkpoint is not None:
            assert load_checkpoint.exists(), f"Checkpoint {load_checkpoint} does not exist"
            loaded_state = torch.load(load_checkpoint, map_location="cpu")
            self._start_step = loaded_state["step"] + 1
            # load the checkpoints for pipeline, optimizers, and gradient scalar
            self.pipeline.load_pipeline(loaded_state["pipeline"], loaded_state["step"])
            self.optimizers.update_parameters(self.pipeline.get_param_groups())
            self.optimizers.load_optimizers(loaded_state["optimizers"])
            if "schedulers" in loaded_state and self.config.load_scheduler:
                self.optimizers.load_schedulers(loaded_state["schedulers"])
            self.grad_scaler.load_state_dict(loaded_state["scalers"])
            CONSOLE.print(f"Done loading Nerfstudio checkpoint from {load_checkpoint}")
        else:
            CONSOLE.print("No Nerfstudio checkpoint to load, so training from scratch.")

    @check_main_thread
    def save_checkpoint(self, step: int) -> None:
        """Save the model and optimizers

        Args:
            step: number of steps in training for given checkpoint
        """
        # possibly make the checkpoint directory
        if not self.checkpoint_dir.exists():
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # save the checkpoint
        ckpt_path: Path = self.checkpoint_dir / f"step-{step:09d}.ckpt"
        torch.save(
            {
                "step": step,
                "pipeline": self.pipeline.module.state_dict()  # type: ignore
                if hasattr(self.pipeline, "module")
                else self.pipeline.state_dict(),
                "optimizers": {k: v.state_dict() for (k, v) in self.optimizers.optimizers.items()},
                "schedulers": {k: v.state_dict() for (k, v) in self.optimizers.schedulers.items()},
                "scalers": self.grad_scaler.state_dict(),
            },
            ckpt_path,
        )
        # possibly delete old checkpoints
        if self.config.save_only_latest_checkpoint:
            # delete every other checkpoint in the checkpoint folder
            for f in self.checkpoint_dir.glob("*.ckpt"):
                if f != ckpt_path:
                    f.unlink()

    def _get_gaussian_consensus_config(self):
        model_config = getattr(self.pipeline.model, "config", None)
        if model_config is not None and getattr(model_config, "gaussian_consensus_enabled", False):
            return model_config
        return None

    def _get_gaussian_trainable_param_groups(self) -> Optional[List[str]]:
        model_config = getattr(self.pipeline.model, "config", None)
        groups = getattr(model_config, "gaussian_trainable_param_groups", None) if model_config is not None else None
        if groups is None:
            return None

        groups = list(groups)
        if len(groups) == 0:
            raise RuntimeError("Gaussian training needs at least one trainable parameter group.")
        gauss_params = getattr(self.pipeline.model, "gauss_params", {})
        missing = [group for group in groups if group not in self.optimizers.parameters or group not in gauss_params]
        if missing:
            raise RuntimeError(
                "Configured Gaussian trainable groups must be Gaussian optimizer groups. "
                f"Missing or invalid groups: {missing}"
            )
        return groups

    def _get_gaussian_consensus_param_groups(self) -> List[str]:
        config = self._get_gaussian_consensus_config()
        assert config is not None
        groups = list(config.gaussian_consensus_trainable_param_groups)
        if len(groups) == 0:
            raise RuntimeError("Gaussian consensus needs at least one trainable parameter group.")
        gauss_params = getattr(self.pipeline.model, "gauss_params", {})
        missing = [group for group in groups if group not in self.optimizers.parameters or group not in gauss_params]
        if missing:
            raise RuntimeError(
                "Gaussian consensus trainable groups must be Gaussian optimizer groups. "
                f"Missing or invalid groups: {missing}"
            )
        return groups

    @contextmanager
    def _only_consensus_groups_require_grad(self, trainable_groups: List[str]):
        trainable = set(trainable_groups)
        previous_requires_grad = {}
        for group, params in self.optimizers.parameters.items():
            group_requires_grad = group in trainable
            for param in params:
                if param not in previous_requires_grad:
                    previous_requires_grad[param] = param.requires_grad
                param.requires_grad_(group_requires_grad)
        try:
            yield
        finally:
            for param, requires_grad in previous_requires_grad.items():
                param.requires_grad_(requires_grad)

    def _get_gaussian_consensus_visualization_groups(self, trainable_groups: List[str], config) -> List[str]:
        groups = list(getattr(config, "gaussian_consensus_visualization_groups", ()))
        if len(groups) == 0:
            groups = list(trainable_groups)
        missing = [group for group in groups if group not in trainable_groups]
        if missing:
            raise RuntimeError(
                "Consensus visualization groups must be trainable consensus groups. "
                f"Missing from trainable groups: {missing}"
            )
        return groups

    def _get_gaussian_consensus_visualization_window_start(self, step: int, config) -> int:
        interval = int(getattr(config, "gaussian_consensus_visualization_interval", 0))
        if interval <= 0:
            return step
        return (step // interval) * interval

    def _should_capture_gaussian_consensus_visualization(self, step: int, config) -> bool:
        if not bool(getattr(config, "gaussian_consensus_visualization_enabled", False)):
            return False
        interval = int(getattr(config, "gaussian_consensus_visualization_interval", 0))
        window = max(1, int(getattr(config, "gaussian_consensus_visualization_window", 1)))
        return interval > 0 and step % interval < window

    def _clone_gaussian_consensus_visualization_grad(self, param: torch.Tensor, grad: Optional[torch.Tensor], config):
        if grad is None:
            return torch.zeros(param.shape, dtype=param.dtype, device="cpu")
        grad_for_view = grad.detach().clone()
        if getattr(config, "gaussian_consensus_store_grads_on_cpu", True):
            grad_for_view = grad_for_view.cpu()
        return grad_for_view

    def _get_consensus_view_cam_idx(self, camera: Any, batch: Dict[str, Any]) -> Optional[int]:
        metadata = getattr(camera, "metadata", None)
        value = None
        if isinstance(metadata, dict) and "cam_idx" in metadata:
            value = metadata["cam_idx"]
        elif "image_idx" in batch:
            value = batch["image_idx"]
        if isinstance(value, torch.Tensor):
            value = value.detach().flatten()[0].cpu().item()
        if value is None:
            return None
        return int(value)

    def _capture_gaussian_consensus_view_record(self, camera: Any, batch: Dict[str, Any]) -> Dict[str, Any]:
        image = batch.get("image")
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu()
        return {
            "camera": camera,
            "image": image,
            "cam_idx": self._get_consensus_view_cam_idx(camera, batch),
        }

    @staticmethod
    def _image_tensor_to_numpy(image: Any) -> np.ndarray:
        if isinstance(image, torch.Tensor):
            tensor = image.detach().cpu()
            array = tensor.float().numpy()
            integer_image = not tensor.dtype.is_floating_point
        else:
            raw_array = np.asarray(image)
            integer_image = np.issubdtype(raw_array.dtype, np.integer)
            array = raw_array.astype(np.float32)
        if array.ndim == 2:
            array = array[..., None]
        if array.shape[-1] == 1:
            array = np.repeat(array, 3, axis=-1)
        if array.shape[-1] > 3:
            array = array[..., :3]
        finite = array[np.isfinite(array)]
        if integer_image or (finite.size > 0 and float(np.max(finite)) > 1.5):
            array = array / 255.0
        return np.clip(array, 0.0, 1.0)

    @staticmethod
    def _float_image_to_uint8(image: np.ndarray) -> np.ndarray:
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        return (np.clip(image, 0.0, 1.0) * 255.0).round().astype(np.uint8)

    @staticmethod
    def _signed_array_to_rgb(array: np.ndarray, scale: Optional[float] = None) -> np.ndarray:
        array = np.nan_to_num(array.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if scale is None:
            scale = float(np.percentile(np.abs(array), 99.0)) if array.size > 0 else 0.0
            if scale <= 0.0:
                scale = float(np.max(np.abs(array))) if array.size > 0 else 1.0
        scale = max(scale, 1e-8)
        if array.ndim == 2 or array.shape[-1] == 1:
            scalar = np.squeeze(array, axis=-1) if array.ndim == 3 else array
            signed = np.clip(scalar / scale, -1.0, 1.0)
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

    @staticmethod
    def _tensor_finite_max(value: torch.Tensor) -> float:
        value = value.detach().float()
        finite = value[torch.isfinite(value)]
        if finite.numel() == 0:
            return 0.0
        return float(finite.max().cpu().item())

    @staticmethod
    def _tensor_finite_abs_max(value: torch.Tensor) -> float:
        value = value.detach().float()
        finite = value[torch.isfinite(value)]
        if finite.numel() == 0:
            return 0.0
        return float(finite.abs().max().cpu().item())

    @staticmethod
    def _is_update_norm_map(name: str) -> bool:
        return name.startswith("view_update_norm_") or name in {"mean_update_norm", "final_update_norm"}

    @staticmethod
    def _scale_record(kind: str, min_value: float, max_value: float) -> Dict[str, Any]:
        return {"kind": kind, "min": float(min_value), "max": float(max_value)}

    @staticmethod
    def _scalar_tensor_to_panel_image(
        name: str,
        value: torch.Tensor,
        value_range: Optional[Tuple[float, float]] = None,
        signed_scale: Optional[float] = None,
    ) -> np.ndarray:
        value = value.detach().cpu().float()
        if value.ndim == 2:
            value = value[..., None]
        bg_mask = torch.isnan(value[..., 0])
        if "opacity_update" in name:
            result = Trainer._signed_array_to_rgb(value.numpy(), scale=signed_scale)
            result[bg_mask.numpy()] = 0.5
            return result
        normalize = name not in {"agreement", "disagreement", "dominance_strength", "suppression_ratio"}
        if value_range is not None:
            min_value, max_value = value_range
            denominator = max(float(max_value) - float(min_value), 1e-8)
            value = torch.nan_to_num(value, nan=0.0, posinf=float(max_value), neginf=float(min_value))
            value = torch.clamp((value - float(min_value)) / denominator, 0.0, 1.0)
            normalize = False
        else:
            value = torch.nan_to_num(value, nan=0.0)
        options = colormaps.ColormapOptions(colormap="turbo", normalize=normalize)
        result = colormaps.apply_colormap(value, colormap_options=options).cpu().numpy()
        result[bg_mask.numpy()] = 0.5
        return result

    @staticmethod
    def _rgb_tensor_to_panel_image(name: str, value: torch.Tensor, signed_scale: Optional[float] = None) -> np.ndarray:
        array = value.detach().cpu().float().numpy()
        bg_mask = np.isnan(array[..., 0])
        if "update" in name:
            result = Trainer._signed_array_to_rgb(np.nan_to_num(array, nan=0.0), scale=signed_scale)
            result[bg_mask] = 0.5
            return result
        result = np.clip(np.nan_to_num(array, nan=0.0), 0.0, 1.0)
        result[bg_mask] = 0.5
        return result

    @staticmethod
    def _write_png(path: Path, image: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        media.write_image(path, Trainer._float_image_to_uint8(image), fmt="png")

    @staticmethod
    def _labeled_image(label: str, image: np.ndarray) -> Image.Image:
        array = Trainer._float_image_to_uint8(image)
        pil_image = Image.fromarray(array)
        label_height = 24
        canvas = Image.new("RGB", (pil_image.width, pil_image.height + label_height), color=(18, 18, 18))
        canvas.paste(pil_image, (0, label_height))
        draw = ImageDraw.Draw(canvas)
        draw.text((6, 5), label, fill=(240, 240, 240))
        return canvas

    @staticmethod
    def _write_dashboard(path: Path, rows: List[List[Tuple[str, np.ndarray]]]) -> None:
        labeled_rows: List[Image.Image] = []
        for row in rows:
            cells = [Trainer._labeled_image(label, image) for label, image in row]
            if len(cells) == 0:
                continue
            width = sum(cell.width for cell in cells)
            height = max(cell.height for cell in cells)
            row_image = Image.new("RGB", (width, height), color=(18, 18, 18))
            x_offset = 0
            for cell in cells:
                row_image.paste(cell, (x_offset, 0))
                x_offset += cell.width
            labeled_rows.append(row_image)
        if len(labeled_rows) == 0:
            return
        width = max(row.width for row in labeled_rows)
        height = sum(row.height for row in labeled_rows)
        dashboard = Image.new("RGB", (width, height), color=(18, 18, 18))
        y_offset = 0
        for row in labeled_rows:
            dashboard.paste(row, (0, y_offset))
            y_offset += row.height
        path.parent.mkdir(parents=True, exist_ok=True)
        dashboard.save(path)

    @staticmethod
    def _read_png_float(path: Path) -> Optional[np.ndarray]:
        if not path.exists():
            return None
        return np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0

    @staticmethod
    def _write_gaussian_consensus_preview_dashboard(step_dir: Path, manifest: Dict[str, Any]) -> None:
        panels = manifest.get("panels", {})
        if not panels:
            return

        def get_panel(name: str) -> Optional[np.ndarray]:
            relative_path = panels.get(name)
            if relative_path is None:
                return None
            return Trainer._read_png_float(step_dir / relative_path)

        input_row: List[Tuple[str, np.ndarray]] = []
        cam_indices = manifest.get("cam_indices", [])
        num_views = int(manifest.get("num_views", 0))
        for view_idx in range(num_views):
            name = "input_anchor" if view_idx == 0 else f"input_ref_{view_idx:02d}"
            image = get_panel(name)
            if image is None:
                continue
            label = "anchor" if view_idx == 0 else f"ref {view_idx}"
            cam_idx = cam_indices[view_idx] if view_idx < len(cam_indices) else None
            if cam_idx is not None:
                label = f"{label} ({cam_idx})"
            input_row.append((label, image))

        norm_row: List[Tuple[str, np.ndarray]] = []
        for view_idx in range(num_views):
            name = f"view_update_norm_{view_idx:02d}"
            image = get_panel(name)
            if image is not None:
                norm_row.append((f"view {view_idx}", image))
        for label, name in (("mean", "mean_update_norm"), ("final", "final_update_norm")):
            image = get_panel(name)
            if image is not None:
                norm_row.append((label, image))

        agreement_row: List[Tuple[str, np.ndarray]] = []
        for label, name in (
            ("visible views", "visible_view_count"),
            ("agreement", "agreement"),
            ("disagreement", "disagreement"),
            ("dominant", "dominant_view"),
            ("dominance", "dominance_strength"),
            ("suppression", "suppression_ratio"),
        ):
            image = get_panel(name)
            if image is not None:
                agreement_row.append((label, image))

        direction_row: List[Tuple[str, np.ndarray]] = []
        for label, name in (
            ("rgb before", "current_rgb"),
            ("rgb after", "target_rgb_after_step"),
            ("rgb mean", "rgb_update_mean"),
            ("rgb final", "rgb_update_final"),
            ("opacity mean", "opacity_update_mean"),
            ("opacity final", "opacity_update_final"),
        ):
            image = get_panel(name)
            if image is not None:
                direction_row.append((label, image))

        dashboard_rows = [row for row in (input_row, norm_row, agreement_row, direction_row) if len(row) > 0]
        Trainer._write_dashboard(step_dir / "preview_dashboard.png", dashboard_rows)

    @staticmethod
    def _tensor_to_npz_array(value: torch.Tensor) -> np.ndarray:
        array = value.detach().cpu().numpy()
        if array.dtype.kind == "f":
            return array.astype(np.float32)
        return array

    def _render_current_rgb_for_consensus_visualization(self, camera: Any) -> Optional[np.ndarray]:
        model = self.pipeline.model
        was_training = model.training
        try:
            model.eval()
            with torch.no_grad():
                outputs = model.get_outputs_for_camera(camera)
            rgb = outputs.get("rgb")
            if isinstance(rgb, torch.Tensor):
                return self._image_tensor_to_numpy(rgb)
        finally:
            if was_training:
                model.train()
        return None

    def _write_gaussian_consensus_visualization_snapshot(
        self,
        step: int,
        config,
        groups: List[str],
        view_records: List[Dict[str, Any]],
        view_grads: Dict[str, List[torch.Tensor]],
        final_grads: Dict[str, torch.Tensor],
    ) -> None:
        if len(view_records) == 0 or len(groups) == 0:
            return
        learning_rates = {group: self.pipeline.model._get_optimizer_lr(group) for group in groups}
        per_gaussian, scalar_attributes, rgb_attributes = compute_consensus_visualization_data(
            {group: view_grads[group] for group in groups if group in view_grads},
            {group: final_grads[group] for group in groups if group in final_grads},
            learning_rates=learning_rates,
            eps=float(config.gaussian_consensus_eps),
            max_views=int(config.gaussian_consensus_max_views_per_gaussian),
        )
        if len(per_gaussian) == 0:
            return

        anchor_camera = view_records[0]["camera"]
        rendered_maps = self.pipeline.model.render_gaussian_attribute_maps_for_camera(
            anchor_camera, scalar_attributes=scalar_attributes, rgb_attributes=rgb_attributes
        )

        output_root = self.base_dir / str(getattr(config, "gaussian_consensus_visualization_output_dir", "consensus_visualizations"))
        step_dir = output_root / f"step_{step:09d}"
        panels: Dict[str, str] = {}
        npz_arrays: Dict[str, np.ndarray] = {}

        cam_indices = [record.get("cam_idx") for record in view_records]
        interval = int(getattr(config, "gaussian_consensus_visualization_interval", 0))
        window_size = max(1, int(getattr(config, "gaussian_consensus_visualization_window", 1)))
        window_start = self._get_gaussian_consensus_visualization_window_start(step, config)
        for view_idx, record in enumerate(view_records):
            image = record.get("image")
            if image is None:
                continue
            name = "input_anchor" if view_idx == 0 else f"input_ref_{view_idx:02d}"
            image_np = self._image_tensor_to_numpy(image)
            npz_arrays[f"input__{name}"] = image_np.astype(np.float32)
            if bool(getattr(config, "gaussian_consensus_visualization_save_png", True)):
                panels[name] = f"panels/{name}.png"
                self._write_png(step_dir / panels[name], image_np)

        panel_scales: Dict[str, Dict[str, Any]] = {}
        norm_scale_values = [
            self._tensor_finite_max(value)
            for name, value in rendered_maps.items()
            if self._is_update_norm_map(name)
        ]
        norm_scale_max = max(norm_scale_values, default=0.0)
        norm_scale_max = max(norm_scale_max, 1e-8)
        rgb_signed_scale = max(
            (self._tensor_finite_abs_max(value) for name, value in rendered_maps.items() if "rgb_update" in name),
            default=0.0,
        )
        rgb_signed_scale = max(rgb_signed_scale, 1e-8)
        opacity_signed_scale = max(
            (self._tensor_finite_abs_max(value) for name, value in rendered_maps.items() if "opacity_update" in name),
            default=0.0,
        )
        opacity_signed_scale = max(opacity_signed_scale, 1e-8)
        max_view_index = max(1, len(view_records) - 1)
        max_visible_count = max(1, len(view_records))

        alpha_map = rendered_maps.pop("_alpha", None)
        if alpha_map is not None:
            npz_arrays["map___alpha"] = self._tensor_to_npz_array(alpha_map)

        for name, value in rendered_maps.items():
            npz_arrays[f"map__{name}"] = self._tensor_to_npz_array(value)
            scalar_range: Optional[Tuple[float, float]] = None
            signed_scale: Optional[float] = None
            if self._is_update_norm_map(name):
                scalar_range = (0.0, norm_scale_max)
                panel_scales[name] = self._scale_record("linear", 0.0, norm_scale_max)
            elif name in {"agreement", "disagreement", "dominance_strength", "suppression_ratio"}:
                scalar_range = (0.0, 1.0)
                panel_scales[name] = self._scale_record("linear", 0.0, 1.0)
            elif name == "visible_view_count":
                scalar_range = (0.0, float(max_visible_count))
                panel_scales[name] = self._scale_record("count", 0.0, float(max_visible_count))
            elif name == "dominant_view":
                scalar_range = (0.0, float(max_view_index))
                panel_scales[name] = self._scale_record("categorical", 0.0, float(max_view_index))
            elif "opacity_update" in name:
                signed_scale = opacity_signed_scale
                panel_scales[name] = self._scale_record("signed", -opacity_signed_scale, opacity_signed_scale)

            if value.shape[-1] == 1:
                panel_image = self._scalar_tensor_to_panel_image(
                    name, value, value_range=scalar_range, signed_scale=signed_scale
                )
            else:
                if "rgb_update" in name:
                    signed_scale = rgb_signed_scale
                    panel_scales[name] = self._scale_record("signed", -rgb_signed_scale, rgb_signed_scale)
                panel_image = self._rgb_tensor_to_panel_image(name, value, signed_scale=signed_scale)
            if bool(getattr(config, "gaussian_consensus_visualization_save_png", True)):
                panels[name] = f"panels/{name}.png"
                self._write_png(step_dir / panels[name], panel_image)

        current_rgb = self._render_current_rgb_for_consensus_visualization(anchor_camera)
        if current_rgb is not None:
            npz_arrays["map__current_rgb"] = current_rgb.astype(np.float32)
            panel_scales["current_rgb"] = self._scale_record("rgb", 0.0, 1.0)
            if bool(getattr(config, "gaussian_consensus_visualization_save_png", True)):
                panels["current_rgb"] = "panels/current_rgb.png"
                self._write_png(step_dir / panels["current_rgb"], current_rgb)

        for name, value in per_gaussian.items():
            npz_arrays[f"gaussian__{name}"] = self._tensor_to_npz_array(value)
        npz_arrays["view_cam_indices"] = np.asarray([-1 if idx is None else int(idx) for idx in cam_indices], dtype=np.int64)

        summary = {
            name: float(torch.nan_to_num(value.float(), nan=0.0, posinf=0.0, neginf=0.0).mean().cpu().item())
            for name, value in per_gaussian.items()
            if value.is_floating_point() and value.ndim <= 2
        }
        manifest = {
            "version": 1,
            "step": step,
            "groups": groups,
            "num_views": len(view_records),
            "cam_indices": cam_indices,
            "capture_window": {
                "start": window_start,
                "end": window_start + window_size - 1,
                "size": window_size,
                "interval": interval,
            },
            "window_start": window_start,
            "learning_rates": learning_rates,
            "summary": summary,
            "snapshot_file": "snapshot.npz" if bool(getattr(config, "gaussian_consensus_visualization_save_npz", True)) else None,
            "panel_scales": panel_scales,
            "panels": panels,
        }

        step_dir.mkdir(parents=True, exist_ok=True)
        if bool(getattr(config, "gaussian_consensus_visualization_save_npz", True)):
            np.savez_compressed(step_dir / "snapshot.npz", **npz_arrays)

        if bool(getattr(config, "gaussian_consensus_visualization_save_png", True)):
            self._write_gaussian_consensus_preview_dashboard(step_dir, manifest)
            manifest["preview_dashboard"] = "preview_dashboard.png"

        (step_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        index_path = output_root / "index.json"
        if index_path.exists():
            index = json.loads(index_path.read_text(encoding="utf-8"))
        else:
            index = {"version": 1, "captures": []}
        captures = [capture for capture in index.get("captures", []) if int(capture.get("step", -1)) != step]
        captures.append(
            {
                "step": step,
                "path": step_dir.name,
                "cam_indices": cam_indices,
                "window_start": window_start,
                "window_end": window_start + window_size - 1,
                "summary": summary,
            }
        )
        captures.sort(key=lambda capture: int(capture.get("step", 0)))
        index["captures"] = captures
        index["interval"] = interval
        index["window_size"] = window_size
        output_root.mkdir(parents=True, exist_ok=True)
        index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")

    def _append_gaussian_consensus_visualization_post_step_rgb(
        self,
        step: int,
        config,
        view_records: List[Dict[str, Any]],
    ) -> None:
        if len(view_records) == 0:
            return
        save_png = bool(getattr(config, "gaussian_consensus_visualization_save_png", True))
        save_npz = bool(getattr(config, "gaussian_consensus_visualization_save_npz", True))
        if not save_png and not save_npz:
            return

        output_root = self.base_dir / str(getattr(config, "gaussian_consensus_visualization_output_dir", "consensus_visualizations"))
        step_dir = output_root / f"step_{step:09d}"
        manifest_path = step_dir / "manifest.json"
        if not manifest_path.exists():
            return

        post_step_rgb = self._render_current_rgb_for_consensus_visualization(view_records[0]["camera"])
        if post_step_rgb is None:
            return

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        panels = manifest.setdefault("panels", {})
        panel_scales = manifest.setdefault("panel_scales", {})
        panel_name = "target_rgb_after_step"

        if save_png:
            panels[panel_name] = f"panels/{panel_name}.png"
            self._write_png(step_dir / panels[panel_name], post_step_rgb)

        snapshot_file = manifest.get("snapshot_file")
        if save_npz and snapshot_file:
            snapshot_path = step_dir / snapshot_file
            npz_arrays: Dict[str, np.ndarray] = {}
            if snapshot_path.exists():
                with np.load(snapshot_path, allow_pickle=False) as snapshot:
                    npz_arrays = {key: snapshot[key] for key in snapshot.files}
            npz_arrays[f"map__{panel_name}"] = post_step_rgb.astype(np.float32)
            np.savez_compressed(snapshot_path, **npz_arrays)

        panel_scales[panel_name] = self._scale_record("rgb", 0.0, 1.0)
        if save_png:
            self._write_gaussian_consensus_preview_dashboard(step_dir, manifest)
            manifest["preview_dashboard"] = "preview_dashboard.png"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def _new_gaussian_consensus_accumulator(self, param: torch.Tensor) -> Dict[str, torch.Tensor]:
        num_gaussians = param.shape[0]
        return {
            "sum_grad": torch.zeros_like(param),
            "visible_counts": torch.zeros(num_gaussians, device=param.device),
            "sum_sq_norms": torch.zeros(num_gaussians, device=param.device),
        }

    @torch.no_grad()
    def _update_gaussian_consensus_accumulator(
        self,
        param: torch.Tensor,
        accumulator: Dict[str, torch.Tensor],
        grad: Optional[torch.Tensor],
        config,
    ) -> None:
        num_gaussians = param.shape[0]
        if grad is None or num_gaussians == 0:
            return

        device = param.device
        dtype = param.dtype
        eps = config.gaussian_consensus_eps
        max_views = config.gaussian_consensus_max_views_per_gaussian
        chunk_size = max(1, config.gaussian_consensus_gaussian_chunk_size)

        grad = grad.detach()
        if grad.device != device or grad.dtype != dtype:
            grad = grad.to(device=device, dtype=dtype, non_blocking=True)

        sum_grad = accumulator["sum_grad"]
        visible_counts = accumulator["visible_counts"]
        sum_sq_norms = accumulator["sum_sq_norms"]

        for start in range(0, num_gaussians, chunk_size):
            end = min(start + chunk_size, num_gaussians)
            grad_chunk = grad[start:end]
            flat_grad = grad_chunk.reshape(end - start, -1)
            grad_norms = flat_grad.norm(dim=-1).to(dtype=visible_counts.dtype)
            visible = grad_norms > eps

            if max_views > 0:
                visible = visible & (visible_counts[start:end] < float(max_views))

            visible_f = visible.to(dtype=visible_counts.dtype)
            mask_shape = (end - start,) + (1,) * (grad_chunk.dim() - 1)
            sum_grad[start:end].add_(grad_chunk * visible.to(dtype=dtype).reshape(mask_shape))
            visible_counts[start:end].add_(visible_f)
            sum_sq_norms[start:end].add_(grad_norms.square() * visible_f)

    @torch.no_grad()
    def _finalize_gaussian_consensus_group(
        self,
        group_name: str,
        param: torch.Tensor,
        accumulator: Dict[str, torch.Tensor],
        config,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        num_gaussians = param.shape[0]
        if num_gaussians == 0:
            return torch.zeros_like(param), {}

        device = param.device
        dtype = param.dtype
        eps = config.gaussian_consensus_eps
        chunk_size = max(1, config.gaussian_consensus_gaussian_chunk_size)
        consensus_grad = torch.empty_like(param)
        visible_total = torch.zeros((), device=device)
        active_total = torch.zeros((), device=device)
        updated_total = torch.zeros((), device=device)
        agreement_total = torch.zeros((), device=device)
        agreement_count = torch.zeros((), device=device)

        sum_grad = accumulator["sum_grad"]
        visible_counts_all = accumulator["visible_counts"]
        sum_sq_norms = accumulator["sum_sq_norms"]

        for start in range(0, num_gaussians, chunk_size):
            end = min(start + chunk_size, num_gaussians)
            visible_counts = visible_counts_all[start:end]
            mask_shape = (end - start,) + (1,) * (param.dim() - 1)
            mean_grad = sum_grad[start:end] / visible_counts.clamp_min(1.0).to(dtype=dtype).reshape(mask_shape)
            active = visible_counts > 0

            if config.gaussian_consensus_aggregator == "mean":
                weights = active.to(dtype=visible_counts.dtype)
            elif config.gaussian_consensus_aggregator == "cosine":
                flat_mean_grad = mean_grad.reshape(end - start, -1)
                mean_norm_sq = flat_mean_grad.norm(dim=-1).to(dtype=visible_counts.dtype).square()
                expected_sq_norm = sum_sq_norms[start:end] / visible_counts.clamp_min(1.0)
                agreement = mean_norm_sq.clamp_min(0.0).sqrt() / expected_sq_norm.clamp_min(eps).sqrt()
                agreement = torch.where(active, agreement.clamp(max=1.0), torch.zeros_like(agreement))
                weights = torch.where(
                    agreement > config.gaussian_consensus_min_alignment,
                    agreement,
                    torch.zeros_like(agreement),
                )
                agreement_total += agreement[active].sum()
                agreement_count += active.sum()
            else:
                raise RuntimeError(f"Unknown Gaussian consensus aggregator: {config.gaussian_consensus_aggregator}")

            aggregate = mean_grad * weights.to(dtype=dtype).reshape(mask_shape)
            consensus_grad[start:end].copy_(aggregate)

            visible_total += visible_counts.sum()
            active_total += active.sum()
            updated_total += (weights > eps).sum()

        denom = torch.tensor(float(num_gaussians), device=device)
        stats = {
            f"Consensus/{group_name}_avg_visible_views": visible_total / denom,
            f"Consensus/{group_name}_active_fraction": active_total / denom,
            f"Consensus/{group_name}_updated_fraction": updated_total / denom,
        }
        if config.gaussian_consensus_aggregator == "cosine":
            stats[f"Consensus/{group_name}_avg_agreement"] = agreement_total / agreement_count.clamp_min(1.0)
        return consensus_grad, stats

    @torch.no_grad()
    def _aggregate_stored_gaussian_consensus_group(
        self,
        group_name: str,
        param: torch.Tensor,
        view_grads: List[torch.Tensor],
        config,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if len(view_grads) == 1:
            grad = view_grads[0].to(device=param.device, dtype=param.dtype, non_blocking=True)
            return grad, {}

        num_gaussians = param.shape[0]
        if num_gaussians == 0:
            return torch.zeros_like(param), {}

        device = param.device
        dtype = param.dtype
        eps = config.gaussian_consensus_eps
        max_views = config.gaussian_consensus_max_views_per_gaussian
        chunk_size = max(1, config.gaussian_consensus_gaussian_chunk_size)
        consensus_grad = torch.empty_like(param)
        visible_total = torch.zeros((), device=device)
        active_total = torch.zeros((), device=device)
        updated_total = torch.zeros((), device=device)

        for start in range(0, num_gaussians, chunk_size):
            end = min(start + chunk_size, num_gaussians)
            grads = torch.stack(
                [grad[start:end].to(device=device, dtype=dtype, non_blocking=True) for grad in view_grads], dim=0
            )
            flat_grads = grads.reshape(grads.shape[0], grads.shape[1], -1)
            grad_norms = flat_grads.norm(dim=-1)
            visible = grad_norms > eps

            if max_views > 0 and max_views < grads.shape[0]:
                top_values, top_indices = grad_norms.topk(k=max_views, dim=0)
                top_visible = torch.zeros_like(visible)
                top_visible.scatter_(0, top_indices, top_values > eps)
                visible = visible & top_visible

            mean_weights = visible.to(dtype=flat_grads.dtype)
            visible_counts = mean_weights.sum(dim=0)
            mean_grad = (flat_grads * mean_weights[..., None]).sum(dim=0)
            mean_grad = mean_grad / visible_counts.clamp_min(1.0)[..., None]

            if config.gaussian_consensus_aggregator == "mean":
                weights = mean_weights
            elif config.gaussian_consensus_aggregator == "cosine":
                alignment = torch.nn.functional.cosine_similarity(
                    flat_grads, mean_grad.unsqueeze(0), dim=-1, eps=eps
                )
                weights = torch.where(
                    alignment > config.gaussian_consensus_min_alignment,
                    alignment,
                    torch.zeros_like(alignment),
                )
                weights = weights * mean_weights
            else:
                raise RuntimeError(f"Unknown Gaussian consensus aggregator: {config.gaussian_consensus_aggregator}")

            weight_sum = weights.sum(dim=0)
            aggregate = (flat_grads * weights[..., None]).sum(dim=0)
            aggregate = aggregate / weight_sum.clamp_min(eps)[..., None]
            aggregate = torch.where(weight_sum[..., None] > eps, aggregate, torch.zeros_like(aggregate))
            consensus_grad[start:end].copy_(aggregate.reshape_as(grads[0]))

            visible_total += visible_counts.sum()
            active_total += (visible_counts > 0).sum()
            updated_total += (weight_sum > eps).sum()

        denom = torch.tensor(float(num_gaussians), device=device)
        stats = {
            f"Consensus/{group_name}_avg_visible_views": visible_total / denom,
            f"Consensus/{group_name}_active_fraction": active_total / denom,
            f"Consensus/{group_name}_updated_fraction": updated_total / denom,
        }
        return consensus_grad, stats

    def _mean_logged_dicts(self, dicts: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        if len(dicts) == 0:
            return {}
        averaged = {}
        for key in dicts[0].keys():
            values = []
            for item in dicts:
                value = item[key]
                if not isinstance(value, torch.Tensor):
                    value = torch.tensor(value, device=self.device)
                values.append(value.detach())
            stacked = torch.stack(values)
            if not stacked.is_floating_point() and not stacked.is_complex():
                stacked = stacked.float()
            averaged[key] = stacked.mean()
        return averaged

    def _iter_gaussian_consensus_train_views(self, step: int, config):
        datamanager = self.pipeline.datamanager
        if hasattr(datamanager, "sample_train_view_indices") and hasattr(datamanager, "get_train_camera_and_data"):
            datamanager.train_count += 1
            image_indices = datamanager.sample_train_view_indices(
                num_views=config.gaussian_consensus_num_views,
                sampling_strategy=config.gaussian_consensus_view_sampling,
                neighbor_pool_size=config.gaussian_consensus_neighbor_pool_size,
                position_weight=config.gaussian_consensus_position_weight,
                direction_weight=config.gaussian_consensus_direction_weight,
            )
            for image_idx in image_indices:
                yield datamanager.get_train_camera_and_data(image_idx)
            return

        if hasattr(datamanager, "next_train_views"):
            yield from datamanager.next_train_views(
                step=step,
                num_views=config.gaussian_consensus_num_views,
                sampling_strategy=config.gaussian_consensus_view_sampling,
                neighbor_pool_size=config.gaussian_consensus_neighbor_pool_size,
                position_weight=config.gaussian_consensus_position_weight,
                direction_weight=config.gaussian_consensus_direction_weight,
            )
            return

        for _ in range(config.gaussian_consensus_num_views):
            yield datamanager.next_train(step)

    @profiler.time_function
    def gaussian_consensus_train_iteration(self, step: int) -> TRAIN_INTERATION_OUTPUT:
        config = self._get_gaussian_consensus_config()
        assert config is not None
        if self.world_size > 1:
            raise RuntimeError("Gaussian consensus training is currently implemented for single-process training only.")
        if config.gaussian_consensus_num_views < 1:
            raise RuntimeError("gaussian_consensus_num_views must be at least 1.")

        trainable_groups = self._get_gaussian_consensus_param_groups()
        capture_visualization = self._should_capture_gaussian_consensus_visualization(step, config)
        visualization_groups = (
            self._get_gaussian_consensus_visualization_groups(trainable_groups, config) if capture_visualization else []
        )
        refine_consensus = not config.gaussian_consensus_disable_refinement
        if refine_consensus:
            required_groups = {"means", "scales", "quats", "features_dc", "features_rest", "opacities"}
            missing_groups = sorted(required_groups.difference(trainable_groups))
            if missing_groups:
                raise RuntimeError(
                    "Gaussian consensus densification requires all Gaussian parameter groups to be trainable. "
                    f"Missing groups: {missing_groups}"
                )

        accumulation = getattr(config, "gaussian_consensus_accumulation", "online")
        if accumulation not in ("online", "stored"):
            raise RuntimeError(f"Unknown Gaussian consensus accumulation mode: {accumulation}")
        if accumulation == "online":
            consensus_accumulators = {
                group: self._new_gaussian_consensus_accumulator(self.optimizers.parameters[group][0])
                for group in trainable_groups
            }
            view_grads: Dict[str, List[torch.Tensor]] = {}
        else:
            consensus_accumulators = {}
            view_grads = {group: [] for group in trainable_groups}
        visualization_view_grads = {group: [] for group in visualization_groups}
        visualization_view_records: List[Dict[str, Any]] = []
        visualization_final_grads: Dict[str, torch.Tensor] = {}
        loss_dicts: List[Dict[str, torch.Tensor]] = []
        metrics_dicts: List[Dict[str, torch.Tensor]] = []
        losses: List[torch.Tensor] = []
        cpu_or_cuda_str: str = self.device.split(":")[0]
        cpu_or_cuda_str = "cpu" if cpu_or_cuda_str == "mps" else cpu_or_cuda_str

        self.optimizers.zero_grad_all()
        with self._only_consensus_groups_require_grad(trainable_groups):
            for ray_bundle, batch in self._iter_gaussian_consensus_train_views(step, config):
                if capture_visualization:
                    visualization_view_records.append(self._capture_gaussian_consensus_view_record(ray_bundle, batch))
                with torch.autocast(device_type=cpu_or_cuda_str, enabled=self.mixed_precision):
                    model_outputs = self.pipeline._model(ray_bundle)
                    metrics_dict = self.pipeline.model.get_metrics_dict(model_outputs, batch)
                    loss_dict = self.pipeline.model.get_loss_dict(model_outputs, batch, metrics_dict)
                    loss = functools.reduce(torch.add, loss_dict.values())

                self.grad_scaler.scale(loss).backward()  # type: ignore
                for group in trainable_groups:
                    param = self.optimizers.parameters[group][0]
                    if accumulation == "online":
                        self._update_gaussian_consensus_accumulator(
                            param, consensus_accumulators[group], param.grad, config
                        )
                        if group in visualization_view_grads:
                            visualization_view_grads[group].append(
                                self._clone_gaussian_consensus_visualization_grad(param, param.grad, config)
                            )
                    else:
                        grad = param.grad
                        if grad is None:
                            grad_for_view = torch.zeros(param.shape, dtype=param.dtype, device="cpu")
                        else:
                            grad_for_view = grad.detach().clone()
                            if config.gaussian_consensus_store_grads_on_cpu:
                                grad_for_view = grad_for_view.cpu()
                        view_grads[group].append(grad_for_view)
                        if group in visualization_view_grads:
                            visualization_view_grads[group].append(grad_for_view)
                if refine_consensus:
                    self._update_gaussian_consensus_refinement_state(self.pipeline.model.info)

                loss_dicts.append({key: value.detach() for key, value in loss_dict.items()})
                metrics_dicts.append(metrics_dict)
                losses.append(loss.detach())
                self.optimizers.zero_grad_all()
                del model_outputs, loss_dict, metrics_dict, loss, ray_bundle, batch

        consensus_stats: Dict[str, torch.Tensor] = {}
        for group in trainable_groups:
            param = self.optimizers.parameters[group][0]
            if accumulation == "online":
                consensus_grad, stats = self._finalize_gaussian_consensus_group(
                    group, param, consensus_accumulators[group], config
                )
            else:
                consensus_grad, stats = self._aggregate_stored_gaussian_consensus_group(
                    group, param, view_grads[group], config
                )
            param.grad = consensus_grad
            if group in visualization_groups:
                visualization_final_grads[group] = consensus_grad.detach()
            consensus_stats.update(stats)

        if capture_visualization:
            self._write_gaussian_consensus_visualization_snapshot(
                step=step,
                config=config,
                groups=visualization_groups,
                view_records=visualization_view_records,
                view_grads=visualization_view_grads,
                final_grads=visualization_final_grads,
            )

        needs_step = [
            group
            for group in trainable_groups
            if step % self.gradient_accumulation_steps[group] == self.gradient_accumulation_steps[group] - 1
        ]
        self.optimizers.optimizer_scaler_step_some(self.grad_scaler, needs_step)

        if self.config.log_gradients:
            total_grad = 0
            for tag, value in self.pipeline.model.named_parameters():
                assert tag != "Total"
                if value.grad is not None:
                    grad = value.grad.norm()
                    consensus_stats[f"Gradients/{tag}"] = grad
                    total_grad += grad
            consensus_stats["Gradients/Total"] = cast(torch.Tensor, total_grad)

        scale = self.grad_scaler.get_scale()
        self.grad_scaler.update()
        if scale <= self.grad_scaler.get_scale():
            for group in needs_step:
                if group in self.optimizers.schedulers:
                    self.optimizers.scheduler_step(group)

        if capture_visualization:
            self._append_gaussian_consensus_visualization_post_step_rgb(
                step=step,
                config=config,
                view_records=visualization_view_records,
            )

        if refine_consensus:
            consensus_stats.update(self._step_gaussian_consensus_refinement(step, config))

        loss_dict = self._mean_logged_dicts(loss_dicts)
        metrics_dict = self._mean_logged_dicts(metrics_dicts)
        metrics_dict.update(consensus_stats)
        metrics_dict["Consensus/num_views"] = torch.tensor(
            float(config.gaussian_consensus_num_views), device=self.device
        )
        loss = torch.stack(losses).mean()
        return loss, loss_dict, metrics_dict  # type: ignore

    def _update_gaussian_batch_refinement_state(self, info: Dict[str, torch.Tensor]) -> None:
        model = self.pipeline.model
        strategy = getattr(model, "strategy", None)
        if strategy is None or not hasattr(strategy, "_update_state"):
            raise RuntimeError("Sequential Gaussian batch refinement currently requires the default gsplat strategy.")
        strategy._update_state(model.gauss_params, model.strategy_state, info, packed=False)

    def _update_gaussian_consensus_refinement_state(self, info: Dict[str, torch.Tensor]) -> None:
        model = self.pipeline.model
        strategy = getattr(model, "strategy", None)
        if strategy is None or not hasattr(strategy, "_update_state"):
            raise RuntimeError("Gaussian consensus densification currently requires the default gsplat strategy.")
        strategy._update_state(model.gauss_params, model.strategy_state, info, packed=False)

    @torch.no_grad()
    def _step_gaussian_consensus_refinement(self, step: int, config) -> Dict[str, torch.Tensor]:
        model = self.pipeline.model
        strategy = getattr(model, "strategy", None)
        if strategy is None or not all(hasattr(strategy, name) for name in ("_grow_gs", "_prune_gs")):
            raise RuntimeError("Gaussian consensus densification currently requires the default gsplat strategy.")

        state = model.strategy_state
        stats: Dict[str, torch.Tensor] = {}
        if step >= strategy.refine_stop_iter:
            return stats

        should_refine = (
            step > strategy.refine_start_iter
            and step % strategy.refine_every == 0
            and step % strategy.reset_every >= strategy.pause_refine_after_reset
        )
        if should_refine:
            min_support = max(1, int(getattr(config, "gaussian_consensus_densify_min_view_support", 1)))
            count = state.get("count")
            grad2d = state.get("grad2d")
            if min_support > 1 and count is not None and grad2d is not None:
                supported = count >= float(min_support)
                avg_grad = grad2d / count.clamp_min(1.0)
                wants_growth = avg_grad > strategy.grow_grad2d
                radii = state.get("radii")
                if step < strategy.refine_scale2d_stop_iter and radii is not None:
                    wants_growth = wants_growth | (radii > strategy.grow_scale2d)
                    radii.masked_fill_(~supported, 0.0)
                blocked = wants_growth & ~supported
                grad2d.masked_fill_(~supported, 0.0)
                denom = torch.tensor(float(max(1, supported.numel())), device=supported.device)
                stats["ConsensusDensify/support_fraction"] = supported.sum().to(dtype=torch.float32) / denom
                stats["ConsensusDensify/blocked_growth_fraction"] = blocked.sum().to(dtype=torch.float32) / denom

            n_dupli, n_split = strategy._grow_gs(model.gauss_params, model.optimizers, state, step)
            if strategy.verbose:
                print(
                    f"Step {step}: {n_dupli} consensus-supported GSs duplicated, {n_split} split. "
                    f"Now having {len(model.gauss_params['means'])} GSs."
                )

            n_prune = strategy._prune_gs(model.gauss_params, model.optimizers, state, step)
            if strategy.verbose:
                print(f"Step {step}: {n_prune} GSs pruned. Now having {len(model.gauss_params['means'])} GSs.")

            stats["ConsensusDensify/duplicated"] = torch.tensor(float(n_dupli), device=self.device)
            stats["ConsensusDensify/split"] = torch.tensor(float(n_split), device=self.device)
            stats["ConsensusDensify/pruned"] = torch.tensor(float(n_prune), device=self.device)

            state["grad2d"].zero_()
            state["count"].zero_()
            if strategy.refine_scale2d_stop_iter > 0:
                state["radii"].zero_()
            torch.cuda.empty_cache()

        if step % strategy.reset_every == 0:
            from gsplat.strategy.ops import reset_opa

            reset_opa(
                params=model.gauss_params,
                optimizers=model.optimizers,
                state=state,
                value=strategy.prune_opa * 2.0,
            )

        return stats

    def _step_gaussian_batch_refinement(self, step: int, info: Dict[str, torch.Tensor]) -> None:
        model = self.pipeline.model
        strategy = getattr(model, "strategy", None)
        if strategy is None or not hasattr(strategy, "_update_state"):
            raise RuntimeError("Sequential Gaussian batch refinement currently requires the default gsplat strategy.")
        strategy.step_post_backward(
            params=model.gauss_params,
            optimizers=model.optimizers,
            state=model.strategy_state,
            step=step,
            info=info,
            packed=False,
        )

    @profiler.time_function
    def gaussian_batch_train_iteration(self, step: int) -> TRAIN_INTERATION_OUTPUT:
        """Run a true multi-view batch while rendering each view sequentially.

        This uses the Gaussian consensus view sampler and trainable parameter groups, but the gradient is the ordinary
        average of the per-view losses. There is no per-Gaussian visibility filtering or soft-consensus weighting.
        """
        config = self._get_gaussian_consensus_config()
        assert config is not None
        if self.world_size > 1:
            raise RuntimeError(
                "Sequential Gaussian batch training is currently implemented for single-process training only."
            )
        if config.gaussian_consensus_num_views < 1:
            raise RuntimeError("gaussian_consensus_num_views must be at least 1.")

        trainable_groups = self._get_gaussian_consensus_param_groups()
        refine_batch = not config.gaussian_consensus_disable_refinement
        if refine_batch:
            required_groups = {"means", "scales", "quats", "features_dc", "features_rest", "opacities"}
            missing_groups = sorted(required_groups.difference(trainable_groups))
            if missing_groups:
                raise RuntimeError(
                    "Sequential Gaussian batch refinement requires all Gaussian parameter groups to be trainable. "
                    f"Missing groups: {missing_groups}"
                )
        loss_dicts: List[Dict[str, torch.Tensor]] = []
        metrics_dicts: List[Dict[str, torch.Tensor]] = []
        losses: List[torch.Tensor] = []
        refinement_infos: List[Dict[str, torch.Tensor]] = []
        cpu_or_cuda_str: str = self.device.split(":")[0]
        cpu_or_cuda_str = "cpu" if cpu_or_cuda_str == "mps" else cpu_or_cuda_str
        batch_size = config.gaussian_consensus_num_views

        self.optimizers.zero_grad_all()
        with self._only_consensus_groups_require_grad(trainable_groups):
            for ray_bundle, batch in self._iter_gaussian_consensus_train_views(step, config):
                with torch.autocast(device_type=cpu_or_cuda_str, enabled=self.mixed_precision):
                    model_outputs = self.pipeline._model(ray_bundle)
                    metrics_dict = self.pipeline.model.get_metrics_dict(model_outputs, batch)
                    loss_dict = self.pipeline.model.get_loss_dict(model_outputs, batch, metrics_dict)
                    loss = functools.reduce(torch.add, loss_dict.values())
                    loss_for_backward = loss / batch_size

                self.grad_scaler.scale(loss_for_backward).backward()  # type: ignore
                if refine_batch:
                    refinement_infos.append(self.pipeline.model.info)
                loss_dicts.append({key: value.detach() for key, value in loss_dict.items()})
                metrics_dicts.append(metrics_dict)
                losses.append(loss.detach())
                del model_outputs, loss_dict, metrics_dict, loss, loss_for_backward, ray_bundle, batch

        needs_step = [
            group
            for group in trainable_groups
            if step % self.gradient_accumulation_steps[group] == self.gradient_accumulation_steps[group] - 1
        ]
        self.optimizers.optimizer_scaler_step_some(self.grad_scaler, needs_step)

        metrics_dict = self._mean_logged_dicts(metrics_dicts)
        if self.config.log_gradients:
            total_grad = 0
            for tag, value in self.pipeline.model.named_parameters():
                assert tag != "Total"
                if value.grad is not None:
                    grad = value.grad.norm()
                    metrics_dict[f"Gradients/{tag}"] = grad
                    total_grad += grad
            metrics_dict["Gradients/Total"] = cast(torch.Tensor, total_grad)

        scale = self.grad_scaler.get_scale()
        self.grad_scaler.update()
        if scale <= self.grad_scaler.get_scale():
            for group in needs_step:
                if group in self.optimizers.schedulers:
                    self.optimizers.scheduler_step(group)

        if refine_batch and len(refinement_infos) > 0:
            for info in refinement_infos[:-1]:
                self._update_gaussian_batch_refinement_state(info)
            self._step_gaussian_batch_refinement(step, refinement_infos[-1])

        loss_dict = self._mean_logged_dicts(loss_dicts)
        metrics_dict["Batch/num_views"] = torch.tensor(float(batch_size), device=self.device)
        loss = torch.stack(losses).mean()
        return loss, loss_dict, metrics_dict  # type: ignore

    @profiler.time_function
    def train_iteration(self, step: int) -> TRAIN_INTERATION_OUTPUT:
        """Run one iteration with a batch of inputs. Returns dictionary of model losses.

        Args:
            step: Current training step.
        """

        gaussian_consensus_config = self._get_gaussian_consensus_config()
        if gaussian_consensus_config is not None:
            gaussian_consensus_mode = getattr(gaussian_consensus_config, "gaussian_consensus_mode", "consensus")
            if gaussian_consensus_mode == "consensus":
                return self.gaussian_consensus_train_iteration(step)
            if gaussian_consensus_mode == "batch":
                return self.gaussian_batch_train_iteration(step)
            raise RuntimeError(f"Unknown Gaussian consensus training mode: {gaussian_consensus_mode}")

        trainable_groups = self._get_gaussian_trainable_param_groups()
        if trainable_groups is not None:
            model_config = getattr(self.pipeline.model, "config", None)
            refinement_disabled = bool(getattr(model_config, "gaussian_disable_refinement", False))
            if not refinement_disabled:
                required_groups = {"means", "scales", "quats", "features_dc", "features_rest", "opacities"}
                missing_groups = sorted(required_groups.difference(trainable_groups))
                if missing_groups:
                    raise RuntimeError(
                        "Standard Gaussian densification requires all Gaussian parameter groups to be trainable. "
                        "Disable densification or enable these groups: "
                        f"{missing_groups}"
                    )
        active_param_groups = (
            trainable_groups if trainable_groups is not None else list(self.optimizers.parameters.keys())
        )

        needs_zero = [
            group for group in active_param_groups if step % self.gradient_accumulation_steps[group] == 0
        ]
        self.optimizers.zero_grad_some(needs_zero)
        cpu_or_cuda_str: str = self.device.split(":")[0]
        cpu_or_cuda_str = "cpu" if cpu_or_cuda_str == "mps" else cpu_or_cuda_str

        if trainable_groups is None:
            with torch.autocast(device_type=cpu_or_cuda_str, enabled=self.mixed_precision):
                _, loss_dict, metrics_dict = self.pipeline.get_train_loss_dict(step=step)
                loss = functools.reduce(torch.add, loss_dict.values())
        else:
            with self._only_consensus_groups_require_grad(trainable_groups):
                with torch.autocast(device_type=cpu_or_cuda_str, enabled=self.mixed_precision):
                    _, loss_dict, metrics_dict = self.pipeline.get_train_loss_dict(step=step)
                    loss = functools.reduce(torch.add, loss_dict.values())
        self.grad_scaler.scale(loss).backward()  # type: ignore
        needs_step = [
            group
            for group in active_param_groups
            if step % self.gradient_accumulation_steps[group] == self.gradient_accumulation_steps[group] - 1
        ]
        self.optimizers.optimizer_scaler_step_some(self.grad_scaler, needs_step)

        if self.config.log_gradients:
            total_grad = 0
            for tag, value in self.pipeline.model.named_parameters():
                assert tag != "Total"
                if value.grad is not None:
                    grad = value.grad.norm()
                    metrics_dict[f"Gradients/{tag}"] = grad  # type: ignore
                    total_grad += grad

            metrics_dict["Gradients/Total"] = cast(torch.Tensor, total_grad)  # type: ignore

        scale = self.grad_scaler.get_scale()
        self.grad_scaler.update()
        # If the gradient scaler is decreased, no optimization step is performed so we should not step the scheduler.
        if scale <= self.grad_scaler.get_scale():
            if trainable_groups is None:
                self.optimizers.scheduler_step_all(step)
            else:
                for group in needs_step:
                    if group in self.optimizers.schedulers:
                        self.optimizers.scheduler_step(group)

        # Merging loss and metrics dict into a single output.
        return loss, loss_dict, metrics_dict  # type: ignore

    @check_eval_enabled
    @profiler.time_function
    def eval_iteration(self, step: int) -> None:
        """Run one iteration with different batch/image/all image evaluations depending on step size.

        Args:
            step: Current training step.
        """
        # a batch of eval rays
        if step_check(step, self.config.steps_per_eval_batch):
            _, eval_loss_dict, eval_metrics_dict = self.pipeline.get_eval_loss_dict(step=step)
            eval_loss = functools.reduce(torch.add, eval_loss_dict.values())
            writer.put_scalar(name="Eval Loss", scalar=eval_loss, step=step)
            writer.put_dict(name="Eval Loss Dict", scalar_dict=eval_loss_dict, step=step)
            writer.put_dict(name="Eval Metrics Dict", scalar_dict=eval_metrics_dict, step=step)

        # one eval image
        if step_check(step, self.config.steps_per_eval_image):
            with TimeWriter(writer, EventName.TEST_RAYS_PER_SEC, write=False) as test_t:
                metrics_dict, images_dict = self.pipeline.get_eval_image_metrics_and_images(step=step)
            writer.put_time(
                name=EventName.TEST_RAYS_PER_SEC,
                duration=metrics_dict["num_rays"] / test_t.duration,
                step=step,
                avg_over_steps=True,
            )
            writer.put_dict(name="Eval Images Metrics", scalar_dict=metrics_dict, step=step)
            group = "Eval Images"
            for image_name, image in images_dict.items():
                writer.put_image(name=group + "/" + image_name, image=image, step=step)

        # all eval images
        if step_check(step, self.config.steps_per_eval_all_images):
            metrics_dict = self.pipeline.get_average_eval_image_metrics(step=step)
            writer.put_dict(name="Eval Images Metrics Dict (all images)", scalar_dict=metrics_dict, step=step)
