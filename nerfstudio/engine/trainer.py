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
import os
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import DefaultDict, Dict, List, Literal, Optional, Tuple, Type, cast

import torch
import viser
from rich import box, style
from rich.panel import Panel
from rich.table import Table
from torch.cuda.amp.grad_scaler import GradScaler

from nerfstudio.configs.experiment_config import ExperimentConfig
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.pipelines.base_pipeline import VanillaPipeline
from nerfstudio.utils import profiler, writer
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
        refine_consensus = not config.gaussian_consensus_disable_refinement
        if refine_consensus:
            required_groups = {"means", "scales", "quats", "features_dc", "features_rest", "opacities"}
            missing_groups = sorted(required_groups.difference(trainable_groups))
            if missing_groups:
                raise RuntimeError(
                    "Gaussian consensus densification requires all Gaussian parameter groups to be trainable. "
                    f"Missing groups: {missing_groups}"
                )

        consensus_accumulators = {
            group: self._new_gaussian_consensus_accumulator(self.optimizers.parameters[group][0])
            for group in trainable_groups
        }
        loss_dicts: List[Dict[str, torch.Tensor]] = []
        metrics_dicts: List[Dict[str, torch.Tensor]] = []
        losses: List[torch.Tensor] = []
        cpu_or_cuda_str: str = self.device.split(":")[0]
        cpu_or_cuda_str = "cpu" if cpu_or_cuda_str == "mps" else cpu_or_cuda_str

        self.optimizers.zero_grad_all()
        with self._only_consensus_groups_require_grad(trainable_groups):
            for ray_bundle, batch in self._iter_gaussian_consensus_train_views(step, config):
                with torch.autocast(device_type=cpu_or_cuda_str, enabled=self.mixed_precision):
                    model_outputs = self.pipeline._model(ray_bundle)
                    metrics_dict = self.pipeline.model.get_metrics_dict(model_outputs, batch)
                    loss_dict = self.pipeline.model.get_loss_dict(model_outputs, batch, metrics_dict)
                    loss = functools.reduce(torch.add, loss_dict.values())

                self.grad_scaler.scale(loss).backward()  # type: ignore
                for group in trainable_groups:
                    param = self.optimizers.parameters[group][0]
                    self._update_gaussian_consensus_accumulator(
                        param, consensus_accumulators[group], param.grad, config
                    )
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
            consensus_grad, stats = self._finalize_gaussian_consensus_group(
                group, param, consensus_accumulators[group], config
            )
            param.grad = consensus_grad
            consensus_stats.update(stats)

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

        needs_zero = [
            group for group in self.optimizers.parameters.keys() if step % self.gradient_accumulation_steps[group] == 0
        ]
        self.optimizers.zero_grad_some(needs_zero)
        cpu_or_cuda_str: str = self.device.split(":")[0]
        cpu_or_cuda_str = "cpu" if cpu_or_cuda_str == "mps" else cpu_or_cuda_str

        with torch.autocast(device_type=cpu_or_cuda_str, enabled=self.mixed_precision):
            _, loss_dict, metrics_dict = self.pipeline.get_train_loss_dict(step=step)
            loss = functools.reduce(torch.add, loss_dict.values())
        self.grad_scaler.scale(loss).backward()  # type: ignore
        needs_step = [
            group
            for group in self.optimizers.parameters.keys()
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
            self.optimizers.scheduler_step_all(step)

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
