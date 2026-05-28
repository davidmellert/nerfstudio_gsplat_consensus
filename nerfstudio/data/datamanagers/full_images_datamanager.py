# Copyright 2022 The Nerfstudio Team. All rights reserved.
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
Data manager that outputs cameras / images instead of raybundles

Good for things like gaussian splatting which require full cameras instead of the standard ray
paradigm
"""

from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from functools import cached_property
from itertools import islice
from pathlib import Path
from typing import Dict, ForwardRef, Generic, List, Literal, Optional, Tuple, Type, Union, cast, get_args, get_origin

import fpsample
import numpy as np
import torch
from rich.progress import track
from torch.nn import Parameter
from torch.utils.data import DataLoader
from typing_extensions import assert_never

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.configs.dataparser_configs import AnnotatedDataParserUnion
from nerfstudio.data.datamanagers.base_datamanager import DataManager, DataManagerConfig, TDataset
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.utils.data_utils import identity_collate
from nerfstudio.data.utils.dataloaders import ImageBatchStream, _undistort_image, undistort_view
from nerfstudio.utils.misc import get_dict_to_torch, get_orig_class
from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class FullImageDatamanagerConfig(DataManagerConfig):
    _target: Type = field(default_factory=lambda: FullImageDatamanager)
    dataparser: AnnotatedDataParserUnion = field(default_factory=NerfstudioDataParserConfig)
    camera_res_scale_factor: float = 1.0
    """The scale factor for scaling spatial data such as images, mask, semantics
    along with relevant information about camera intrinsics
    """
    eval_num_images_to_sample_from: int = -1
    """Number of images to sample during eval iteration."""
    eval_num_times_to_repeat_images: int = -1
    """When not evaluating on all images, number of iterations before picking
    new images. If -1, never pick new images."""
    cache_images: Literal["cpu", "gpu", "disk"] = "gpu"
    """Where to cache images in memory. 
        - If "cpu", caches images on cpu RAM as pytorch tensors. 
        - If "gpu", caches images on device as pytorch tensors. 
        - If "disk", keeps images on disk which conserves memory. Datamanager will use parallel dataloader"""
    cache_images_type: Literal["uint8", "float32"] = "float32"
    """The image type returned from manager, caching images in uint8 saves memory"""
    max_thread_workers: Optional[int] = None
    """The maximum number of threads to use for caching images. If None, uses all available threads."""
    train_cameras_sampling_strategy: Literal["random", "fps"] = "random"
    """Specifies which sampling strategy is used to generate train cameras, 'random' means sampling 
    uniformly random without replacement, 'fps' means farthest point sampling which is helpful to reduce the artifacts 
    due to oversampling subsets of cameras that are very close to each other."""
    train_cameras_sampling_seed: int = 42
    """Random seed for sampling train cameras. Fixing seed may help reduce variance of trained models across 
    different runs."""
    fps_reset_every: int = 100
    """The number of iterations before one resets fps sampler repeatly, which is essentially drawing fps_reset_every
    samples from the pool of all training cameras without replacement before a new round of sampling starts."""
    dataloader_num_workers: int = 4
    """The number of workers performing the dataloading from either disk/RAM, which 
    includes collating, pixel sampling, unprojecting, ray generation etc."""
    prefetch_factor: Optional[int] = 4
    """The limit number of batches a worker will start loading once an iterator is created. 
    More details are described here: https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader"""
    cache_compressed_images: bool = False
    """If True, cache raw image files as byte strings to RAM."""


class FullImageDatamanager(DataManager, Generic[TDataset]):
    """
    A datamanager that outputs full images and cameras instead of raybundles. This makes the
    datamanager more lightweight since we don't have to do generate rays. Useful for full-image
    training e.g. rasterization pipelines
    """

    config: FullImageDatamanagerConfig
    train_dataset: TDataset
    eval_dataset: TDataset

    def __init__(
        self,
        config: FullImageDatamanagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,
    ):
        if config.cache_images == "disk":
            try:
                torch.multiprocessing.set_start_method("spawn")
            except RuntimeError:
                assert torch.multiprocessing.get_start_method() == "spawn", 'start method must be "spawn"'
        self.config = config
        self.device = device
        self.world_size = world_size
        self.local_rank = local_rank
        self.sampler = None
        self.test_mode = test_mode
        self.test_split = "test" if test_mode in ["test", "inference"] else "val"
        self.dataparser_config = self.config.dataparser
        if self.config.data is not None:
            self.config.dataparser.data = Path(self.config.data)
        else:
            self.config.data = self.config.dataparser.data
        self.dataparser = self.dataparser_config.setup()
        if test_mode == "inference":
            self.dataparser.downscale_factor = 1  # Avoid opening images
        self.includes_time = self.dataparser.includes_time
        self.train_dataparser_outputs: DataparserOutputs = self.dataparser.get_dataparser_outputs(split="train")
        self.train_dataset = self.create_train_dataset()
        self.eval_dataset = self.create_eval_dataset()
        if len(self.train_dataset) > 500 and self.config.cache_images == "gpu":
            CONSOLE.print(
                "Train dataset has over 500 images, overriding cache_images to cpu. If you still get OOM errors or segfault, please consider seting cache_images to 'disk'",
                style="bold yellow",
            )
            self.config.cache_images = "cpu"

        # Some logic to make sure we sample every camera in equal amounts
        self.train_unseen_cameras = self.sample_train_cameras()
        self.eval_unseen_cameras = [i for i in range(len(self.eval_dataset))]
        assert len(self.train_unseen_cameras) > 0, "No data found in dataset"
        super().__init__()

    @staticmethod
    def _set_camera_resolution_from_image(cameras: Cameras, image_idx: int, image: torch.Tensor) -> None:
        """Keep camera intrinsics/render size aligned with the supervision image tensor."""
        image_width = image.shape[1]
        image_height = image.shape[0]
        camera_width = float(cameras.width[image_idx].item())
        camera_height = float(cameras.height[image_idx].item())
        if camera_width > 0 and camera_height > 0:
            scale_x = image_width / camera_width
            scale_y = image_height / camera_height
            cameras.fx[image_idx] *= scale_x
            cameras.cx[image_idx] *= scale_x
            cameras.fy[image_idx] *= scale_y
            cameras.cy[image_idx] *= scale_y
        cameras.width[image_idx] = image_width
        cameras.height[image_idx] = image_height

    def sample_train_cameras(self):
        """Return a list of camera indices sampled using the strategy specified by
        self.config.train_cameras_sampling_strategy"""
        num_train_cameras = len(self.train_dataset)
        if self.config.train_cameras_sampling_strategy == "random":
            if not hasattr(self, "random_generator"):
                self.random_generator = random.Random(self.config.train_cameras_sampling_seed)
            indices = list(range(num_train_cameras))
            self.random_generator.shuffle(indices)
            return indices
        elif self.config.train_cameras_sampling_strategy == "fps":
            if not hasattr(self, "train_unsampled_epoch_count"):
                np.random.seed(self.config.train_cameras_sampling_seed)  # fix random seed of fpsample
                self.train_unsampled_epoch_count = np.zeros(num_train_cameras)
            camera_origins = self.train_dataset.cameras.camera_to_worlds[..., 3].numpy()
            # We concatenate camera origins with weighted train_unsampled_epoch_count because we want to
            # increase the chance to sample camera that hasn't been sampled in consecutive epochs previously.
            # We assume the camera origins are also rescaled, so the weight 0.1 is relative to the scale of scene
            data = np.concatenate(
                (camera_origins, 0.1 * np.expand_dims(self.train_unsampled_epoch_count, axis=-1)), axis=-1
            )
            n = self.config.fps_reset_every
            if num_train_cameras < n:
                CONSOLE.log(
                    f"num_train_cameras={num_train_cameras} is smaller than fps_reset_ever={n}, the behavior of "
                    "camera sampler will be very similar to sampling random without replacement (default setting)."
                )
                n = num_train_cameras
            kdline_fps_samples_idx = fpsample.bucket_fps_kdline_sampling(data, n, h=3)
            self.train_unsampled_epoch_count += 1
            self.train_unsampled_epoch_count[kdline_fps_samples_idx] = 0
            return kdline_fps_samples_idx.tolist()
        else:
            raise ValueError(f"Unknown train camera sampling strategy: {self.config.train_cameras_sampling_strategy}")

    @cached_property
    def cached_train(self) -> List[Dict[str, torch.Tensor]]:
        """Get the training images. Will load and undistort the images the
        first time this (cached) property is accessed."""
        assert self.config.cache_images != "disk", "Can not call _load_images() with `disk` as input"
        return self._load_images("train", cache_images_device=self.config.cache_images)

    @cached_property
    def cached_eval(self) -> List[Dict[str, torch.Tensor]]:
        """Get the eval images. Will load and undistort the images the
        first time this (cached) property is accessed."""
        assert self.config.cache_images != "disk", "Can not call _load_images() with `disk` as input"
        return self._load_images("eval", cache_images_device=self.config.cache_images)

    def _load_images(
        self, split: Literal["train", "eval"], cache_images_device: Literal["cpu", "gpu"]
    ) -> List[Dict[str, torch.Tensor]]:
        undistorted_images: List[Dict[str, torch.Tensor]] = []
        # Which dataset?
        if split == "train":
            dataset = self.train_dataset
        elif split == "eval":
            dataset = self.eval_dataset
        else:
            assert_never(split)

        def undistort_idx(idx: int) -> Dict[str, torch.Tensor]:
            data = dataset.get_data(idx, image_type=self.config.cache_images_type)
            self._set_camera_resolution_from_image(dataset.cameras, idx, data["image"])
            camera = dataset.cameras[idx].reshape(())
            if camera.distortion_params is None or torch.all(camera.distortion_params == 0):
                return data
            K = camera.get_intrinsics_matrices().numpy()
            distortion_params = camera.distortion_params.numpy()
            image = data["image"].numpy()

            K, image, mask = _undistort_image(camera, distortion_params, data, image, K)
            data["image"] = torch.from_numpy(image)
            if mask is not None:
                data["mask"] = mask

            dataset.cameras.fx[idx] = float(K[0, 0])
            dataset.cameras.fy[idx] = float(K[1, 1])
            dataset.cameras.cx[idx] = float(K[0, 2])
            dataset.cameras.cy[idx] = float(K[1, 2])
            dataset.cameras.width[idx] = image.shape[1]
            dataset.cameras.height[idx] = image.shape[0]
            return data

        CONSOLE.log(f"Caching / undistorting {split} images")
        with ThreadPoolExecutor(max_workers=2) as executor:
            undistorted_images = list(
                track(
                    executor.map(
                        undistort_idx,
                        range(len(dataset)),
                    ),
                    description=f"Caching / undistorting {split} images",
                    transient=True,
                    total=len(dataset),
                )
            )
        # Move to device.
        if cache_images_device == "gpu":
            for cache in undistorted_images:
                cache["image"] = cache["image"].to(self.device)
                if "mask" in cache:
                    cache["mask"] = cache["mask"].to(self.device)
                if "depth" in cache:
                    cache["depth"] = cache["depth"].to(self.device)
                self.train_cameras = self.train_dataset.cameras.to(self.device)
        elif cache_images_device == "cpu":
            for cache in undistorted_images:
                cache["image"] = cache["image"].pin_memory()
                if "mask" in cache:
                    cache["mask"] = cache["mask"].pin_memory()
                self.train_cameras = self.train_dataset.cameras
        else:
            assert_never(cache_images_device)
        return undistorted_images

    def create_train_dataset(self) -> TDataset:
        """Sets up the data loaders for training"""
        return self.dataset_type(
            dataparser_outputs=self.train_dataparser_outputs,
            scale_factor=self.config.camera_res_scale_factor,
            cache_compressed_images=self.config.cache_compressed_images,
        )

    def create_eval_dataset(self) -> TDataset:
        """Sets up the data loaders for evaluation"""
        return self.dataset_type(
            dataparser_outputs=self.dataparser.get_dataparser_outputs(split=self.test_split),
            scale_factor=self.config.camera_res_scale_factor,
            cache_compressed_images=self.config.cache_compressed_images,
        )

    @cached_property
    def dataset_type(self) -> Type[TDataset]:
        """Returns the dataset type passed as the generic argument"""
        default: Type[TDataset] = cast(TDataset, TDataset.__default__)  # type: ignore
        orig_class: Type[FullImageDatamanager] = get_orig_class(self, default=None)  # type: ignore
        if type(self) is FullImageDatamanager and orig_class is None:
            return default
        if orig_class is not None and get_origin(orig_class) is FullImageDatamanager:
            return get_args(orig_class)[0]
        # For inherited classes, we need to find the correct type to instantiate
        for base in getattr(self, "__orig_bases__", []):
            if get_origin(base) is FullImageDatamanager:
                for value in get_args(base):
                    if isinstance(value, ForwardRef):
                        if value.__forward_evaluated__:
                            value = value.__forward_value__
                        elif value.__forward_module__ is None:
                            value.__forward_module__ = type(self).__module__
                            value = getattr(value, "_evaluate")(None, None, set())
                    assert isinstance(value, type)
                    if issubclass(value, InputDataset):
                        return cast(Type[TDataset], value)
        return default

    def get_datapath(self) -> Path:
        return self.config.dataparser.data

    def setup_train(self):
        """Sets up the data loaders for training"""
        if self.config.cache_images == "disk":
            self.train_imagebatch_stream = ImageBatchStream(
                input_dataset=self.train_dataset,
                sampling_seed=self.config.train_cameras_sampling_seed,
                cache_images_type=self.config.cache_images_type,
                device=self.device,
                custom_image_processor=self.custom_image_processor,
            )
            self.train_image_dataloader = DataLoader(
                self.train_imagebatch_stream,
                batch_size=1,
                num_workers=self.config.dataloader_num_workers,
                collate_fn=identity_collate,
            )
            self.iter_train_image_dataloader = iter(self.train_image_dataloader)

    def setup_eval(self):
        """Sets up the data loader for evaluation"""
        if self.config.cache_images == "disk":
            self.eval_imagebatch_stream = ImageBatchStream(
                input_dataset=self.eval_dataset,
                sampling_seed=self.config.train_cameras_sampling_seed,
                cache_images_type=self.config.cache_images_type,
                device=self.device,
                custom_image_processor=self.custom_image_processor,
            )
            self.eval_image_dataloader = DataLoader(
                self.eval_imagebatch_stream,
                batch_size=1,
                num_workers=0,  # This must be 0 otherwise there is a crash when trying to pickle custom_image_processor
                collate_fn=identity_collate,
            )
            self.iter_eval_image_dataloader = iter(self.eval_image_dataloader)

    @property
    def fixed_indices_eval_dataloader(self) -> List[Tuple[Cameras, Dict]]:
        """
        Pretends to be the dataloader for evaluation, it returns a list of (camera, data) tuples
        """
        if self.config.cache_images == "disk":
            dataloader = DataLoader(
                self.eval_imagebatch_stream,
                batch_size=1,
                num_workers=0,
                collate_fn=lambda x: x[0],
            )
            return list(islice(dataloader, len(self.eval_dataset)))

        image_indices = [i for i in range(len(self.eval_dataset))]
        data = [d.copy() for d in self.cached_eval]
        _cameras = deepcopy(self.eval_dataset.cameras).to(self.device)
        cameras = []
        for i in image_indices:
            data[i]["image"] = data[i]["image"].to(self.device)
            cameras.append(_cameras[i : i + 1])
        assert len(self.eval_dataset.cameras.shape) == 1, "Assumes single batch dimension"
        return list(zip(cameras, data))

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Get the param groups for the data manager.
        Returns:
            A list of dictionaries containing the data manager's param groups.
        """
        return {}

    def get_train_rays_per_batch(self) -> int:
        """Returns resolution of the image returned from datamanager."""
        camera = self.train_dataset.cameras[0].reshape(())
        return int(camera.width[0].item() * camera.height[0].item())

    def _next_train_image_idx(self) -> int:
        """Returns the next train image index according to the configured train camera sampler."""
        image_idx = int(self.train_unseen_cameras.pop(0))
        if len(self.train_unseen_cameras) == 0:
            self.train_unseen_cameras = self.sample_train_cameras()
        return image_idx

    def _consensus_random_generator(self) -> random.Random:
        if not hasattr(self, "_consensus_random"):
            self._consensus_random = random.Random(self.config.train_cameras_sampling_seed + 17)
        return self._consensus_random

    def get_train_camera_and_data(self, image_idx: int) -> Tuple[Cameras, Dict]:
        """Returns one indexed train camera/image pair from the edited train set."""
        if self.config.cache_images == "disk":
            camera, data = undistort_view(image_idx, self.train_dataset, self.config.cache_images_type)  # type: ignore
            if camera.metadata is None:
                camera.metadata = {}
            camera.metadata["cam_idx"] = image_idx
            if self.custom_image_processor:
                camera, data = self.custom_image_processor(camera, data)
            if camera.metadata is None:
                camera.metadata = {}
            camera.metadata["cam_idx"] = image_idx
            camera = camera.to(self.device)
            data = get_dict_to_torch(data, self.device)
            return camera, data

        data = self.cached_train[image_idx]
        # We're going to copy to make sure we don't mutate the cached dictionary.
        # This can cause a memory leak: https://github.com/nerfstudio-project/nerfstudio/issues/3335
        data = data.copy()
        data["image"] = data["image"].to(self.device)

        assert len(self.train_cameras.shape) == 1, "Assumes single batch dimension"
        camera = self.train_cameras[image_idx : image_idx + 1].to(self.device)
        if camera.metadata is None:
            camera.metadata = {}
        camera.metadata["cam_idx"] = image_idx
        return camera, data

    def _get_pose_neighbor_indices(
        self,
        anchor_idx: int,
        neighbor_pool_size: int,
        position_weight: float,
        direction_weight: float,
    ) -> List[int]:
        camera_to_worlds = self.train_dataset.cameras.camera_to_worlds.float()
        origins = camera_to_worlds[..., 3]
        directions = torch.nn.functional.normalize(camera_to_worlds[..., :3, 2], dim=-1)
        anchor_origin = origins[anchor_idx]
        anchor_direction = directions[anchor_idx]

        position_distances = torch.linalg.norm(origins - anchor_origin, dim=-1)
        nonzero_position_distances = position_distances[position_distances > 1e-8]
        if nonzero_position_distances.numel() > 0:
            position_scale = nonzero_position_distances.median().clamp_min(1e-8)
        else:
            position_scale = torch.ones((), dtype=position_distances.dtype, device=position_distances.device)
        position_score = position_distances / position_scale

        direction_dot = torch.clamp((directions * anchor_direction).sum(dim=-1), -1.0, 1.0)
        direction_score = 1.0 - direction_dot
        pose_score = position_weight * position_score + direction_weight * direction_score
        pose_score[anchor_idx] = torch.inf

        ordered = torch.argsort(pose_score).tolist()
        if neighbor_pool_size > 0:
            ordered = ordered[:neighbor_pool_size]
        return [int(idx) for idx in ordered if idx != anchor_idx]

    def sample_train_view_indices(
        self,
        num_views: int,
        sampling_strategy: Literal["global", "pose_neighborhood"] = "global",
        neighbor_pool_size: int = 16,
        position_weight: float = 1.0,
        direction_weight: float = 0.25,
    ) -> List[int]:
        """Samples train image indices for one multi-view Gaussian consensus step."""
        if num_views < 1:
            raise ValueError("num_views must be at least 1")

        anchor_idx = self._next_train_image_idx()
        image_indices = [anchor_idx]
        if num_views == 1 or len(self.train_dataset) == 1:
            return image_indices

        rng = self._consensus_random_generator()
        if sampling_strategy == "pose_neighborhood":
            neighbor_indices = self._get_pose_neighbor_indices(
                anchor_idx=anchor_idx,
                neighbor_pool_size=neighbor_pool_size,
                position_weight=position_weight,
                direction_weight=direction_weight,
            )
            num_neighbors = min(num_views - 1, len(neighbor_indices))
            if num_neighbors > 0:
                image_indices.extend(rng.sample(neighbor_indices, k=num_neighbors))
        elif sampling_strategy != "global":
            raise ValueError(f"Unknown train view sampling strategy: {sampling_strategy}")

        seen = set(image_indices)
        max_unique = min(num_views, len(self.train_dataset))
        while len(image_indices) < max_unique:
            image_idx = self._next_train_image_idx()
            if image_idx not in seen:
                image_indices.append(image_idx)
                seen.add(image_idx)

        while len(image_indices) < num_views:
            image_indices.append(anchor_idx)
        return image_indices

    def next_train_views(
        self,
        step: int,
        num_views: int,
        sampling_strategy: Literal["global", "pose_neighborhood"] = "global",
        neighbor_pool_size: int = 16,
        position_weight: float = 1.0,
        direction_weight: float = 0.25,
    ) -> List[Tuple[Cameras, Dict]]:
        """Returns one anchor train view plus optional neighboring train views."""
        self.train_count += 1
        image_indices = self.sample_train_view_indices(
            num_views=num_views,
            sampling_strategy=sampling_strategy,
            neighbor_pool_size=neighbor_pool_size,
            position_weight=position_weight,
            direction_weight=direction_weight,
        )
        return [self.get_train_camera_and_data(image_idx) for image_idx in image_indices]

    def next_train(self, step: int) -> Tuple[Cameras, Dict]:
        """Returns the next training batch
        Returns a Camera instead of raybundle"""
        self.train_count += 1
        if self.config.cache_images == "disk":
            camera, data = next(self.iter_train_image_dataloader)[0]
            camera = camera.to(self.device)
            data = get_dict_to_torch(data, self.device)
            return camera, data

        image_idx = self._next_train_image_idx()
        return self.get_train_camera_and_data(image_idx)

    def next_eval(self, step: int) -> Tuple[Cameras, Dict]:
        """Returns the next evaluation batch
        Returns a Camera instead of raybundle"""
        self.eval_count += 1
        if self.config.cache_images == "disk":
            camera, data = next(self.iter_eval_image_dataloader)[0]
            camera = camera.to(self.device)
            data = get_dict_to_torch(data, self.device)
            return camera, data

        return self.next_eval_image(step=step)

    def next_eval_image(self, step: int) -> Tuple[Cameras, Dict]:
        """Returns the next evaluation batch
        Returns a Camera instead of raybundle
        TODO: Make sure this logic is consistent with the vanilladatamanager"""
        if self.config.cache_images == "disk":
            camera, data = next(self.iter_eval_image_dataloader)[0]
            return camera, data
        image_idx = self.eval_unseen_cameras.pop(random.randint(0, len(self.eval_unseen_cameras) - 1))
        # Make sure to re-populate the unseen cameras list if we have exhausted it
        if len(self.eval_unseen_cameras) == 0:
            self.eval_unseen_cameras = [i for i in range(len(self.eval_dataset))]
        data = self.cached_eval[image_idx]
        data = data.copy()
        data["image"] = data["image"].to(self.device)
        assert len(self.eval_dataset.cameras.shape) == 1, "Assumes single batch dimension"
        camera = self.eval_dataset.cameras[image_idx : image_idx + 1].to(self.device)
        return camera, data

    def custom_image_processor(self, camera: Cameras, data: Dict) -> Tuple[Cameras, Dict]:
        """An API to add latents, metadata, or other further customization an camera-and-image view dataloading process that is parallelized"""
        return camera, data
