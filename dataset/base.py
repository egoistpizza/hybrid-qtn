from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .pairing import pair_by_stem
from .transforms import (
    DEFAULT_AUGMENTATION_SEED,
    EVAL_STAGE,
    TRAIN_STAGE,
    assert_deterministic_pipeline,
    build_transforms_from_config,
    stage_transform_specs,
)


class GenericSegmentationDataset(Dataset):
    def __init__(self, config: dict[str, Any], stage: str = TRAIN_STAGE) -> None:
        self.config = config
        self.dataset_name: str = config["dataset_name"]
        self.mask_threshold: int = int(config.get("mask_threshold", 127))
        self.align_mask_to_image: bool = bool(config.get("align_mask_to_image", False))
        self.samples: list[tuple[str, str, str]] = pair_by_stem(
            images_dir=config["images_dir"],
            masks_dir=config["masks_dir"],
            image_ext=config["image_ext"],
            mask_ext=config["mask_ext"],
            strict_pairing=bool(config.get("strict_pairing", True)),
        )
        self._apply_stage(stage)

    def _apply_stage(self, stage: str) -> None:
        specs = stage_transform_specs(self.config, stage)
        if stage == EVAL_STAGE:
            assert_deterministic_pipeline(specs)
        self.stage = stage
        self.transform = build_transforms_from_config(specs, seed=DEFAULT_AUGMENTATION_SEED)

    def with_stage(self, stage: str) -> GenericSegmentationDataset:
        staged = copy.copy(self)
        staged._apply_stage(stage)
        return staged

    def reseed_augmentations(self, seed: int) -> None:
        self.transform.set_random_seed(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: str) -> np.ndarray:
        with Image.open(path) as image:
            return np.array(image.convert("RGB"), dtype=np.uint8)

    def _load_mask(self, path: str) -> np.ndarray:
        with Image.open(path) as mask:
            raw = np.array(mask.convert("L"), dtype=np.uint8)
        return (raw > self.mask_threshold).astype(np.uint8)

    def _match_image_frame(self, mask: np.ndarray, image: np.ndarray, sample_id: str) -> np.ndarray:
        height, width = image.shape[:2]
        if mask.shape == (height, width):
            return mask
        if not self.align_mask_to_image:
            raise ValueError(
                f"{self.dataset_name}/{sample_id}: mask {mask.shape} and image {(height, width)} differ in size; "
                "set align_mask_to_image to resize such masks to the image frame"
            )
        return np.array(Image.fromarray(mask).resize((width, height), Image.Resampling.NEAREST))

    @staticmethod
    def _as_image_tensor(image: Any) -> torch.Tensor:
        if isinstance(image, torch.Tensor):
            return image.float()
        tensor = torch.from_numpy(np.ascontiguousarray(image))
        if tensor.ndim == 3 and tensor.shape[-1] in (1, 3):
            tensor = tensor.permute(2, 0, 1).contiguous()
        return tensor.float()

    @staticmethod
    def _as_mask_tensor(mask: Any) -> torch.Tensor:
        tensor = mask if isinstance(mask, torch.Tensor) else torch.from_numpy(np.ascontiguousarray(mask))
        tensor = tensor.float()
        if tensor.ndim == 2 or (tensor.ndim == 3 and tensor.shape[0] != 1):
            tensor = tensor.unsqueeze(0)
        return tensor

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_path, mask_path, sample_id = self.samples[index]
        image = self._load_image(image_path)
        mask = self._match_image_frame(self._load_mask(mask_path), image, sample_id)
        transformed = self.transform(image=image, mask=mask)

        return {
            "image": self._as_image_tensor(transformed["image"]),
            "mask": self._as_mask_tensor(transformed["mask"]),
            "metadata": {
                "image_path": image_path,
                "mask_path": mask_path,
                "sample_id": sample_id,
                "original_size": (int(image.shape[0]), int(image.shape[1])),
                "dataset_name": self.dataset_name,
                "index": index,
            },
        }
