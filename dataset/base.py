"""Config-driven binary segmentation dataset.

Each item is a dict::

    {
        "image": FloatTensor (C, H, W),
        "mask": FloatTensor (1, H, W) with values in {0, 1},
        "metadata": {image_path, mask_path, sample_id, original_size, dataset_name, index},
    }

``sample_id`` is the filename stem and ``original_size`` the ``(H, W)`` of the image on disk.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .config import Pipelines, SourceConfig
from .pairing import pair_by_stem
from .transforms import DEFAULT_AUGMENTATION_SEED, TRAIN_STAGE, build_transforms_from_config


class GenericSegmentationDataset(Dataset):
    """Image/mask pairs of one source matched by filename stem, transformed by the pipeline of one stage.

    Args:
        source: Directories, extensions and mask handling of the source.
        pipelines: Train and eval pipelines; both end with ``ToTensorV2``.
        stage: ``train`` or ``eval``.
    """

    def __init__(self, source: SourceConfig, pipelines: Pipelines, stage: str = TRAIN_STAGE) -> None:
        self.source = source
        self.pipelines = pipelines
        self.samples: list[tuple[str, str, str]] = pair_by_stem(
            images_dir=source.images_dir,
            masks_dir=source.masks_dir,
            image_ext=source.image_ext,
            mask_ext=source.mask_ext,
            strict_pairing=source.strict_pairing,
        )
        self._apply_stage(stage)

    def _apply_stage(self, stage: str) -> None:
        self.transform = build_transforms_from_config(self.pipelines.for_stage(stage), seed=DEFAULT_AUGMENTATION_SEED)
        self.stage = stage

    def with_stage(self, stage: str) -> GenericSegmentationDataset:
        """Shallow copy sharing the sample list but using the pipeline of ``stage``."""
        staged = copy.copy(self)
        staged._apply_stage(stage)
        return staged

    def reseed_augmentations(self, seed: int) -> None:
        """Reseed the Albumentations RNG, which does not follow the global seeds."""
        self.transform.set_random_seed(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: str) -> np.ndarray:
        with Image.open(path) as image:
            return np.array(image.convert("RGB"), dtype=np.uint8)

    def _load_mask(self, path: str) -> np.ndarray:
        with Image.open(path) as mask:
            raw = np.array(mask.convert("L"), dtype=np.uint8)
        return (raw > self.source.mask_threshold).astype(np.uint8)

    def _match_image_frame(self, mask: np.ndarray, image: np.ndarray, sample_id: str) -> np.ndarray:
        """Return the mask in the image frame; resize it (nearest) only if the source allows it."""
        height, width = image.shape[:2]
        if mask.shape == (height, width):
            return mask
        if not self.source.align_mask_to_image:
            raise ValueError(
                f"{self.source.name}/{sample_id}: mask {mask.shape} and image {(height, width)} differ in size; "
                "set align_mask_to_image to resize such masks to the image frame"
            )
        return np.array(Image.fromarray(mask).resize((width, height), Image.Resampling.NEAREST))

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_path, mask_path, sample_id = self.samples[index]
        image = self._load_image(image_path)
        mask = self._match_image_frame(self._load_mask(mask_path), image, sample_id)
        transformed = self.transform(image=image, mask=mask)

        return {
            "image": transformed["image"].float(),
            "mask": transformed["mask"].unsqueeze(0).float(),
            "metadata": {
                "image_path": image_path,
                "mask_path": mask_path,
                "sample_id": sample_id,
                "original_size": (int(image.shape[0]), int(image.shape[1])),
                "dataset_name": self.source.name,
                "index": index,
            },
        }
