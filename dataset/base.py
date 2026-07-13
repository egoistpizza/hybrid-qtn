"""Generic YAML-based segmentation dataset.

Output contract (per ``__getitem__``)::

    {
        "image": torch.FloatTensor,   # (C, H, W), float32
        "mask":  torch.FloatTensor,   # (1, H, W), float32, values in {0.0, 1.0}
        "metadata": {
            "image_path":    str,
            "mask_path":     str,
            "sample_id":     str,             # filename stem
            "original_size": tuple[int, int], # (H0, W0) pre-transform
            "dataset_name":  str,
            "index":         int,
        },
    }
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .pairing import pair_by_stem
from .transforms import build_transforms_from_config


class GenericSegmentationDataset(Dataset):
    """Config-driven binary segmentation dataset.

    Expected ``config`` keys:

    - ``images_dir`` (str)
    - ``masks_dir`` (str)
    - ``image_ext`` (str, e.g. ``".jpg"``)
    - ``mask_ext`` (str, e.g. ``".jpg"`` or ``".png"``)
    - ``dataset_name`` (str)
    - ``strict_pairing`` (bool, default True)
    - ``mask_threshold`` (int, 0–255, default 127) — foreground cutoff for the raw mask
    - ``transforms`` (list[dict], optional) — see ``build_transforms_from_config``
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.dataset_name: str = config["dataset_name"]
        self.mask_threshold: int = int(config.get("mask_threshold", 127))

        self.samples: list[tuple[str, str, str]] = pair_by_stem(
            images_dir=config["images_dir"],
            masks_dir=config["masks_dir"],
            image_ext=config["image_ext"],
            mask_ext=config["mask_ext"],
            strict_pairing=bool(config.get("strict_pairing", True)),
        )

        self.transform = build_transforms_from_config(config.get("transforms"))

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: str) -> np.ndarray:
        with Image.open(path) as im:
            return np.array(im.convert("RGB"), dtype=np.uint8)

    def _load_mask(self, path: str) -> np.ndarray:
        with Image.open(path) as im:
            raw = np.array(im.convert("L"), dtype=np.uint8)
        return (raw > self.mask_threshold).astype(np.uint8)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_path, mask_path, sample_id = self.samples[index]

        image = self._load_image(image_path)
        mask = self._load_mask(mask_path)
        original_size = (int(image.shape[0]), int(image.shape[1]))

        out = self.transform(image=image, mask=mask)
        image_t, mask_t = out["image"], out["mask"]

        if not isinstance(image_t, torch.Tensor):
            image_t = torch.from_numpy(np.ascontiguousarray(image_t))
            if image_t.ndim == 3 and image_t.shape[-1] in (1, 3):
                image_t = image_t.permute(2, 0, 1).contiguous()
        image_t = image_t.float()

        if not isinstance(mask_t, torch.Tensor):
            mask_t = torch.from_numpy(np.ascontiguousarray(mask_t))
        mask_t = mask_t.float()
        if mask_t.ndim == 2:
            mask_t = mask_t.unsqueeze(0)
        elif mask_t.ndim == 3 and mask_t.shape[0] != 1:
            mask_t = mask_t.unsqueeze(0)

        return {
            "image": image_t,
            "mask": mask_t,
            "metadata": {
                "image_path": image_path,
                "mask_path": mask_path,
                "sample_id": sample_id,
                "original_size": original_size,
                "dataset_name": self.dataset_name,
                "index": index,
            },
        }
