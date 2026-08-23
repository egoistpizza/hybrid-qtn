"""Segmentation dataset package."""

from .base import GenericSegmentationDataset
from .loader import load_dataset
from .pairing import pair_by_stem
from .transforms import build_transforms_from_config

__all__ = [
    "GenericSegmentationDataset",
    "load_dataset",
    "pair_by_stem",
    "build_transforms_from_config",
]
