"""Segmentation dataset package."""

from .base import GenericSegmentationDataset
from .loader import load_dataset
from .pairing import pair_by_stem
from .splits import (
    build_train_val,
    discover_dataset_configs,
    make_splits,
    parse_dataset_arg,
    resolve_config,
    slugify_dataset_arg,
)
from .transforms import build_transforms_from_config
from .groups import grouped_split_indices, groups_for_samples, load_group_map

__all__ = [
    "GenericSegmentationDataset",
    "load_dataset",
    "pair_by_stem",
    "build_transforms_from_config",
    "build_train_val",
    "discover_dataset_configs",
    "make_splits",
    "parse_dataset_arg",
    "resolve_config",
    "slugify_dataset_arg",
    "grouped_split_indices",
    "groups_for_samples",
    "load_group_map",
]
