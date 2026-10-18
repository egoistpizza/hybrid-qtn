"""YAML-driven segmentation datasets, deterministic splits and seeded DataLoaders."""

from .base import GenericSegmentationDataset
from .config import (
    DEFAULT_SPLIT_FRACTIONS,
    DatasetConfig,
    GroupedSplit,
    GroupMap,
    Pipelines,
    PredefinedSplit,
    RandomSplit,
    SourceConfig,
    SplitFractions,
    discover_dataset_configs,
    load_config,
    parse_config,
)
from .groups import grouped_partition_indices, groups_for_samples, load_group_map
from .pairing import pair_by_stem
from .predefined import carve_validation
from .splits import (
    DEFAULT_TEST_SET,
    DatasetSplits,
    EvaluationSet,
    build_splits,
    make_splits,
    parse_dataset_arg,
    slugify_dataset_arg,
)
from .loading import build_loader, iter_segmentation_datasets, seed_dataloader_worker
from .transforms import (
    EVAL_STAGE,
    TRAIN_STAGE,
    Denormalization,
    assert_deterministic_pipeline,
    build_transforms_from_config,
)

__all__ = [
    "GenericSegmentationDataset",
    "DatasetConfig",
    "SourceConfig",
    "Pipelines",
    "RandomSplit",
    "GroupedSplit",
    "GroupMap",
    "PredefinedSplit",
    "SplitFractions",
    "DEFAULT_SPLIT_FRACTIONS",
    "discover_dataset_configs",
    "load_config",
    "parse_config",
    "pair_by_stem",
    "build_transforms_from_config",
    "assert_deterministic_pipeline",
    "Denormalization",
    "TRAIN_STAGE",
    "EVAL_STAGE",
    "DEFAULT_TEST_SET",
    "DatasetSplits",
    "EvaluationSet",
    "build_splits",
    "make_splits",
    "parse_dataset_arg",
    "slugify_dataset_arg",
    "carve_validation",
    "grouped_partition_indices",
    "groups_for_samples",
    "load_group_map",
    "build_loader",
    "iter_segmentation_datasets",
    "seed_dataloader_worker",
]
