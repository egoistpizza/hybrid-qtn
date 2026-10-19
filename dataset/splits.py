"""Train / validation / test splits shared by training and evaluation.

The split seed is fixed and independent of the training seed, so every run of a dataset
and its later evaluation see the same partitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch
from torch.utils.data import ConcatDataset, Dataset, Subset

from .base import GenericSegmentationDataset
from .config import DatasetConfig, GroupedSplit, PredefinedSplit, SplitFractions, load_config
from .groups import grouped_kfold_indices, grouped_partition_indices, groups_for_samples
from .predefined import predefined_split_datasets
from .transforms import EVAL_STAGE, TRAIN_STAGE, Denormalization

DEFAULT_SPLIT_SEED = 42
DEFAULT_TEST_SET = "test"


@dataclass(frozen=True)
class EvaluationSet:
    """A validation or test set, whether it is scored with the foreground-map metrics, and
    how its network inputs map back to displayable images.
    """

    dataset: Dataset
    foreground_maps: bool
    denormalization: Denormalization


@dataclass(frozen=True)
class DatasetSplits:
    """Partitions of one or more datasets at a common input size.

    ``test_sets`` maps test-set names to sets: a single ``"test"`` entry for holdout
    splits, or one entry per official test source for predefined splits.
    """

    train: Dataset
    val: EvaluationSet
    test_sets: Mapping[str, EvaluationSet]
    image_size: int


def parse_dataset_arg(value: str) -> list[str]:
    """Split a comma-separated ``--dataset`` value into config names."""
    names = [part.strip() for part in str(value).split(",") if part.strip()]
    if not names:
        raise ValueError("--dataset must name at least one dataset config")
    return names


def slugify_dataset_arg(names: list[str]) -> str:
    """Dataset part of a run name; joint training joins the names with ``+``."""
    return "+".join(names)


def _random_partition_indices(total: int, fractions: SplitFractions, seed: int) -> tuple[list[int], list[int], list[int]]:
    permutation = torch.randperm(total, generator=torch.Generator().manual_seed(seed)).tolist()
    train_size, val_size, _ = fractions.sizes(total)
    return (
        sorted(permutation[:train_size]),
        sorted(permutation[train_size : train_size + val_size]),
        sorted(permutation[train_size + val_size :]),
    )


def _holdout_split_datasets(
    config: DatasetConfig, seed: int, fold: int | None
) -> tuple[Dataset, Dataset, dict[str, Dataset]]:
    split = config.split
    if fold is not None and not (isinstance(split, GroupedSplit) and split.folds):
        raise ValueError(f"{config.name}: --fold needs a grouped config with cross_validation_folds")
    train_dataset = GenericSegmentationDataset(split.source, config.pipelines, stage=TRAIN_STAGE)
    eval_dataset = train_dataset.with_stage(EVAL_STAGE)
    if isinstance(split, GroupedSplit):
        groups = groups_for_samples((sample_id for _, _, sample_id in train_dataset.samples), split.group_map)
        if fold is None:
            train_indices, val_indices, test_indices = grouped_partition_indices(
                groups, (split.fractions.val, split.fractions.test), seed
            )
        else:
            train_indices, val_indices, test_indices = grouped_kfold_indices(
                groups, split.folds, fold, split.fractions.val, seed
            )
    else:
        train_indices, val_indices, test_indices = _random_partition_indices(len(train_dataset), split.fractions, seed)
    return (
        Subset(train_dataset, train_indices),
        Subset(eval_dataset, val_indices),
        {DEFAULT_TEST_SET: Subset(eval_dataset, test_indices)},
    )


def make_splits(config: DatasetConfig, seed: int = DEFAULT_SPLIT_SEED, fold: int | None = None) -> DatasetSplits:
    """Split one dataset by its split policy, or by one cross-validation ``fold``.

    Training uses the ``train`` pipeline; validation and test use the deterministic
    ``eval`` pipeline.

    Raises:
        ValueError: If ``fold`` is given for a config without ``cross_validation_folds``.
    """
    if isinstance(config.split, PredefinedSplit):
        if fold is not None:
            raise ValueError(f"{config.name}: --fold needs a grouped config with cross_validation_folds")
        train, val, test_sets = predefined_split_datasets(config, seed)
    else:
        train, val, test_sets = _holdout_split_datasets(config, seed, fold)
    def evaluation_set(dataset: Dataset) -> EvaluationSet:
        return EvaluationSet(dataset, config.foreground_map_metrics, config.pipelines.denormalization)

    return DatasetSplits(
        train=train,
        val=evaluation_set(val),
        test_sets={name: evaluation_set(dataset) for name, dataset in test_sets.items()},
        image_size=config.image_size,
    )


def _concatenate(sets: list[EvaluationSet]) -> EvaluationSet:
    if len(sets) == 1:
        return sets[0]
    denormalizations = {evaluation_set.denormalization for evaluation_set in sets}
    if len(denormalizations) != 1:
        raise ValueError(f"Jointly evaluated datasets must share their normalization, got {denormalizations}")
    return EvaluationSet(
        ConcatDataset([evaluation_set.dataset for evaluation_set in sets]),
        any(evaluation_set.foreground_maps for evaluation_set in sets),
        denormalizations.pop(),
    )


def build_splits(
    names: list[str],
    seed: int = DEFAULT_SPLIT_SEED,
    config_dir: str | Path | None = None,
    fold: int | None = None,
) -> DatasetSplits:
    """Split each named dataset and concatenate the partitions for joint training.

    Test sets that share a name are concatenated; differently named sets stay separate.
    ``fold`` selects one cross-validation fold of every dataset (see :func:`make_splits`).

    Raises:
        ValueError: If the datasets produce different input sizes.
    """
    splits = [make_splits(load_config(name, config_dir), seed, fold) for name in names]
    if len(splits) == 1:
        return splits[0]
    sizes = {name: split.image_size for name, split in zip(names, splits)}
    if len(set(sizes.values())) != 1:
        raise ValueError(f"Jointly trained datasets must share an input size, got {sizes}")

    grouped: dict[str, list[EvaluationSet]] = {}
    for split in splits:
        for name, evaluation_set in split.test_sets.items():
            grouped.setdefault(name, []).append(evaluation_set)
    return DatasetSplits(
        train=ConcatDataset([split.train for split in splits]),
        val=_concatenate([split.val for split in splits]),
        test_sets={name: _concatenate(sets) for name, sets in grouped.items()},
        image_size=splits[0].image_size,
    )
