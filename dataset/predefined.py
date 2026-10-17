"""Splits for benchmarks that ship official training and test folders.

Validation is carved from each training source, so the official test sets are never
used for model selection, and each test source is reported as its own test set.
"""

from __future__ import annotations

import warnings

import torch
from torch.utils.data import ConcatDataset, Dataset, Subset

from .base import GenericSegmentationDataset
from .config import DatasetConfig
from .transforms import EVAL_STAGE, TRAIN_STAGE


def _sample_ids(dataset: GenericSegmentationDataset) -> set[str]:
    return {sample_id for _, _, sample_id in dataset.samples}


def _indices_outside(dataset: GenericSegmentationDataset, held_out_ids: set[str]) -> list[int]:
    """Indices of samples whose stem does not occur in a test source."""
    kept = [index for index, (_, _, sample_id) in enumerate(dataset.samples) if sample_id not in held_out_ids]
    excluded = len(dataset) - len(kept)
    if excluded:
        warnings.warn(
            f"Excluding {excluded} training sample(s) of {dataset.source.name} that also appear in a test source",
            UserWarning,
            stacklevel=2,
        )
    return kept


def carve_validation(indices: list[int], fraction: float, seed: int) -> tuple[list[int], list[int]]:
    """Randomly move ``round(fraction * len(indices))`` indices to validation.

    Returns:
        Sorted ``(train, validation)`` index lists.
    """
    order = torch.randperm(len(indices), generator=torch.Generator().manual_seed(seed)).tolist()
    validation_size = round(fraction * len(indices))
    shuffled = [indices[position] for position in order]
    return sorted(shuffled[validation_size:]), sorted(shuffled[:validation_size])


def predefined_split_datasets(
    config: DatasetConfig, seed: int
) -> tuple[Dataset, Dataset, dict[str, GenericSegmentationDataset]]:
    """Build the training set, the validation set and the named test sets of a predefined split.

    Training samples whose stem also appears in a test source are excluded with a
    warning before validation is carved, so no test image reaches training or
    model selection.
    """
    split = config.split
    test_sets = {
        name: GenericSegmentationDataset(source, config.pipelines, stage=EVAL_STAGE)
        for name, source in split.test_sources.items()
    }
    held_out_ids = set().union(*(_sample_ids(dataset) for dataset in test_sets.values()))

    train_parts: list[Subset] = []
    validation_parts: list[Subset] = []
    for source in split.train_sources:
        train_dataset = GenericSegmentationDataset(source, config.pipelines, stage=TRAIN_STAGE)
        eligible = _indices_outside(train_dataset, held_out_ids)
        train_indices, validation_indices = carve_validation(eligible, split.validation_fraction, seed)
        train_parts.append(Subset(train_dataset, train_indices))
        validation_parts.append(Subset(train_dataset.with_stage(EVAL_STAGE), validation_indices))

    return ConcatDataset(train_parts), ConcatDataset(validation_parts), test_sets
