from __future__ import annotations

import warnings
from typing import Any

import torch
from torch.utils.data import ConcatDataset, Dataset, Subset

from .base import GenericSegmentationDataset
from .transforms import EVAL_STAGE, TRAIN_STAGE

SOURCES_KEY = "sources"
TRAIN_SOURCES_KEY = "train_sources"
TEST_SOURCES_KEY = "test_sources"
VALIDATION_FRACTION_KEY = "validation_fraction"
PREDEFINED_SPLIT_KEYS = (SOURCES_KEY, TRAIN_SOURCES_KEY, TEST_SOURCES_KEY, VALIDATION_FRACTION_KEY)
SOURCE_DIRECTORY_KEYS = ("images_dir", "masks_dir")
INCOMPATIBLE_KEYS = ("group_map", *SOURCE_DIRECTORY_KEYS)


def declares_predefined_splits(config: dict[str, Any]) -> bool:
    present = [key for key in PREDEFINED_SPLIT_KEYS if key in config]
    if present and len(present) != len(PREDEFINED_SPLIT_KEYS):
        missing = sorted(set(PREDEFINED_SPLIT_KEYS) - set(present))
        raise ValueError(f"{config.get('dataset_name')}: predefined splits also need {missing}")
    return bool(present)


def _validate(config: dict[str, Any]) -> None:
    name = config.get("dataset_name")
    conflicting = [key for key in INCOMPATIBLE_KEYS if key in config]
    if conflicting:
        raise ValueError(f"{name}: {conflicting} cannot be combined with predefined splits")

    sources = config[SOURCES_KEY]
    train_sources, test_sources = config[TRAIN_SOURCES_KEY], config[TEST_SOURCES_KEY]
    if not train_sources or not test_sources:
        raise ValueError(f"{name}: {TRAIN_SOURCES_KEY} and {TEST_SOURCES_KEY} must each name at least one source")
    referenced = [*train_sources, *test_sources]
    if len(set(referenced)) != len(referenced):
        raise ValueError(f"{name}: a source may appear only once across {TRAIN_SOURCES_KEY} and {TEST_SOURCES_KEY}")
    undefined = [source for source in referenced if source not in sources]
    if undefined:
        raise ValueError(f"{name}: sources {undefined} are not defined under {SOURCES_KEY}")
    incomplete = [source for source in referenced if not all(key in sources[source] for key in SOURCE_DIRECTORY_KEYS)]
    if incomplete:
        raise ValueError(f"{name}: sources {incomplete} need both {' and '.join(SOURCE_DIRECTORY_KEYS)}")
    if not 0.0 < config[VALIDATION_FRACTION_KEY] < 1.0:
        raise ValueError(f"{name}: {VALIDATION_FRACTION_KEY} must lie in (0, 1), got {config[VALIDATION_FRACTION_KEY]}")


def source_dataset(config: dict[str, Any], source: str, stage: str) -> GenericSegmentationDataset:
    shared = {key: value for key, value in config.items() if key not in PREDEFINED_SPLIT_KEYS}
    return GenericSegmentationDataset(
        {**shared, **config[SOURCES_KEY][source], "dataset_name": f"{config['dataset_name']}/{source}"},
        stage=stage,
    )


def _sample_ids(dataset: GenericSegmentationDataset) -> set[str]:
    return {sample_id for _, _, sample_id in dataset.samples}


def _indices_outside(dataset: GenericSegmentationDataset, held_out_ids: set[str]) -> list[int]:
    kept = [index for index, (_, _, sample_id) in enumerate(dataset.samples) if sample_id not in held_out_ids]
    excluded = len(dataset) - len(kept)
    if excluded:
        warnings.warn(
            f"Excluding {excluded} training sample(s) of {dataset.dataset_name} that also appear in a test source",
            UserWarning,
            stacklevel=2,
        )
    return kept


def carve_validation(indices: list[int], fraction: float, seed: int) -> tuple[list[int], list[int]]:
    order = torch.randperm(len(indices), generator=torch.Generator().manual_seed(seed)).tolist()
    validation_size = round(fraction * len(indices))
    shuffled = [indices[position] for position in order]
    return sorted(shuffled[validation_size:]), sorted(shuffled[:validation_size])


def predefined_split_datasets(
    config: dict[str, Any], seed: int
) -> tuple[Dataset, Dataset, dict[str, GenericSegmentationDataset]]:
    _validate(config)
    test_sets = {source: source_dataset(config, source, EVAL_STAGE) for source in config[TEST_SOURCES_KEY]}
    held_out_ids = set().union(*(_sample_ids(dataset) for dataset in test_sets.values()))

    train_parts: list[Subset] = []
    validation_parts: list[Subset] = []
    for source in config[TRAIN_SOURCES_KEY]:
        train_dataset = source_dataset(config, source, TRAIN_STAGE)
        eligible = _indices_outside(train_dataset, held_out_ids)
        train_indices, validation_indices = carve_validation(eligible, config[VALIDATION_FRACTION_KEY], seed)
        train_parts.append(Subset(train_dataset, train_indices))
        validation_parts.append(Subset(train_dataset.with_stage(EVAL_STAGE), validation_indices))

    return ConcatDataset(train_parts), ConcatDataset(validation_parts), test_sets
