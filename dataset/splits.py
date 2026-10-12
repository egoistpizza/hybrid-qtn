from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml
from torch.utils.data import ConcatDataset, Dataset, Subset

from .base import GenericSegmentationDataset
from .groups import grouped_partition_indices, groups_for_samples
from .loader import read_config
from .predefined import PREDEFINED_SPLIT_KEYS, declares_predefined_splits, predefined_split_datasets
from .transforms import EVAL_STAGE, TRAIN_STAGE

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"
DEFAULT_SPLIT_SEED = 42
DEFAULT_TEST_SET = "test"
REQUIRED_KEYS = ("images_dir", "masks_dir")
FOREGROUND_MAP_METRICS_KEY = "foreground_map_metrics"


@dataclass(frozen=True)
class SplitFractions:
    val: float = 0.15
    test: float = 0.15

    def __post_init__(self) -> None:
        if not (0.0 < self.val < 1.0 and 0.0 < self.test < 1.0 and self.val + self.test < 1.0):
            raise ValueError(f"Invalid split fractions val={self.val}, test={self.test}")

    @property
    def train(self) -> float:
        return 1.0 - self.val - self.test

    def sizes(self, total: int) -> tuple[int, int, int]:
        val_size = round(self.val * total)
        test_size = round(self.test * total)
        return total - val_size - test_size, val_size, test_size


DEFAULT_SPLIT_FRACTIONS = SplitFractions()


@dataclass(frozen=True)
class DatasetSplits:
    train: Dataset
    val: Dataset
    test_sets: Mapping[str, Dataset]

    @property
    def test(self) -> Dataset:
        if len(self.test_sets) != 1:
            raise ValueError(f"Expected a single test set, found {sorted(self.test_sets)}")
        return next(iter(self.test_sets.values()))


def _is_dataset_config(config: dict[str, Any]) -> bool:
    return all(key in config for key in REQUIRED_KEYS) or all(key in config for key in PREDEFINED_SPLIT_KEYS)


def discover_dataset_configs(config_dir: str | Path | None = None) -> dict[str, Path]:
    directory = Path(config_dir) if config_dir is not None else CONFIG_DIR
    found: dict[str, Path] = {}
    for path in sorted(directory.glob("*.yaml")):
        try:
            config = yaml.safe_load(path.read_text())
        except yaml.YAMLError:
            continue
        if isinstance(config, dict) and _is_dataset_config(config):
            found[path.stem] = path
    return found


def resolve_config(name: str, config_dir: str | Path | None = None) -> Path:
    available = discover_dataset_configs(config_dir)
    if name not in available:
        directory = Path(config_dir) if config_dir is not None else CONFIG_DIR
        listed = ", ".join(sorted(available)) or "(none found)"
        raise ValueError(
            f"Unknown dataset {name!r}. Available: {listed}. "
            f"Add a YAML with {' and '.join(REQUIRED_KEYS)}, or with {', '.join(PREDEFINED_SPLIT_KEYS)}, under {directory}."
        )
    return available[name]


def parse_dataset_arg(value: str) -> list[str]:
    names = [part.strip() for part in str(value).split(",") if part.strip()]
    if not names:
        raise ValueError("--dataset must name at least one dataset config")
    return names


def slugify_dataset_arg(names: list[str]) -> str:
    return "+".join(names)


def uses_foreground_map_metrics(names: list[str], config_dir: str | Path | None = None) -> bool:
    return any(read_config(resolve_config(name, config_dir)).get(FOREGROUND_MAP_METRICS_KEY, False) for name in names)


def _partition_indices(
    dataset: GenericSegmentationDataset,
    fractions: SplitFractions,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    group_spec = dataset.config.get("group_map")
    if group_spec is not None:
        groups = groups_for_samples(
            (sample_id for _, _, sample_id in dataset.samples),
            group_spec,
            root=CONFIG_DIR.parent,
        )
        return grouped_partition_indices(groups, (fractions.val, fractions.test), seed)

    permutation = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed)).tolist()
    train_size, val_size, _ = fractions.sizes(len(dataset))
    return (
        sorted(permutation[:train_size]),
        sorted(permutation[train_size : train_size + val_size]),
        sorted(permutation[train_size + val_size :]),
    )


def make_splits(
    name: str,
    fractions: SplitFractions = DEFAULT_SPLIT_FRACTIONS,
    seed: int = DEFAULT_SPLIT_SEED,
    config_dir: str | Path | None = None,
) -> DatasetSplits:
    config = read_config(resolve_config(name, config_dir))
    if declares_predefined_splits(config):
        train, val, test_sets = predefined_split_datasets(config, seed)
        return DatasetSplits(train=train, val=val, test_sets=test_sets)

    train_dataset = GenericSegmentationDataset(config, stage=TRAIN_STAGE)
    eval_dataset = train_dataset.with_stage(EVAL_STAGE)
    train_indices, val_indices, test_indices = _partition_indices(train_dataset, fractions, seed)
    return DatasetSplits(
        train=Subset(train_dataset, train_indices),
        val=Subset(eval_dataset, val_indices),
        test_sets={DEFAULT_TEST_SET: Subset(eval_dataset, test_indices)},
    )


def _merge_test_sets(splits: list[DatasetSplits]) -> dict[str, Dataset]:
    grouped: dict[str, list[Dataset]] = {}
    for split in splits:
        for name, dataset in split.test_sets.items():
            grouped.setdefault(name, []).append(dataset)
    return {name: parts[0] if len(parts) == 1 else ConcatDataset(parts) for name, parts in grouped.items()}


def build_splits(
    names: list[str],
    fractions: SplitFractions = DEFAULT_SPLIT_FRACTIONS,
    seed: int = DEFAULT_SPLIT_SEED,
    config_dir: str | Path | None = None,
) -> DatasetSplits:
    splits = [make_splits(name, fractions, seed, config_dir) for name in names]
    if len(splits) == 1:
        return splits[0]
    return DatasetSplits(
        train=ConcatDataset([split.train for split in splits]),
        val=ConcatDataset([split.val for split in splits]),
        test_sets=_merge_test_sets(splits),
    )
