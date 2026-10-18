"""Typed dataset configs, parsed and validated once from the YAML files under ``configs/``.

A config names its image sources, its split policy and its train / eval pipelines:

- a single ``images_dir`` / ``masks_dir`` source split randomly (:class:`RandomSplit`),
  or by the groups of a ``group_map`` CSV (:class:`GroupedSplit`);
- named ``sources`` with official ``train_sources`` / ``test_sources`` and a
  ``validation_fraction`` (:class:`PredefinedSplit`).

Datasets are addressed by the stem of their YAML file. Relative paths in a config are
resolved against the repository root, independent of the working directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .transforms import (
    EVAL_STAGE,
    STAGES,
    TENSOR_CONVERSION,
    TRAIN_STAGE,
    Denormalization,
    assert_deterministic_pipeline,
    denormalization,
    output_size,
    spec_name,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"

SOURCE_DIRECTORY_KEYS = ("images_dir", "masks_dir")
SOURCE_FILE_KEYS = ("image_ext", "mask_ext", "strict_pairing", "mask_threshold", "align_mask_to_image")
SOURCE_KEYS = (*SOURCE_DIRECTORY_KEYS, *SOURCE_FILE_KEYS)
PIPELINE_KEYS = ("train_transforms", "eval_transforms", "transforms")
PREDEFINED_SPLIT_KEYS = ("sources", "train_sources", "test_sources", "validation_fraction")
GROUP_MAP_KEY = "group_map"
GROUP_MAP_KEYS = ("csv", "sample_column", "group_column")
FOREGROUND_MAP_METRICS_KEY = "foreground_map_metrics"
KNOWN_KEYS = frozenset(
    {"dataset_name", FOREGROUND_MAP_METRICS_KEY, GROUP_MAP_KEY, *SOURCE_KEYS, *PIPELINE_KEYS, *PREDEFINED_SPLIT_KEYS}
)


@dataclass(frozen=True)
class SplitFractions:
    """Validation and test fractions of a holdout split; training receives the remainder."""

    val: float = 0.15
    test: float = 0.15

    def __post_init__(self) -> None:
        if not (0.0 < self.val < 1.0 and 0.0 < self.test < 1.0 and self.val + self.test < 1.0):
            raise ValueError(f"Invalid split fractions val={self.val}, test={self.test}")

    @property
    def train(self) -> float:
        return 1.0 - self.val - self.test

    def sizes(self, total: int) -> tuple[int, int, int]:
        """Train, validation and test sizes that add up to ``total``."""
        val_size = round(self.val * total)
        test_size = round(self.test * total)
        return total - val_size - test_size, val_size, test_size


DEFAULT_SPLIT_FRACTIONS = SplitFractions()


@dataclass(frozen=True)
class SourceConfig:
    """One image / mask directory pair.

    ``name`` is echoed as ``dataset_name`` in the sample metadata. Raw mask values above
    ``mask_threshold`` are foreground. With ``align_mask_to_image``, a mask whose size
    differs from its image is resized (nearest) instead of raising.
    """

    name: str
    images_dir: Path
    masks_dir: Path
    image_ext: str
    mask_ext: str
    strict_pairing: bool = True
    mask_threshold: int = 127
    align_mask_to_image: bool = False


@dataclass(frozen=True)
class Pipelines:
    """Albumentations specs of both stages; both end with ``ToTensorV2`` at ``image_size``.

    ``denormalization`` inverts the eval pipeline's ``Normalize`` step for display.
    """

    train: tuple[Mapping[str, Any], ...]
    eval: tuple[Mapping[str, Any], ...]
    image_size: int
    denormalization: Denormalization

    def for_stage(self, stage: str) -> tuple[Mapping[str, Any], ...]:
        if stage not in STAGES:
            raise ValueError(f"Unknown stage {stage!r}. Expected one of {STAGES}.")
        return self.train if stage == TRAIN_STAGE else self.eval


@dataclass(frozen=True)
class RandomSplit:
    source: SourceConfig
    fractions: SplitFractions = DEFAULT_SPLIT_FRACTIONS


@dataclass(frozen=True)
class GroupMap:
    """CSV assigning every sample id to a group, e.g. a colonoscopy sequence."""

    csv: Path
    sample_column: str = "sample_id"
    group_column: str = "group_id"


@dataclass(frozen=True)
class GroupedSplit:
    """Holdout split in which every group of ``group_map`` lands in one partition; see ``dataset/groups.py``."""

    source: SourceConfig
    group_map: GroupMap
    fractions: SplitFractions = DEFAULT_SPLIT_FRACTIONS


@dataclass(frozen=True)
class PredefinedSplit:
    """Official training and test sources; validation is carved from each training source."""

    train_sources: tuple[SourceConfig, ...]
    test_sources: Mapping[str, SourceConfig]
    validation_fraction: float


SplitPolicy = RandomSplit | GroupedSplit | PredefinedSplit


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    split: SplitPolicy
    pipelines: Pipelines
    foreground_map_metrics: bool = False

    @property
    def image_size(self) -> int:
        return self.pipelines.image_size


def _resolve(path: Any) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def _source(name: str, entries: Mapping[str, Any]) -> SourceConfig:
    missing = [key for key in (*SOURCE_DIRECTORY_KEYS, "image_ext", "mask_ext") if key not in entries]
    if missing:
        raise ValueError(f"{name}: missing {missing}")
    return SourceConfig(
        name=name,
        images_dir=_resolve(entries["images_dir"]),
        masks_dir=_resolve(entries["masks_dir"]),
        image_ext=str(entries["image_ext"]),
        mask_ext=str(entries["mask_ext"]),
        strict_pairing=bool(entries.get("strict_pairing", True)),
        mask_threshold=int(entries.get("mask_threshold", 127)),
        align_mask_to_image=bool(entries.get("align_mask_to_image", False)),
    )


def _stage_pipeline(name: str, raw: Mapping[str, Any], stage: str) -> tuple[Mapping[str, Any], ...]:
    specs = tuple(raw.get(f"{stage}_transforms", raw.get("transforms")) or ())
    if not specs or spec_name(specs[-1]) != TENSOR_CONVERSION:
        raise ValueError(f"{name}: the {stage} pipeline must end with {TENSOR_CONVERSION}")
    return specs


def _pipelines(name: str, raw: Mapping[str, Any]) -> Pipelines:
    train, evaluation = _stage_pipeline(name, raw, TRAIN_STAGE), _stage_pipeline(name, raw, EVAL_STAGE)
    assert_deterministic_pipeline(evaluation)
    sizes = {stage: output_size(specs) for stage, specs in ((TRAIN_STAGE, train), (EVAL_STAGE, evaluation))}
    if len(set(sizes.values())) != 1:
        raise ValueError(f"{name}: train and eval pipelines produce different sizes {sizes}")
    height, width = sizes[TRAIN_STAGE]
    if height != width:
        raise ValueError(f"{name}: pipelines must produce square inputs, got {height}x{width}")
    return Pipelines(train=train, eval=evaluation, image_size=height, denormalization=denormalization(evaluation))


def _predefined_split(name: str, raw: Mapping[str, Any]) -> PredefinedSplit:
    missing = [key for key in PREDEFINED_SPLIT_KEYS if key not in raw]
    if missing:
        raise ValueError(f"{name}: predefined splits also need {missing}")
    conflicting = [key for key in (GROUP_MAP_KEY, *SOURCE_DIRECTORY_KEYS) if key in raw]
    if conflicting:
        raise ValueError(f"{name}: {conflicting} cannot be combined with predefined splits")

    sources = raw["sources"]
    train_names, test_names = list(raw["train_sources"]), list(raw["test_sources"])
    if not train_names or not test_names:
        raise ValueError(f"{name}: train_sources and test_sources must each name at least one source")
    referenced = [*train_names, *test_names]
    if len(set(referenced)) != len(referenced):
        raise ValueError(f"{name}: a source may appear only once across train_sources and test_sources")
    undefined = [source for source in referenced if source not in sources]
    if undefined:
        raise ValueError(f"{name}: sources {undefined} are not defined under sources")
    incomplete = [source for source in referenced if not all(key in sources[source] for key in SOURCE_DIRECTORY_KEYS)]
    if incomplete:
        raise ValueError(f"{name}: sources {incomplete} need both {' and '.join(SOURCE_DIRECTORY_KEYS)}")
    for source in referenced:
        unknown = sorted(set(sources[source]) - set(SOURCE_KEYS))
        if unknown:
            raise ValueError(f"{name}: source {source!r} has unknown keys {unknown}")
    validation_fraction = float(raw["validation_fraction"])
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError(f"{name}: validation_fraction must lie in (0, 1), got {validation_fraction}")

    shared = {key: raw[key] for key in SOURCE_FILE_KEYS if key in raw}

    def source(source_name: str) -> SourceConfig:
        return _source(f"{name}/{source_name}", {**shared, **sources[source_name]})

    return PredefinedSplit(
        train_sources=tuple(source(source_name) for source_name in train_names),
        test_sources={source_name: source(source_name) for source_name in test_names},
        validation_fraction=validation_fraction,
    )


def _holdout_split(name: str, raw: Mapping[str, Any]) -> RandomSplit | GroupedSplit:
    source = _source(name, raw)
    group_map = raw.get(GROUP_MAP_KEY)
    if group_map is None:
        return RandomSplit(source)
    if not isinstance(group_map, Mapping) or "csv" not in group_map:
        raise ValueError(f"{name}: {GROUP_MAP_KEY} must be a mapping with a csv entry")
    unknown = sorted(set(group_map) - set(GROUP_MAP_KEYS))
    if unknown:
        raise ValueError(f"{name}: {GROUP_MAP_KEY} has unknown keys {unknown}")
    columns = {key: str(group_map[key]) for key in ("sample_column", "group_column") if key in group_map}
    return GroupedSplit(source, GroupMap(csv=_resolve(group_map["csv"]), **columns))


def parse_config(raw: Mapping[str, Any]) -> DatasetConfig:
    """Validate a parsed YAML mapping and return its typed config.

    Raises:
        ValueError: On unknown or missing keys, inconsistent sources, a stochastic eval
            pipeline, or pipelines that disagree on their output size.
    """
    if not isinstance(raw, Mapping) or "dataset_name" not in raw:
        raise ValueError(f"A dataset config must be a mapping with a dataset_name, got {raw!r}")
    name = str(raw["dataset_name"])
    unknown = sorted(set(raw) - KNOWN_KEYS)
    if unknown:
        raise ValueError(f"{name}: unknown config keys {unknown}")
    is_predefined = any(key in raw for key in PREDEFINED_SPLIT_KEYS)
    return DatasetConfig(
        name=name,
        split=_predefined_split(name, raw) if is_predefined else _holdout_split(name, raw),
        pipelines=_pipelines(name, raw),
        foreground_map_metrics=bool(raw.get(FOREGROUND_MAP_METRICS_KEY, False)),
    )


def discover_dataset_configs(config_dir: str | Path | None = None) -> dict[str, Path]:
    """Dataset YAMLs in ``config_dir`` by file stem."""
    directory = Path(config_dir) if config_dir is not None else CONFIG_DIR
    return {path.stem: path for path in sorted(directory.glob("*.yaml"))}


def load_config(name: str, config_dir: str | Path | None = None) -> DatasetConfig:
    """Parse the config ``<config_dir>/<name>.yaml``."""
    available = discover_dataset_configs(config_dir)
    if name not in available:
        listed = ", ".join(sorted(available)) or "(none found)"
        raise ValueError(f"Unknown dataset {name!r}. Available: {listed}.")
    return parse_config(yaml.safe_load(available[name].read_text()))
