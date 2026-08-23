"""Dataset config discovery and train/val splitting.

Shared by ``train.py`` and ``evaluate.py`` so the two cannot drift: an
evaluation run must see exactly the validation split its training run held out.

Datasets are addressed by the stem of their YAML under ``configs/``, so adding
a dataset means adding a config file — no code change, no new choices list.
"""

from __future__ import annotations

from pathlib import Path

import torch
import yaml
from torch.utils.data import ConcatDataset, Subset

from .loader import load_dataset

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"

DEFAULT_VAL_FRACTION = 0.2
DEFAULT_SPLIT_SEED = 42

# Keys that mark a YAML as a dataset config rather than, say, a hyperparameter sweep.
_REQUIRED_KEYS = ("images_dir", "masks_dir")


def discover_dataset_configs(config_dir: str | Path | None = None) -> dict[str, Path]:
    """Map config stem -> path for every dataset YAML under ``configs/``."""
    directory = Path(config_dir) if config_dir is not None else CONFIG_DIR
    found: dict[str, Path] = {}
    for path in sorted(directory.glob("*.yaml")):
        try:
            config = yaml.safe_load(path.read_text())
        except yaml.YAMLError:
            continue
        if isinstance(config, dict) and all(k in config for k in _REQUIRED_KEYS):
            found[path.stem] = path
    return found


def resolve_config(name: str, config_dir: str | Path | None = None) -> Path:
    """Look up a dataset config by name, with an actionable error if absent."""
    available = discover_dataset_configs(config_dir)
    if name not in available:
        directory = Path(config_dir) if config_dir is not None else CONFIG_DIR
        listed = ", ".join(sorted(available)) or "(none found)"
        raise ValueError(
            f"Unknown dataset {name!r}. Available: {listed}. "
            f"Add a YAML with {' and '.join(_REQUIRED_KEYS)} under {directory}."
        )
    return available[name]


def parse_dataset_arg(value: str) -> list[str]:
    """Parse a comma-separated ``--dataset`` value into config names."""
    names = [part.strip() for part in str(value).split(",") if part.strip()]
    if not names:
        raise ValueError("--dataset must name at least one dataset config")
    return names


def slugify_dataset_arg(names: list[str]) -> str:
    """Filesystem/run-name-safe label for one or more datasets."""
    return "+".join(names)


def make_splits(
    name: str,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    seed: int = DEFAULT_SPLIT_SEED,
    config_dir: str | Path | None = None,
) -> tuple[Subset, Subset]:
    """Deterministic train/val split for a single dataset.

    ``train_size`` uses truncation, not rounding, to stay bit-identical to the
    splits produced before this helper existed — changing it would invalidate
    every checkpoint trained against the old arithmetic.
    """
    dataset = load_dataset(resolve_config(name, config_dir))
    train_size = int((1.0 - val_fraction) * len(dataset))
    val_size = len(dataset) - train_size
    return torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )


def build_train_val(
    names: list[str],
    val_fraction: float = DEFAULT_VAL_FRACTION,
    seed: int = DEFAULT_SPLIT_SEED,
    config_dir: str | Path | None = None,
):
    """Train/val datasets for one or more datasets.

    Each dataset is split first and the halves concatenated afterwards, so the
    validation half stays separable per-dataset for reporting.
    """
    splits = [make_splits(n, val_fraction, seed, config_dir) for n in names]
    if len(splits) == 1:
        return splits[0]
    return (
        ConcatDataset([train for train, _ in splits]),
        ConcatDataset([val for _, val in splits]),
    )
