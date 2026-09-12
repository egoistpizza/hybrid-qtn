from __future__ import annotations

import copy
from pathlib import Path

import torch
import yaml
from torch.utils.data import ConcatDataset, Subset

from .groups import grouped_split_indices, groups_for_samples
from .loader import load_dataset
from .transforms import build_transforms_from_config

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"
DEFAULT_VAL_FRACTION = 0.2
DEFAULT_SPLIT_SEED = 42
REQUIRED_KEYS = ("images_dir", "masks_dir")

def discover_dataset_configs(config_dir: str | Path | None = None) -> dict[str, Path]:
    directory = Path(config_dir) if config_dir is not None else CONFIG_DIR
    found: dict[str, Path] = {}
    
    for path in sorted(directory.glob("*.yaml")):
        try:
            config = yaml.safe_load(path.read_text())
        except yaml.YAMLError:
            continue
            
        if isinstance(config, dict) and all(k in config for k in REQUIRED_KEYS):
            found[path.stem] = path
            
    return found

def resolve_config(name: str, config_dir: str | Path | None = None) -> Path:
    available = discover_dataset_configs(config_dir)
    
    if name not in available:
        directory = Path(config_dir) if config_dir is not None else CONFIG_DIR
        listed = ", ".join(sorted(available)) or "(none found)"
        raise ValueError(
            f"Unknown dataset {name!r}. Available: {listed}. "
            f"Add a YAML with {' and '.join(REQUIRED_KEYS)} under {directory}."
        )
        
    return available[name]

def parse_dataset_arg(value: str) -> list[str]:
    names = [part.strip() for part in str(value).split(",") if part.strip()]
    if not names:
        raise ValueError("--dataset must name at least one dataset config")
    return names

def slugify_dataset_arg(names: list[str]) -> str:
    return "+".join(names)

def make_splits(
    name: str,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    seed: int = DEFAULT_SPLIT_SEED,
    config_dir: str | Path | None = None,
) -> tuple[Subset, Subset]:
    base_dataset = load_dataset(resolve_config(name, config_dir))
    
    group_spec = base_dataset.config.get("group_map")
    
    if group_spec is not None:
        groups = groups_for_samples(
            (sample_id for _, _, sample_id in base_dataset.samples),
            group_spec,
            root=CONFIG_DIR.parent,
        )
        train_idx, val_idx = grouped_split_indices(groups, val_fraction, seed)
    else:
        train_size = int((1.0 - val_fraction) * len(base_dataset))
        val_size = len(base_dataset) - train_size
        
        subsets = torch.utils.data.random_split(
            base_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(seed),
        )
        train_idx, val_idx = subsets[0].indices, subsets[1].indices

    train_dataset = base_dataset
    val_dataset = copy.copy(base_dataset)

    if "train_transforms" in base_dataset.config and "val_transforms" in base_dataset.config:
        train_dataset.transform = build_transforms_from_config(base_dataset.config["train_transforms"])
        val_dataset.transform = build_transforms_from_config(base_dataset.config["val_transforms"])

    return Subset(train_dataset, train_idx), Subset(val_dataset, val_idx)

def build_train_val(
    names: list[str],
    val_fraction: float = DEFAULT_VAL_FRACTION,
    seed: int = DEFAULT_SPLIT_SEED,
    config_dir: str | Path | None = None,
):
    splits = [make_splits(n, val_fraction, seed, config_dir) for n in names]
    
    if len(splits) == 1:
        return splits[0]
        
    return (
        ConcatDataset([train for train, _ in splits]),
        ConcatDataset([val for _, val in splits]),
    )