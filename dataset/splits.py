from __future__ import annotations

import copy
from pathlib import Path

import torch
import yaml
from torch.utils.data import ConcatDataset, Subset

from models import DEFAULT_IMAGE_SIZE

from .base import GenericSegmentationDataset
from .groups import grouped_split_indices, groups_for_samples
from .loader import load_dataset
from .transforms import build_transforms_from_config

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"
DEFAULT_VAL_FRACTION = 0.2
DEFAULT_SPLIT_SEED = 42
REQUIRED_KEYS = ("images_dir", "masks_dir")
# A dataset shipped with an official split (e.g. DUTS-TR / DUTS-TE) names its
# validation folders here instead of being split randomly.
VAL_DIR_KEYS = ("val_images_dir", "val_masks_dir")
PIPELINE_KEYS = ("transforms", "train_transforms", "val_transforms")

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

def dataset_image_size(config: dict, name: str = "dataset") -> int:
    """Model input size declared by a dataset config (default 512).

    Every transform pipeline in the config must end at that size, judged by its
    last step carrying ``height``/``width`` (Resize, RandomCrop, CenterCrop).
    """
    image_size = int(config.get("image_size", DEFAULT_IMAGE_SIZE))
    pipelines = {key: config[key] for key in PIPELINE_KEYS if config.get(key)}

    if not pipelines:
        raise ValueError(f"{name}: no transform pipeline, so the output size is not fixed")

    for key, pipeline in pipelines.items():
        sized = [step for step in pipeline if "height" in step and "width" in step]
        if not sized:
            raise ValueError(
                f"{name}: {key} has no step with height/width, so its output "
                f"cannot be checked against image_size={image_size}"
            )
        last = sized[-1]
        if (last["height"], last["width"]) != (image_size, image_size):
            raise ValueError(
                f"{name}: image_size is {image_size} but {key} ends at "
                f"{last['name']} {last['height']}x{last['width']}"
            )

    return image_size

def resolve_image_size(names: list[str], config_dir: str | Path | None = None) -> int:
    """Shared input size of one or more datasets. Joint training needs them to agree."""
    sizes = {
        name: dataset_image_size(yaml.safe_load(resolve_config(name, config_dir).read_text()), name)
        for name in names
    }
    distinct = set(sizes.values())
    if len(distinct) > 1:
        raise ValueError(f"Datasets trained together must share image_size, got {sizes}")
    return distinct.pop()

def make_splits(
    name: str,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    seed: int = DEFAULT_SPLIT_SEED,
    config_dir: str | Path | None = None,
) -> tuple[Subset, Subset]:
    base_dataset = load_dataset(resolve_config(name, config_dir))
    config = base_dataset.config
    
    group_spec = config.get("group_map")
    val_keys_present = [key for key in VAL_DIR_KEYS if key in config]
    val_base = None
    
    if val_keys_present:
        if len(val_keys_present) != len(VAL_DIR_KEYS):
            raise ValueError(f"{name}: {' and '.join(VAL_DIR_KEYS)} must be set together")
        if group_spec is not None:
            raise ValueError(f"{name}: group_map and {VAL_DIR_KEYS[0]} are mutually exclusive")
        
        val_base = GenericSegmentationDataset({
            **config,
            "images_dir": config["val_images_dir"],
            "masks_dir": config["val_masks_dir"],
        })
        train_idx = list(range(len(base_dataset)))
        val_idx = list(range(len(val_base)))
    elif group_spec is not None:
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
    val_dataset = val_base if val_base is not None else copy.copy(base_dataset)

    if "train_transforms" in config and "val_transforms" in config:
        train_dataset.transform = build_transforms_from_config(config["train_transforms"])
        val_dataset.transform = build_transforms_from_config(config["val_transforms"])

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