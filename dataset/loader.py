"""YAML-driven segmentation dataset loader.

One loader serves every dataset: ``GenericSegmentationDataset`` is fully
config-driven, so a dataset's identity lives in its YAML file, not in Python.
Adding a dataset means adding a config under ``configs/``, not a new module.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from .base import GenericSegmentationDataset


def load_dataset(yaml_path: str | Path) -> GenericSegmentationDataset:
    """Load a segmentation dataset from a YAML config file."""
    yaml_path = Path(yaml_path)
    with yaml_path.open("r") as f:
        config = yaml.safe_load(f)
    return GenericSegmentationDataset(config)
