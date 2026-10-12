"""Dataset construction from YAML configs; a dataset is defined by its config, not by code."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .base import GenericSegmentationDataset
from .transforms import TRAIN_STAGE


def read_config(yaml_path: str | Path) -> dict[str, Any]:
    with Path(yaml_path).open("r") as config_file:
        return yaml.safe_load(config_file)


def load_dataset(yaml_path: str | Path, stage: str = TRAIN_STAGE) -> GenericSegmentationDataset:
    return GenericSegmentationDataset(read_config(yaml_path), stage=stage)
