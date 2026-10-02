from __future__ import annotations

from pathlib import Path

import yaml

from .base import GenericSegmentationDataset
from .transforms import TRAIN_STAGE


def load_dataset(yaml_path: str | Path, stage: str = TRAIN_STAGE) -> GenericSegmentationDataset:
    with Path(yaml_path).open("r") as config_file:
        config = yaml.safe_load(config_file)
    return GenericSegmentationDataset(config, stage=stage)
