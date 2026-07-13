"""Kvasir-SEG loader — thin YAML wrapper around ``GenericSegmentationDataset``."""

from __future__ import annotations

from pathlib import Path

import yaml

from .base import GenericSegmentationDataset


def load_kvasir_seg(yaml_path: str | Path) -> GenericSegmentationDataset:
    """Load the Kvasir-SEG dataset from a YAML config file."""
    yaml_path = Path(yaml_path)
    with yaml_path.open("r") as f:
        config = yaml.safe_load(f)
    return GenericSegmentationDataset(config)
