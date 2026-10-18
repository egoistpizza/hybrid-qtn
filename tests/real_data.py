from pathlib import Path

import pytest

from dataset import TRAIN_STAGE, DatasetConfig, GenericSegmentationDataset, load_config


REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"


REAL_DATASETS = {
    "kvasir_seg": ("data/kvasir-seg/Kvasir-SEG/images", 1000),
    "cvc_clinicdb": ("data/cvc-clinicdb/CVC-ClinicDB/Original", 612),
    "cvc_clinicdb_affine": ("data/cvc-clinicdb/CVC-ClinicDB/Original", 612),
    "mass_roads": ("data/mass_roads/images", 1171),
}


def require_on_disk(name: str) -> DatasetConfig:
    images_dir, _ = REAL_DATASETS[name]
    if not (REPO_ROOT / images_dir).is_dir():
        pytest.skip(f"{name} not on disk")
    return load_config(name, CONFIG_DIR)


def whole_dataset(config: DatasetConfig, stage: str = TRAIN_STAGE) -> GenericSegmentationDataset:
    return GenericSegmentationDataset(config.split.source, config.pipelines, stage)
