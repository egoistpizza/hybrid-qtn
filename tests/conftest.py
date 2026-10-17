from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from dataset import TRAIN_STAGE, GenericSegmentationDataset, parse_config


def _write_image(path: Path, h: int, w: int, fill: int = 128) -> None:
    arr = np.full((h, w, 3), fill, dtype=np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)


def _write_mask(path: Path, h: int, w: int, value: int) -> None:
    arr = np.full((h, w), value, dtype=np.uint8)
    Image.fromarray(arr, mode="L").save(path)


@pytest.fixture
def fake_dataset_dir(tmp_path: Path) -> dict:
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    images_dir.mkdir()
    masks_dir.mkdir()

    stems = ["sample_01", "sample_02", "sample_03", "sample_04"]
    sizes = [(64, 64), (48, 80), (100, 60), (32, 32)]
    mask_fills = [255, 0, 255, 200]

    for stem, (h, w), mv in zip(stems, sizes, mask_fills):
        _write_image(images_dir / f"{stem}.jpg", h, w)
        _write_mask(masks_dir / f"{stem}.png", h, w, mv)

    return {
        "images_dir": images_dir,
        "masks_dir": masks_dir,
        "stems": stems,
        "sizes": sizes,
        "mask_fills": mask_fills,
    }


@pytest.fixture
def base_config(fake_dataset_dir: dict) -> dict:
    return {
        "dataset_name": "fake",
        "images_dir": str(fake_dataset_dir["images_dir"]),
        "masks_dir": str(fake_dataset_dir["masks_dir"]),
        "image_ext": ".jpg",
        "mask_ext": ".png",
        "strict_pairing": True,
        "mask_threshold": 127,
    }


@pytest.fixture
def standard_transforms() -> list[dict]:
    return [
        {"name": "Resize", "height": 32, "width": 32},
        {
            "name": "Normalize",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "max_pixel_value": 255.0,
        },
        {"name": "ToTensorV2"},
    ]


@pytest.fixture
def mask_writer():
    return _write_mask


@pytest.fixture
def make_dataset():
    def build(raw_config: dict, stage: str = TRAIN_STAGE) -> GenericSegmentationDataset:
        config = parse_config(raw_config)
        return GenericSegmentationDataset(config.split.source, config.pipelines, stage)

    return build
