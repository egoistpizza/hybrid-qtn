"""Tests for the YAML-driven segmentation dataloader.

Run from the project root:

    pip install pytest
    python -m pytest tests/dataloader_test.py -v

Semantic categories (one test class per concern):

    TestPairing            — pairing.pair_by_stem behavior
    TestTransforms         — transforms.build_transforms_from_config
    TestOutputContract     — GenericSegmentationDataset output shape/dtype/keys
    TestConfigLoader       — kvasir_seg.load_kvasir_seg (YAML -> dataset)
    TestDataLoaderBatching — torch DataLoader collation into batches
    TestKvasirIntegration  — real Kvasir-SEG on disk (auto-skips if missing)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is importable so `from dataset import ...` works
# regardless of the working directory pytest was launched from.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pytest
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader

from dataset import (
    GenericSegmentationDataset,
    build_transforms_from_config,
    load_kvasir_seg,
    pair_by_stem,
)


# ---------------------------------------------------------------------------
# Fixtures — synthetic on-disk dataset
# ---------------------------------------------------------------------------

def _write_image(path: Path, h: int, w: int, fill: int = 128) -> None:
    arr = np.full((h, w, 3), fill, dtype=np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)


def _write_mask(path: Path, h: int, w: int, value: int) -> None:
    arr = np.full((h, w), value, dtype=np.uint8)
    Image.fromarray(arr, mode="L").save(path)


@pytest.fixture
def fake_dataset_dir(tmp_path: Path) -> dict:
    """Four matched image/mask pairs, varying sizes, mixed mask fill values."""
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    images_dir.mkdir()
    masks_dir.mkdir()

    stems = ["sample_01", "sample_02", "sample_03", "sample_04"]
    sizes = [(64, 64), (48, 80), (100, 60), (32, 32)]
    # After threshold=127: 255→1, 0→0, 255→1, 200→1
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
    """A minimal contract-conformant pipeline: Resize -> Normalize -> ToTensorV2."""
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


# ---------------------------------------------------------------------------
# 1) Pairing — filename-stem image ↔ mask matching
# ---------------------------------------------------------------------------

class TestPairing:
    def test_pairs_matched_stems_in_sorted_order(self, fake_dataset_dir):
        pairs = pair_by_stem(
            fake_dataset_dir["images_dir"],
            fake_dataset_dir["masks_dir"],
            image_ext=".jpg",
            mask_ext=".png",
        )
        assert len(pairs) == 4
        assert [stem for _, _, stem in pairs] == sorted(fake_dataset_dir["stems"])

    def test_extension_normalization_accepts_no_leading_dot(self, fake_dataset_dir):
        pairs = pair_by_stem(
            fake_dataset_dir["images_dir"],
            fake_dataset_dir["masks_dir"],
            image_ext="jpg",
            mask_ext="png",
        )
        assert len(pairs) == 4

    def test_orphan_strict_raises(self, fake_dataset_dir):
        (fake_dataset_dir["masks_dir"] / "sample_01.png").unlink()
        with pytest.raises(FileNotFoundError):
            pair_by_stem(
                fake_dataset_dir["images_dir"],
                fake_dataset_dir["masks_dir"],
                image_ext=".jpg",
                mask_ext=".png",
                strict_pairing=True,
            )

    def test_orphan_non_strict_warns_and_skips(self, fake_dataset_dir):
        (fake_dataset_dir["masks_dir"] / "sample_01.png").unlink()
        with pytest.warns(UserWarning, match="Skipping unpaired"):
            pairs = pair_by_stem(
                fake_dataset_dir["images_dir"],
                fake_dataset_dir["masks_dir"],
                image_ext=".jpg",
                mask_ext=".png",
                strict_pairing=False,
            )
        assert len(pairs) == 3
        assert all(stem != "sample_01" for _, _, stem in pairs)

    def test_missing_images_dir_raises(self, tmp_path):
        with pytest.raises(NotADirectoryError):
            pair_by_stem(tmp_path / "nope", tmp_path, ".jpg", ".png")


# ---------------------------------------------------------------------------
# 2) Transforms — build an Albumentations pipeline from list[dict]
# ---------------------------------------------------------------------------

class TestTransforms:
    def test_empty_specs_produce_identity_pipeline(self):
        t = build_transforms_from_config(None)
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        msk = np.zeros((10, 10), dtype=np.uint8)
        out = t(image=img, mask=msk)
        assert out["image"].shape == (10, 10, 3)
        assert out["mask"].shape == (10, 10)

    def test_resize_kwargs_passthrough(self):
        t = build_transforms_from_config(
            [{"name": "Resize", "height": 24, "width": 40}]
        )
        out = t(
            image=np.zeros((100, 100, 3), dtype=np.uint8),
            mask=np.zeros((100, 100), dtype=np.uint8),
        )
        assert out["image"].shape == (24, 40, 3)
        assert out["mask"].shape == (24, 40)

    def test_totensor_produces_chw_tensor(self):
        t = build_transforms_from_config([
            {"name": "Resize", "height": 8, "width": 8},
            {"name": "ToTensorV2"},
        ])
        out = t(
            image=np.zeros((16, 16, 3), dtype=np.uint8),
            mask=np.zeros((16, 16), dtype=np.uint8),
        )
        assert isinstance(out["image"], torch.Tensor)
        assert out["image"].shape == (3, 8, 8)

    def test_unknown_transform_name_raises(self):
        with pytest.raises(ValueError, match="Unknown Albumentations"):
            build_transforms_from_config([{"name": "DefinitelyNotAThing"}])

    def test_missing_name_key_raises(self):
        with pytest.raises(ValueError, match="missing 'name'"):
            build_transforms_from_config([{"height": 8, "width": 8}])


# ---------------------------------------------------------------------------
# 3) Output contract — dict keys, tensor shapes/dtypes, metadata correctness
# ---------------------------------------------------------------------------

class TestOutputContract:
    def test_length_matches_pair_count(self, base_config, standard_transforms):
        base_config["transforms"] = standard_transforms
        ds = GenericSegmentationDataset(base_config)
        assert len(ds) == 4

    def test_top_level_keys(self, base_config, standard_transforms):
        base_config["transforms"] = standard_transforms
        sample = GenericSegmentationDataset(base_config)[0]
        assert set(sample.keys()) == {"image", "mask", "metadata"}

    def test_image_shape_and_dtype(self, base_config, standard_transforms):
        base_config["transforms"] = standard_transforms
        sample = GenericSegmentationDataset(base_config)[0]
        assert sample["image"].shape == (3, 32, 32)
        assert sample["image"].dtype == torch.float32

    def test_mask_shape_dtype_and_binary_values(self, base_config, standard_transforms):
        base_config["transforms"] = standard_transforms
        ds = GenericSegmentationDataset(base_config)
        for i in range(len(ds)):
            m = ds[i]["mask"]
            assert m.shape == (1, 32, 32)
            assert m.dtype == torch.float32
            # every pixel is exactly 0 or 1
            assert torch.all((m == 0.0) | (m == 1.0))

    def test_mask_threshold_maps_expected_values(self, base_config):
        # No resize / normalize so threshold arithmetic is unambiguous.
        base_config["transforms"] = [{"name": "ToTensorV2"}]
        ds = GenericSegmentationDataset(base_config)
        # samples sorted by stem: 01(255)→1, 02(0)→0, 03(255)→1, 04(200)→1
        expected = [1.0, 0.0, 1.0, 1.0]
        for i, exp in enumerate(expected):
            assert ds[i]["mask"].mean().item() == pytest.approx(exp)

    def test_metadata_fields_populated_correctly(self, base_config, standard_transforms):
        base_config["transforms"] = standard_transforms
        sample = GenericSegmentationDataset(base_config)[2]
        meta = sample["metadata"]
        assert meta["dataset_name"] == "fake"
        assert meta["index"] == 2
        assert meta["sample_id"] == "sample_03"
        assert isinstance(meta["image_path"], str)
        assert isinstance(meta["mask_path"], str)
        # sample_03 was written at (H=100, W=60); original_size is pre-transform
        assert meta["original_size"] == (100, 60)


# ---------------------------------------------------------------------------
# 4) Config loader — YAML → GenericSegmentationDataset
# ---------------------------------------------------------------------------

class TestConfigLoader:
    def test_load_kvasir_seg_reads_yaml_and_instantiates(
        self, tmp_path, base_config, standard_transforms,
    ):
        base_config["transforms"] = standard_transforms
        yaml_path = tmp_path / "cfg.yaml"
        yaml_path.write_text(yaml.safe_dump(base_config))
        ds = load_kvasir_seg(yaml_path)
        assert len(ds) == 4
        assert ds[0]["image"].shape == (3, 32, 32)
        assert ds[0]["mask"].shape == (1, 32, 32)


# ---------------------------------------------------------------------------
# 5) DataLoader batching — default collate produces the batched contract
# ---------------------------------------------------------------------------

class TestDataLoaderBatching:
    def test_batched_shapes_via_default_collate(self, base_config, standard_transforms):
        base_config["transforms"] = standard_transforms
        ds = GenericSegmentationDataset(base_config)
        loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
        batch = next(iter(loader))
        assert batch["image"].shape == (4, 3, 32, 32)
        assert batch["mask"].shape == (4, 1, 32, 32)
        # metadata dict values are collated as lists / stacked tensors
        assert batch["metadata"]["dataset_name"] == ["fake"] * 4
        assert list(batch["metadata"]["index"]) == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# 6) Integration — real Kvasir-SEG on disk (auto-skips if not downloaded)
# ---------------------------------------------------------------------------

KVASIR_YAML = _ROOT / "configs" / "kvasir_seg.yaml"
KVASIR_IMAGES = _ROOT / "data" / "kvasir-seg" / "Kvasir-SEG" / "images"


def _resize_from_yaml(cfg_path: Path) -> tuple[int, int]:
    cfg = yaml.safe_load(cfg_path.read_text())
    for step in cfg.get("transforms", []):
        if step["name"] == "Resize":
            return step["height"], step["width"]
    raise AssertionError("no Resize step in kvasir_seg.yaml")


@pytest.mark.skipif(
    not KVASIR_IMAGES.is_dir(),
    reason="Kvasir-SEG not on disk — run scripts/download_kvasir_seg.py first",
)
class TestKvasirIntegration:
    def test_len_matches_official_1000(self):
        ds = load_kvasir_seg(KVASIR_YAML)
        assert len(ds) == 1000

    def test_single_sample_shapes_match_yaml_resize(self):
        h, w = _resize_from_yaml(KVASIR_YAML)
        ds = load_kvasir_seg(KVASIR_YAML)
        sample = ds[0]
        assert sample["image"].shape == (3, h, w)
        assert sample["mask"].shape == (1, h, w)
        assert sample["image"].dtype == torch.float32
        assert sample["mask"].dtype == torch.float32

    def test_dataloader_one_batch(self):
        h, w = _resize_from_yaml(KVASIR_YAML)
        ds = load_kvasir_seg(KVASIR_YAML)
        loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
        batch = next(iter(loader))
        assert batch["image"].shape == (4, 3, h, w)
        assert batch["mask"].shape == (4, 1, h, w)
