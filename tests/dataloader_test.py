from __future__ import annotations

import csv
import sys
from pathlib import Path

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
    discover_dataset_configs,
    load_dataset,
    make_splits,
    pair_by_stem,
    resolve_config,
    grouped_split_indices,
    load_group_map,
    groups_for_samples,
)

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
            assert torch.all((m == 0.0) | (m == 1.0))

    def test_mask_threshold_maps_expected_values(self, base_config):
        base_config["transforms"] = [{"name": "ToTensorV2"}]
        ds = GenericSegmentationDataset(base_config)
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
        assert meta["original_size"] == (100, 60)

class TestConfigLoader:
    def test_load_dataset_reads_yaml_and_instantiates(
        self, tmp_path, base_config, standard_transforms,
    ):
        base_config["transforms"] = standard_transforms
        yaml_path = tmp_path / "cfg.yaml"
        yaml_path.write_text(yaml.safe_dump(base_config))
        ds = load_dataset(yaml_path)
        assert len(ds) == 4
        assert ds[0]["image"].shape == (3, 32, 32)
        assert ds[0]["mask"].shape == (1, 32, 32)

class TestDataLoaderBatching:
    def test_batched_shapes_via_default_collate(self, base_config, standard_transforms):
        base_config["transforms"] = standard_transforms
        ds = GenericSegmentationDataset(base_config)
        loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
        batch = next(iter(loader))
        assert batch["image"].shape == (4, 3, 32, 32)
        assert batch["mask"].shape == (4, 1, 32, 32)
        assert batch["metadata"]["dataset_name"] == ["fake"] * 4
        assert list(batch["metadata"]["index"]) == [0, 1, 2, 3]

REAL_DATASETS = {
    "kvasir_seg": ("data/kvasir-seg/Kvasir-SEG/images", 1000),
    "cvc_clinicdb": ("data/cvc-clinicdb/CVC-ClinicDB/Original", 612),
    "mass_roads": ("data/mass_roads/images", 1171),
}

def _resize_from_yaml(cfg_path: Path) -> tuple[int, int]:
    cfg = yaml.safe_load(cfg_path.read_text())
    pipeline = cfg.get("train_transforms", cfg.get("transforms", []))
    for step in pipeline:
        if step["name"] in ("Resize", "RandomCrop", "CenterCrop"):
            return step["height"], step["width"]
    raise AssertionError(f"no Resize, RandomCrop, or CenterCrop step in {cfg_path.name}")

def _require_on_disk(name: str) -> Path:
    images_dir, _ = REAL_DATASETS[name]
    if not (_ROOT / images_dir).is_dir():
        pytest.skip(f"{name} not on disk")
    return resolve_config(name, _ROOT / "configs")

@pytest.mark.parametrize("name", sorted(REAL_DATASETS))
class TestRealDatasetIntegration:
    def test_config_is_discoverable(self, name):
        assert name in discover_dataset_configs(_ROOT / "configs")

    def test_len_matches_official_count(self, name):
        cfg_path = _require_on_disk(name)
        assert len(load_dataset(cfg_path)) == REAL_DATASETS[name][1]

    def test_config_points_at_its_own_data(self, name):
        cfg_path = resolve_config(name, _ROOT / "configs")
        cfg = yaml.safe_load(cfg_path.read_text())
        expected_root = REAL_DATASETS[name][0].split("/")[1]
        assert expected_root in cfg["images_dir"]
        assert expected_root in cfg["masks_dir"]

    def test_single_sample_shapes_match_yaml_resize(self, name):
        cfg_path = _require_on_disk(name)
        h, w = _resize_from_yaml(cfg_path)
        sample = load_dataset(cfg_path)[0]
        assert sample["image"].shape == (3, h, w)
        assert sample["mask"].shape == (1, h, w)
        assert sample["image"].dtype == torch.float32
        assert sample["mask"].dtype == torch.float32

    def test_masks_are_binary(self, name):
        cfg_path = _require_on_disk(name)
        mask = load_dataset(cfg_path)[0]["mask"]
        assert torch.all((mask == 0.0) | (mask == 1.0))

    def test_dataloader_one_batch(self, name):
        cfg_path = _require_on_disk(name)
        h, w = _resize_from_yaml(cfg_path)
        loader = DataLoader(load_dataset(cfg_path), batch_size=4, shuffle=False, num_workers=0)
        batch = next(iter(loader))
        assert batch["image"].shape == (4, 3, h, w)
        assert batch["mask"].shape == (4, 1, h, w)

class TestDatasetConsistency:
    def test_medical_configs_share_a_transform_pipeline(self):
        configs = discover_dataset_configs(_ROOT / "configs")
        medical_configs = {k: v for k, v in configs.items() if k in ("kvasir_seg", "cvc_clinicdb")}
        pipelines = {
            name: yaml.safe_load(path.read_text()).get("transforms")
            for name, path in medical_configs.items()
        }
        if len(medical_configs) >= 2:
            reference_name, reference = next(iter(pipelines.items()))
            for name, pipeline in pipelines.items():
                assert pipeline == reference

    def test_splits_are_deterministic_and_disjoint(self):
        for name in discover_dataset_configs(_ROOT / "configs"):
            if not (_ROOT / REAL_DATASETS[name][0]).is_dir():
                continue
            first_train, first_val = make_splits(name, config_dir=_ROOT / "configs")
            again_train, again_val = make_splits(name, config_dir=_ROOT / "configs")
            assert first_train.indices == again_train.indices
            assert first_val.indices == again_val.indices
            assert not set(first_train.indices) & set(first_val.indices)

CVC_GROUP_SPEC = {
    "csv": "configs/cvc_clinicdb_sequences.csv",
    "sample_column": "sample_id",
    "group_column": "sequence_id",
}

class TestSequenceGrouping:
    def test_group_map_covers_every_cvc_frame(self):
        _require_on_disk("cvc_clinicdb")
        mapping = load_group_map(CVC_GROUP_SPEC, root=_ROOT)
        assert len(mapping) == 612
        assert len(set(mapping.values())) == 29

    def test_cvc_split_shares_no_sequence(self):
        _require_on_disk("cvc_clinicdb")
        mapping = load_group_map(CVC_GROUP_SPEC, root=_ROOT)
        train, val = make_splits("cvc_clinicdb", config_dir=_ROOT / "configs")
        def seqs(subset):
            return {mapping[subset.dataset.samples[i][2]] for i in subset.indices}
        assert not seqs(train) & seqs(val)
        assert len(seqs(val)) >= 4

    def test_cvc_split_covers_dataset_exactly_once(self):
        _require_on_disk("cvc_clinicdb")
        train, val = make_splits("cvc_clinicdb", config_dir=_ROOT / "configs")
        assert sorted([*train.indices, *val.indices]) == list(range(612))

    def test_cvc_val_fraction_is_near_target(self):
        _require_on_disk("cvc_clinicdb")
        _, val = make_splits("cvc_clinicdb", config_dir=_ROOT / "configs")
        assert 0.15 <= len(val) / 612 <= 0.25

    def test_cvc_split_is_deterministic(self):
        _require_on_disk("cvc_clinicdb")
        a_train, a_val = make_splits("cvc_clinicdb", config_dir=_ROOT / "configs")
        b_train, b_val = make_splits("cvc_clinicdb", config_dir=_ROOT / "configs")
        assert a_train.indices == b_train.indices
        assert a_val.indices == b_val.indices

    def test_committed_map_matches_mirror_metadata(self):
        metadata = _ROOT / "data/cvc-clinicdb/metadata.csv"
        if not metadata.is_file():
            pytest.skip("mirror metadata.csv not present")
        expected = {
            str(int(r["frame_id"])): str(int(r["sequence_id"]))
            for r in csv.DictReader(metadata.open(newline=""))
        }
        assert load_group_map(CVC_GROUP_SPEC, root=_ROOT) == expected

    def test_ungrouped_sample_raises_rather_than_leaking(self, tmp_path):
        path = tmp_path / "partial.csv"
        path.write_text("sample_id,sequence_id\n1,1\n")
        with pytest.raises(KeyError, match="no group"):
            groups_for_samples(["1", "2"], {"csv": str(path),
                                            "sample_column": "sample_id",
                                            "group_column": "sequence_id"})

    def test_missing_group_csv_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="group_map csv not found"):
            load_group_map({"csv": "nope.csv"}, root=tmp_path)

    def test_grouped_split_is_group_pure_on_synthetic_data(self):
        groups = [f"s{i // 10}" for i in range(100)]
        train, val = grouped_split_indices(groups, 0.2, seed=42)
        assert not {groups[i] for i in train} & {groups[i] for i in val}
        assert sorted([*train, *val]) == list(range(100))

class TestKvasirSplitUnchanged:
    def test_kvasir_matches_legacy_random_split_exactly(self):
        cfg = _require_on_disk("kvasir_seg")
        n = len(load_dataset(cfg))
        train_size = int(0.8 * n)
        legacy_train, legacy_val = torch.utils.data.random_split(
            range(n), [train_size, n - train_size],
            generator=torch.Generator().manual_seed(42),
        )
        train, val = make_splits("kvasir_seg", config_dir=_ROOT / "configs")
        assert list(train.indices) == list(legacy_train.indices)
        assert list(val.indices) == list(legacy_val.indices)

    def test_kvasir_config_declares_no_group_map(self):
        cfg = yaml.safe_load(
            resolve_config("kvasir_seg", _ROOT / "configs").read_text()
        )
        assert "group_map" not in cfg
