from pathlib import Path

import pytest
import torch
import yaml
from torch.utils.data import DataLoader

from dataset import GenericSegmentationDataset, load_config


class TestOutputContract:
    def test_length_matches_pair_count(self, base_config, standard_transforms, make_dataset):
        base_config["transforms"] = standard_transforms
        ds = make_dataset(base_config)
        assert len(ds) == 4

    def test_top_level_keys(self, base_config, standard_transforms, make_dataset):
        base_config["transforms"] = standard_transforms
        sample = make_dataset(base_config)[0]
        assert set(sample.keys()) == {"image", "mask", "metadata"}

    def test_image_shape_and_dtype(self, base_config, standard_transforms, make_dataset):
        base_config["transforms"] = standard_transforms
        sample = make_dataset(base_config)[0]
        assert sample["image"].shape == (3, 32, 32)
        assert sample["image"].dtype == torch.float32

    def test_mask_shape_dtype_and_binary_values(self, base_config, standard_transforms, make_dataset):
        base_config["transforms"] = standard_transforms
        ds = make_dataset(base_config)
        for i in range(len(ds)):
            m = ds[i]["mask"]
            assert m.shape == (1, 32, 32)
            assert m.dtype == torch.float32
            assert torch.all((m == 0.0) | (m == 1.0))

    def test_mask_threshold_maps_expected_values(self, base_config, make_dataset):
        base_config["transforms"] = [{"name": "Resize", "height": 16, "width": 16}, {"name": "ToTensorV2"}]
        ds = make_dataset(base_config)
        expected = [1.0, 0.0, 1.0, 1.0]
        for i, exp in enumerate(expected):
            assert ds[i]["mask"].mean().item() == pytest.approx(exp)

    def test_metadata_fields_populated_correctly(self, base_config, standard_transforms, make_dataset):
        base_config["transforms"] = standard_transforms
        sample = make_dataset(base_config)[2]
        meta = sample["metadata"]
        assert meta["dataset_name"] == "fake"
        assert meta["index"] == 2
        assert meta["sample_id"] == "sample_03"
        assert isinstance(meta["image_path"], str)
        assert isinstance(meta["mask_path"], str)
        assert meta["original_size"] == (100, 60)


class TestMaskImageAlignment:
    @pytest.fixture
    def mismatched_config(self, base_config, standard_transforms, mask_writer):
        mask_writer(Path(base_config["masks_dir"]) / "sample_01.png", 60, 64, 255)
        base_config["transforms"] = standard_transforms
        return base_config

    def test_size_mismatch_raises_by_default(self, mismatched_config, make_dataset):
        with pytest.raises(ValueError, match="sample_01: mask \\(60, 64\\) and image \\(64, 64\\) differ"):
            make_dataset(mismatched_config)[0]

    def test_opt_in_aligns_the_mask_to_the_image_frame(self, mismatched_config, make_dataset):
        mismatched_config["align_mask_to_image"] = True
        sample = make_dataset(mismatched_config)[0]
        assert sample["mask"].shape == (1, 32, 32)
        assert torch.all(sample["mask"] == 1.0)


class TestConfigLoader:
    def test_yaml_config_builds_its_source_dataset(self, tmp_path, base_config, standard_transforms):
        base_config["transforms"] = standard_transforms
        (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(base_config))
        config = load_config("cfg", tmp_path)
        ds = GenericSegmentationDataset(config.split.source, config.pipelines)
        assert len(ds) == 4
        assert ds[0]["image"].shape == (3, 32, 32)
        assert ds[0]["mask"].shape == (1, 32, 32)


class TestDataLoaderBatching:
    def test_batched_shapes_via_default_collate(self, base_config, standard_transforms, make_dataset):
        base_config["transforms"] = standard_transforms
        ds = make_dataset(base_config)
        loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
        batch = next(iter(loader))
        assert batch["image"].shape == (4, 3, 32, 32)
        assert batch["mask"].shape == (4, 1, 32, 32)
        assert batch["metadata"]["dataset_name"] == ["fake"] * 4
        assert list(batch["metadata"]["index"]) == [0, 1, 2, 3]
