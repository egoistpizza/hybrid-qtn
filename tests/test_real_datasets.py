import pytest
import torch
from torch.utils.data import DataLoader

from dataset import EVAL_STAGE, TRAIN_STAGE, discover_dataset_configs, load_config, make_splits
from tests.real_data import CONFIG_DIR, REAL_DATASETS, REPO_ROOT, require_on_disk, whole_dataset


@pytest.mark.parametrize("name", sorted(REAL_DATASETS))
class TestRealDatasetIntegration:
    def test_config_is_discoverable(self, name):
        assert name in discover_dataset_configs(CONFIG_DIR)

    def test_len_matches_official_count(self, name):
        assert len(whole_dataset(require_on_disk(name))) == REAL_DATASETS[name][1]

    def test_config_points_at_its_own_data(self, name):
        source = load_config(name, CONFIG_DIR).split.source
        expected_root = REPO_ROOT / "/".join(REAL_DATASETS[name][0].split("/")[:2])
        assert source.images_dir.is_relative_to(expected_root)
        assert source.masks_dir.is_relative_to(expected_root)

    def test_single_sample_shapes_match_the_config_image_size(self, name):
        config = require_on_disk(name)
        sample = whole_dataset(config)[0]
        assert sample["image"].shape == (3, config.image_size, config.image_size)
        assert sample["mask"].shape == (1, config.image_size, config.image_size)
        assert sample["image"].dtype == torch.float32
        assert sample["mask"].dtype == torch.float32

    def test_masks_are_binary(self, name):
        mask = whole_dataset(require_on_disk(name))[0]["mask"]
        assert torch.all((mask == 0.0) | (mask == 1.0))

    def test_dataloader_one_batch(self, name):
        config = require_on_disk(name)
        loader = DataLoader(whole_dataset(config), batch_size=4, shuffle=False, num_workers=0)
        batch = next(iter(loader))
        assert batch["image"].shape == (4, 3, config.image_size, config.image_size)
        assert batch["mask"].shape == (4, 1, config.image_size, config.image_size)


class TestDatasetConsistency:
    def test_medical_configs_share_transform_pipelines(self):
        medical = [load_config(name, CONFIG_DIR).pipelines for name in ("kvasir_seg", "cvc_clinicdb")]
        assert medical[0] == medical[1]

    def test_every_config_in_the_repository_parses(self):
        assert {name: load_config(name, CONFIG_DIR).image_size for name in discover_dataset_configs(CONFIG_DIR)} == {
            "cod": 512,
            "cvc_clinicdb": 512,
            "kvasir_seg": 512,
            "mass_roads": 512,
        }

    def test_splits_are_deterministic_and_disjoint(self):
        for name, (images_dir, _) in REAL_DATASETS.items():
            if not (REPO_ROOT / images_dir).is_dir():
                continue
            first, again = make_splits(load_config(name, CONFIG_DIR)), make_splits(load_config(name, CONFIG_DIR))
            partitions = (first.train.indices, first.val.dataset.indices, first.test_sets["test"].dataset.indices)
            assert partitions == (again.train.indices, again.val.dataset.indices, again.test_sets["test"].dataset.indices)
            assert not set(partitions[0]) & set(partitions[1])
            assert not set(partitions[0]) & set(partitions[2])
            assert not set(partitions[1]) & set(partitions[2])

    def test_holdout_splits_use_the_eval_pipeline(self):
        splits = make_splits(require_on_disk("kvasir_seg"))
        assert splits.train.dataset.stage == TRAIN_STAGE
        assert splits.val.dataset.dataset.stage == EVAL_STAGE
        assert splits.test_sets["test"].dataset.dataset.stage == EVAL_STAGE
