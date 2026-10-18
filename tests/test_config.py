from pathlib import Path

import numpy as np
import pytest
import torch

from dataset import GroupedSplit, GroupMap, RandomSplit, discover_dataset_configs, load_config, parse_config
from dataset.config import REPO_ROOT

TO_TENSOR = {"name": "ToTensorV2"}


def resize(height: int, width: int) -> dict:
    return {"name": "Resize", "height": height, "width": width}


def test_single_source_config_uses_a_random_split(base_config, standard_transforms):
    base_config["transforms"] = standard_transforms
    config = parse_config(base_config)
    assert isinstance(config.split, RandomSplit)
    assert config.split.source.name == "fake"
    assert config.image_size == 32
    assert config.foreground_map_metrics is False


def test_group_map_selects_a_grouped_split(base_config, standard_transforms):
    base_config["transforms"] = standard_transforms
    base_config["group_map"] = {"csv": "groups.csv"}
    assert isinstance(parse_config(base_config).split, GroupedSplit)


def test_cross_validation_folds_are_read_from_a_grouped_config(base_config, standard_transforms):
    base_config["transforms"] = standard_transforms
    base_config["group_map"] = {"csv": "groups.csv"}
    assert parse_config(base_config).split.folds is None
    base_config["cross_validation_folds"] = 5
    assert parse_config(base_config).split.folds == 5


def test_source_defaults_follow_the_schema(base_config, standard_transforms):
    for key in ("strict_pairing", "mask_threshold"):
        del base_config[key]
    base_config["transforms"] = standard_transforms
    source = parse_config(base_config).split.source
    assert (source.strict_pairing, source.mask_threshold, source.align_mask_to_image) == (True, 127, False)


def test_image_size_follows_the_last_sizing_transform(base_config):
    base_config["train_transforms"] = [resize(64, 64), {"name": "RandomCrop", "height": 48, "width": 48}, TO_TENSOR]
    base_config["eval_transforms"] = [resize(64, 64), {"name": "CenterCrop", "height": 48, "width": 48}, TO_TENSOR]
    assert parse_config(base_config).image_size == 48


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"trasforms": []}, "unknown config keys"),
        ({"transforms": [resize(32, 32)]}, "must end with ToTensorV2"),
        ({"transforms": [{"name": "Normalize"}, TO_TENSOR]}, "fix its output size"),
        ({"train_transforms": [resize(32, 32), TO_TENSOR], "eval_transforms": [resize(16, 16), TO_TENSOR]}, "different sizes"),
        ({"transforms": [resize(32, 16), TO_TENSOR]}, "square"),
        ({"group_map": "groups.csv"}, "must be a mapping"),
        ({"group_map": {"csv": "groups.csv", "sample": "id"}}, "unknown keys"),
        ({"cross_validation_folds": 5}, "needs a group_map"),
        ({"group_map": {"csv": "groups.csv"}, "cross_validation_folds": 1}, "at least 2"),
        ({"group_map": {"csv": "groups.csv"}, "cross_validation_folds": "5"}, "at least 2"),
        ({"transforms": [resize(32, 32), {"name": "Normalize", "normalization": "min_max"}, TO_TENSOR]}, "can be inverted"),
    ],
)
def test_invalid_configs_are_rejected(base_config, standard_transforms, overrides, message):
    base_config["transforms"] = standard_transforms
    base_config.update(overrides)
    with pytest.raises(ValueError, match=message):
        parse_config(base_config)


def test_missing_source_keys_are_rejected(base_config, standard_transforms):
    del base_config["masks_dir"]
    base_config["transforms"] = standard_transforms
    with pytest.raises(ValueError, match="missing \\['masks_dir'\\]"):
        parse_config(base_config)


def test_unknown_dataset_names_list_the_available_configs(tmp_path):
    (tmp_path / "known.yaml").write_text("dataset_name: known\n")
    assert list(discover_dataset_configs(tmp_path)) == ["known"]
    with pytest.raises(ValueError, match="Available: known"):
        load_config("unknown", tmp_path)


def test_relative_paths_resolve_against_the_repository_root(base_config, standard_transforms):
    base_config.update(images_dir="data/images", masks_dir="data/masks", transforms=standard_transforms)
    base_config["group_map"] = {"csv": "configs/groups.csv", "group_column": "sequence_id"}
    split = parse_config(base_config).split
    assert split.source.images_dir == REPO_ROOT / "data/images"
    assert split.group_map == GroupMap(REPO_ROOT / "configs/groups.csv", group_column="sequence_id")


def test_absolute_paths_are_kept(base_config, standard_transforms):
    base_config["transforms"] = standard_transforms
    assert parse_config(base_config).split.source.images_dir == Path(base_config["images_dir"])


@pytest.mark.parametrize(
    "normalize, offset, scale",
    [
        ({"name": "Normalize"}, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ({"name": "Normalize", "mean": [0.5, 0.5, 0.5], "std": [0.25, 0.25, 0.25]}, (0.5,) * 3, (0.25,) * 3),
        ({"name": "Normalize", "mean": 0.5, "std": 0.5, "max_pixel_value": 127.5}, (0.25,) * 3, (0.25,) * 3),
        (None, (0.0,) * 3, (1.0 / 255.0,) * 3),
    ],
)
def test_denormalization_inverts_the_eval_normalize_step(base_config, normalize, offset, scale):
    base_config["transforms"] = [resize(8, 8), *([normalize] if normalize else []), TO_TENSOR]
    denormalization = parse_config(base_config).pipelines.denormalization
    assert denormalization.offset == pytest.approx(offset)
    assert denormalization.scale == pytest.approx(scale)


def test_denormalized_images_are_hwc_in_the_unit_range(base_config, standard_transforms):
    base_config["transforms"] = standard_transforms
    image = parse_config(base_config).pipelines.denormalization(torch.randn(3, 8, 6) * 5)
    assert image.shape == (8, 6, 3)
    assert np.all((image >= 0.0) & (image <= 1.0))
