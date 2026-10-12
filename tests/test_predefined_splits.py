import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pytest
import yaml
from PIL import Image
from torch.utils.data import ConcatDataset

from dataset import (
    EVAL_STAGE,
    TRAIN_STAGE,
    build_splits,
    carve_validation,
    declares_predefined_splits,
    discover_dataset_configs,
    make_splits,
    uses_foreground_map_metrics,
)

EVAL_PIPELINE = [
    {"name": "Resize", "height": 16, "width": 16},
    {"name": "Normalize", "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5], "max_pixel_value": 255.0},
    {"name": "ToTensorV2"},
]
SOURCE_STEMS = {
    "train_a": [f"a_{index:02d}" for index in range(10)],
    "train_b": [f"b_{index:02d}" for index in range(6)],
    "test_x": ["x_00", "x_01", "x_02", "a_03"],
    "test_y": ["y_00", "y_01", "y_02"],
}


def write_source(root: Path, stems: list[str]) -> dict[str, str]:
    images, masks = root / "images", root / "masks"
    images.mkdir(parents=True)
    masks.mkdir(parents=True)
    for stem in stems:
        Image.fromarray(np.full((20, 24, 3), 90, np.uint8)).save(images / f"{stem}.jpg")
        Image.fromarray(np.full((20, 24), 255, np.uint8)).save(masks / f"{stem}.png")
    return {"images_dir": str(images), "masks_dir": str(masks)}


def predefined_config(tmp_path: Path, **overrides) -> dict:
    config = {
        "dataset_name": "synthetic",
        "image_ext": ".jpg",
        "mask_ext": ".png",
        "strict_pairing": True,
        "mask_threshold": 127,
        "sources": {name: write_source(tmp_path / "data" / name, stems) for name, stems in SOURCE_STEMS.items()},
        "train_sources": ["train_a", "train_b"],
        "test_sources": ["test_x", "test_y"],
        "validation_fraction": 0.2,
        "train_transforms": [{"name": "HorizontalFlip", "p": 0.5}, *EVAL_PIPELINE],
        "eval_transforms": EVAL_PIPELINE,
    }
    config.update(overrides)
    return config


def write_config(directory: Path, name: str, config: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.yaml"
    path.write_text(yaml.safe_dump(config))
    return directory


def sample_ids(dataset: ConcatDataset) -> list[str]:
    return [part.dataset.samples[index][2] for part in dataset.datasets for index in part.indices]


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    return write_config(tmp_path / "configs", "synthetic", predefined_config(tmp_path))


def build(config_dir: Path, seed: int = 42):
    with pytest.warns(UserWarning, match="Excluding 1 training sample"):
        return make_splits("synthetic", seed=seed, config_dir=config_dir)


def test_validation_is_carved_per_training_source(config_dir):
    splits = build(config_dir)
    assert (len(splits.train), len(splits.val)) == (12, 3)
    validation = sample_ids(splits.val)
    assert sum(sample.startswith("a_") for sample in validation) == 2
    assert sum(sample.startswith("b_") for sample in validation) == 1


def test_test_sources_stay_whole_and_named(config_dir):
    splits = build(config_dir)
    assert {name: len(dataset) for name, dataset in splits.test_sets.items()} == {"test_x": 4, "test_y": 3}
    assert all(dataset.stage == EVAL_STAGE for dataset in splits.test_sets.values())


def test_training_samples_shared_with_a_test_source_are_excluded(config_dir):
    splits = build(config_dir)
    held_out = {sample for dataset in splits.test_sets.values() for _, _, sample in dataset.samples}
    assert not set(sample_ids(splits.train)) & held_out
    assert not set(sample_ids(splits.val)) & held_out
    assert not set(sample_ids(splits.train)) & set(sample_ids(splits.val))


def test_partitions_use_the_stage_pipelines(config_dir):
    splits = build(config_dir)
    assert all(part.dataset.stage == TRAIN_STAGE for part in splits.train.datasets)
    assert all(part.dataset.stage == EVAL_STAGE for part in splits.val.datasets)


def test_carve_is_deterministic_and_seed_dependent(config_dir):
    first, again, other = build(config_dir), build(config_dir), build(config_dir, seed=7)
    assert sample_ids(first.val) == sample_ids(again.val)
    assert sample_ids(first.val) != sample_ids(other.val)


def test_single_test_set_accessor_rejects_several_sets(config_dir):
    with pytest.raises(ValueError, match="single test set"):
        build(config_dir).test


def test_carve_validation_covers_every_index_once():
    indices = [3, 5, 8, 13, 21, 34, 55, 89, 144, 233]
    train, validation = carve_validation(indices, 0.3, seed=0)
    assert len(validation) == 3
    assert sorted(train + validation) == indices


def test_joint_training_merges_test_sets_by_name(tmp_path):
    configs = tmp_path / "configs"
    write_config(configs, "first", predefined_config(tmp_path / "first"))
    write_config(configs, "second", predefined_config(tmp_path / "second"))
    with pytest.warns(UserWarning):
        splits = build_splits(["first", "second"], config_dir=configs)
    assert {name: len(dataset) for name, dataset in splits.test_sets.items()} == {"test_x": 8, "test_y": 6}


def test_predefined_configs_are_discovered(config_dir):
    assert "synthetic" in discover_dataset_configs(config_dir)


def test_foreground_map_metrics_follow_the_config_flag(tmp_path):
    configs = tmp_path / "configs"
    write_config(configs, "flagged", predefined_config(tmp_path / "flagged", foreground_map_metrics=True))
    write_config(configs, "plain", predefined_config(tmp_path / "plain"))
    assert uses_foreground_map_metrics(["flagged"], configs)
    assert not uses_foreground_map_metrics(["plain"], configs)
    assert uses_foreground_map_metrics(["plain", "flagged"], configs)


def test_partial_predefined_declaration_is_rejected():
    with pytest.raises(ValueError, match="predefined splits also need"):
        declares_predefined_splits({"dataset_name": "partial", "sources": {}, "train_sources": ["a"]})


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"train_sources": ["train_a", "missing"]}, "not defined"),
        ({"test_sources": ["test_x", "train_a"]}, "only once"),
        ({"test_sources": []}, "at least one source"),
        ({"validation_fraction": 1.0}, "must lie in"),
        ({"group_map": {"csv": "groups.csv"}}, "cannot be combined"),
        ({"images_dir": "data/images"}, "cannot be combined"),
    ],
)
def test_invalid_predefined_configs_are_rejected(tmp_path, overrides, message):
    config_dir = write_config(tmp_path / "configs", "synthetic", predefined_config(tmp_path, **overrides))
    with pytest.raises(ValueError, match=message):
        make_splits("synthetic", config_dir=config_dir)


def test_source_without_directories_is_rejected(tmp_path):
    config = predefined_config(tmp_path)
    config["sources"]["test_y"] = {"images_dir": config["sources"]["test_y"]["images_dir"]}
    config_dir = write_config(tmp_path / "configs", "synthetic", config)
    with pytest.raises(ValueError, match="need both"):
        make_splits("synthetic", config_dir=config_dir)


COD_EXPECTED_TEST_SIZES = {"cod10k": 2026, "camo": 250, "chameleon": 76, "nc4k": 4121}


def test_cod_benchmark_protocol():
    if not (_ROOT / "data/cod/test/nc4k/images").is_dir():
        pytest.skip("cod not on disk")
    with pytest.warns(UserWarning, match="Excluding 2 training sample"):
        splits = make_splits("cod", config_dir=_ROOT / "configs")
    assert (len(splits.train), len(splits.val)) == (3634, 404)
    assert {name: len(dataset) for name, dataset in splits.test_sets.items()} == COD_EXPECTED_TEST_SIZES
    held_out = {sample for dataset in splits.test_sets.values() for _, _, sample in dataset.samples}
    assert not (set(sample_ids(splits.train)) | set(sample_ids(splits.val))) & held_out


def test_cod_config_declares_the_benchmark_protocol():
    config = yaml.safe_load((_ROOT / "configs" / "cod.yaml").read_text())
    assert config["train_sources"] == ["cod10k_train", "camo_train"]
    assert config["test_sources"] == list(COD_EXPECTED_TEST_SIZES)
    assert config["foreground_map_metrics"] is True


def test_every_cod_test_pair_loads():
    if not (_ROOT / "data/cod/test/chameleon/images").is_dir():
        pytest.skip("cod not on disk")
    with pytest.warns(UserWarning):
        chameleon = make_splits("cod", config_dir=_ROOT / "configs").test_sets["chameleon"]
    shapes = {tuple(chameleon[index]["mask"].shape) for index in range(len(chameleon))}
    assert shapes == {(1, 512, 512)}
