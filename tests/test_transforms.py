from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from dataset import (
    assert_deterministic_pipeline,
    build_transforms_from_config,
    EVAL_STAGE,
    parse_config,
    TRAIN_STAGE,
)


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


STOCHASTIC_SPEC = {"name": "HorizontalFlip", "p": 0.5}


class TestStageTransforms:
    def test_stage_specific_key_takes_precedence(self, base_config, standard_transforms):
        base_config["transforms"] = [STOCHASTIC_SPEC, *standard_transforms]
        base_config["eval_transforms"] = standard_transforms
        assert parse_config(base_config).pipelines.for_stage(EVAL_STAGE) == tuple(standard_transforms)

    def test_shared_key_is_the_fallback(self, base_config, standard_transforms):
        base_config["transforms"] = standard_transforms
        assert parse_config(base_config).pipelines.for_stage(TRAIN_STAGE) == tuple(standard_transforms)

    def test_unknown_stage_raises(self, base_config, standard_transforms):
        base_config["transforms"] = standard_transforms
        with pytest.raises(ValueError, match="Unknown stage"):
            parse_config(base_config).pipelines.for_stage("inference")

    def test_deterministic_pipeline_is_accepted(self, standard_transforms):
        assert_deterministic_pipeline(standard_transforms)

    def test_stochastic_pipeline_is_rejected(self, standard_transforms):
        with pytest.raises(ValueError, match="must be deterministic"):
            assert_deterministic_pipeline([STOCHASTIC_SPEC, *standard_transforms])

    def test_config_refuses_a_stochastic_shared_eval_pipeline(self, base_config, standard_transforms):
        base_config["transforms"] = [STOCHASTIC_SPEC, *standard_transforms]
        with pytest.raises(ValueError, match="must be deterministic"):
            parse_config(base_config)

    def test_with_stage_switches_pipeline_without_reloading(self, base_config, standard_transforms, make_dataset):
        base_config["train_transforms"] = [STOCHASTIC_SPEC, *standard_transforms]
        base_config["eval_transforms"] = standard_transforms
        train_dataset = make_dataset(base_config, TRAIN_STAGE)
        eval_dataset = train_dataset.with_stage(EVAL_STAGE)
        assert eval_dataset.samples is train_dataset.samples
        assert train_dataset.stage == TRAIN_STAGE
        assert eval_dataset.stage == EVAL_STAGE

    def test_eval_samples_are_identical_across_reads(self, base_config, standard_transforms, make_dataset):
        base_config["train_transforms"] = [STOCHASTIC_SPEC, *standard_transforms]
        base_config["eval_transforms"] = standard_transforms
        eval_dataset = make_dataset(base_config, EVAL_STAGE)
        assert torch.equal(eval_dataset[0]["image"], eval_dataset[0]["image"])

    def test_reseeded_augmentations_are_reproducible(self, base_config, standard_transforms, make_dataset):
        gradient = np.tile(np.arange(64, dtype=np.uint8), (64, 1))
        image = np.stack([gradient] * 3, axis=-1)
        Image.fromarray(image, mode="RGB").save(Path(base_config["images_dir"]) / "sample_01.jpg")
        base_config["train_transforms"] = [{"name": "HorizontalFlip", "p": 0.5}, *standard_transforms]
        base_config["eval_transforms"] = standard_transforms

        def draws(seed: int) -> list[torch.Tensor]:
            dataset = make_dataset(base_config, TRAIN_STAGE)
            dataset.reseed_augmentations(seed)
            return [dataset[0]["image"] for _ in range(8)]

        first, second = draws(7), draws(7)
        assert all(torch.equal(a, b) for a, b in zip(first, second))
