import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pytest
import torch

from models.factory import build_segmentation_model, spatial_size_at


@pytest.mark.parametrize("model_type", ["vanilla", "hybrid", "deep_hybrid"])
def test_models_return_one_logit_map_at_input_resolution(model_type):
    model = build_segmentation_model(model_type, bond_dim=16, image_size=64).eval()
    with torch.no_grad():
        logits = model(torch.randn(2, 3, 64, 64))
    assert logits.shape == (2, 1, 64, 64)


def test_vanilla_model_has_no_transforms():
    model = build_segmentation_model("vanilla", bond_dim=32)
    assert model.__class__.__name__ == "UNet"
    assert not hasattr(model, "transform_1024")


def test_hybrid_model_attaches_only_the_bottleneck_transform():
    model = build_segmentation_model("hybrid", bond_dim=32)
    assert model.transform_512 is None
    assert model.transform_1024 is not None


def test_deep_hybrid_model_attaches_both_transforms():
    model = build_segmentation_model("deep_hybrid", bond_dim=16)
    assert model.transform_512 is not None
    assert model.transform_1024 is not None


def test_mps_cores_follow_the_image_size():
    model = build_segmentation_model("hybrid", bond_dim=8, image_size=256)
    assert model.transform_1024.axial_mps.row_operator.materialize().shape == (8, 16, 16, 8)


def test_hybrid_forward_preserves_spatial_shape():
    model = build_segmentation_model("hybrid", bond_dim=4, image_size=64).eval()
    with torch.no_grad():
        assert model(torch.randn(1, 3, 64, 64)).shape == (1, 1, 64, 64)


def test_unknown_model_type_raises():
    with pytest.raises(ValueError, match="Unknown model_type"):
        build_segmentation_model("transformer", bond_dim=32)


def test_image_size_must_be_divisible_by_sixteen():
    with pytest.raises(ValueError, match="divisible by 16"):
        spatial_size_at(500, 16)


def test_tensorized_bottleneck_is_below_ten_percent_of_the_standard_bottleneck():
    def count(module):
        return sum(parameter.numel() for parameter in module.parameters())

    vanilla = build_segmentation_model("vanilla", bond_dim=32)
    hybrid = build_segmentation_model("hybrid", bond_dim=32)
    assert count(hybrid.down4) + count(hybrid.transform_1024) < 0.1 * count(vanilla.down4)
