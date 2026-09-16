import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pytest
import torch

from models import bottleneck_sizes, build_model, build_run_name


def _mps_grid(bottleneck) -> tuple[int, int]:
    return tuple(bottleneck.axial_mps.positional_encoding.spatial_embeddings.shape[-2:])


@pytest.mark.parametrize("image_size, size_512, size_1024", [(512, 64, 32), (256, 32, 16)])
def test_bottleneck_sizes_follow_image_size(image_size, size_512, size_1024):
    assert bottleneck_sizes(image_size) == (size_512, size_1024)

    model = build_model("deep_hybrid", bond_dim=4, image_size=image_size)
    assert _mps_grid(model.transform_512) == (size_512, size_512)
    assert _mps_grid(model.transform_1024) == (size_1024, size_1024)
    assert model.transform_1024.axial_mps.T_mid_H.shape == (4, size_1024, size_1024, 4)


def test_default_image_size_is_512():
    model = build_model("hybrid", bond_dim=4)
    assert model.transform_512 is None
    assert _mps_grid(model.transform_1024) == (32, 32)


@pytest.mark.parametrize("model_type", ["vanilla", "hybrid", "deep_hybrid"])
def test_forward_matches_input_size(model_type):
    image_size = 64
    model = build_model(model_type, bond_dim=4, image_size=image_size).eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, image_size, image_size))
    assert out.shape == (1, 1, image_size, image_size)


def test_hybrid_rejects_mismatched_input():
    model = build_model("hybrid", bond_dim=4, image_size=64).eval()
    with torch.no_grad(), pytest.raises(RuntimeError):
        model(torch.randn(1, 3, 128, 128))


@pytest.mark.parametrize("image_size", [0, 100, 250])
def test_image_size_must_be_multiple_of_16(image_size):
    with pytest.raises(ValueError, match="multiple of 16"):
        build_model("vanilla", bond_dim=4, image_size=image_size)


def test_unknown_model_type_raises():
    with pytest.raises(ValueError, match="Unknown model type"):
        build_model("resnet", bond_dim=4)


def test_run_name_includes_loss_and_size():
    assert (
        build_run_name("cvc_clinicdb", "hybrid", 32, "focal_tversky", 256)
        == "cvc_clinicdb__hybrid_unet_b32_ckpt__focal_tversky__s256"
    )
    assert (
        build_run_name("kvasir_seg", "vanilla", 32, "bce_dice", 512, tag="r1")
        == "kvasir_seg__vanilla_unet__bce_dice__s512__r1"
    )
