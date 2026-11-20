import argparse

import pytest
import torch

from models.factory import build_segmentation_model
from models.tensor_train import DenseToeplitzAxialOperator, RelativeAxialOperator
from models.variant import MPSVariant
from scripts import bottleneck_profiler as profiler
from utils.run_naming import build_run_name

HYBRID_NETWORK_PARAMETERS = 15_950_210
CONTROLS = [
    (MPSVariant("toeplitz_dense"), "_tdense_nope", 1_207_217),
    (MPSVariant(mixer="attention"), "_attn", 1_084_529),
    (MPSVariant(mixer="none"), "_nomix", 1_078_193),
]


def test_dense_toeplitz_operator_reproduces_the_relative_operator_with_the_same_kernel():
    torch.manual_seed(0)
    relative = RelativeAxialOperator(bond_dim=4, spatial_size=8, max_rank=8)
    dense = DenseToeplitzAxialOperator(bond_dim=4, spatial_size=8, max_rank=8)
    with torch.no_grad():
        dense.weight.copy_(relative.kernel()[:, 1:, :])
    torch.testing.assert_close(dense.materialize(), relative.materialize())


def test_dense_toeplitz_operator_has_one_weight_per_channel_pair_and_offset():
    operator = DenseToeplitzAxialOperator(bond_dim=32, spatial_size=32, max_rank=64)
    assert sum(p.numel() for p in operator.parameters()) == 32 * 32 * 63


@pytest.mark.parametrize("variant, suffix, bottleneck_parameters", CONTROLS)
def test_control_variants_keep_the_output_shape_name_and_profiled_parameter_count(variant, suffix, bottleneck_parameters):
    model = build_segmentation_model("hybrid", 32, 64, variant).eval()
    with torch.no_grad():
        assert model(torch.randn(1, 3, 64, 64)).shape == (1, 1, 64, 64)
    assert variant.name_suffix == suffix
    assert build_run_name("kvasir_seg", "hybrid", 32, "focal_tversky", 42, "kaggle", variant) == (
        f"kvasir_seg__hybrid_unet_b32{suffix}__focal_tversky__seed42__kaggle"
    )
    profile = profiler.axial_mps_bottleneck_profile(512, 1024, 32, 32, variant)
    assert profile.total == profiler.measured_axial_mps_bottleneck(32, 512, variant) == bottleneck_parameters


def test_mixer_option_round_trips_through_the_command_line():
    parser = argparse.ArgumentParser()
    MPSVariant.add_arguments(parser)
    assert MPSVariant.from_arguments(parser.parse_args(["--mixer", "attention"])) == MPSVariant(mixer="attention")


@pytest.mark.parametrize(
    "arguments",
    [{"mixer": "attention", "positional_encoding": True}, {"mixer": "none", "axial_operator": "absolute"},
     {"mixer": "none", "boundary": "mean"}, {"mixer": "dilated"}],
)
def test_non_axial_mixers_reject_axial_options(arguments):
    with pytest.raises(ValueError):
        MPSVariant(**arguments)


def test_slim_vanilla_matches_the_hybrid_network_within_one_percent():
    model = build_segmentation_model("slim_vanilla", 32)
    count = profiler.count_parameters(model)
    assert abs(count - HYBRID_NETWORK_PARAMETERS) / HYBRID_NETWORK_PARAMETERS < 0.01
    with torch.no_grad():
        assert model.eval()(torch.randn(1, 3, 64, 64)).shape == (1, 1, 64, 64)
    assert build_run_name("kvasir_seg", "slim_vanilla", 32, "focal_tversky", 42) == "kvasir_seg__slim_vanilla_unet__focal_tversky__seed42"


def test_slim_vanilla_rejects_mps_options():
    with pytest.raises(ValueError):
        build_segmentation_model("slim_vanilla", 32, variant=MPSVariant(mixer="none"))
