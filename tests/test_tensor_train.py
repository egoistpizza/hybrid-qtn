import itertools
import math

import pytest
import torch
import torch.nn.functional as F

from models.tensor_train import (
    AbsoluteAxialOperator,
    RelativeAxialOperator,
    TensorTrain,
    TTConv2d,
    capped_ranks,
    interleaved_site_roles,
    quantize_dimension,
    tensor_train_parameter_count,
)

IDENTITY = torch.eye(2).reshape(1, 2, 2, 1)
BIT_FLIP = torch.tensor([[0.0, 1.0], [1.0, 0.0]]).reshape(1, 2, 2, 1)


def reference_materialize(chain: TensorTrain) -> torch.Tensor:
    full = torch.zeros(chain.input_size, chain.output_size)
    for input_digits in itertools.product(*(range(m) for m in chain.input_modes)):
        for output_digits in itertools.product(*(range(n) for n in chain.output_modes)):
            product = torch.ones(1, 1)
            for core, i, o in zip(chain.cores, input_digits, output_digits):
                product = product @ core[:, i, o, :]
            row = sum(i * math.prod(chain.input_modes[k + 1 :]) for k, i in enumerate(input_digits))
            column = sum(o * math.prod(chain.output_modes[k + 1 :]) for k, o in enumerate(output_digits))
            full[row, column] = product.item()
    return full


def set_cores(chain: TensorTrain, cores: list[torch.Tensor]) -> None:
    with torch.no_grad():
        for parameter, value in zip(chain.cores, cores):
            parameter.zero_()
            parameter[:1, :, :, :1] = value


@pytest.mark.parametrize(
    "value, expected",
    [(1024, (4, 4, 4, 4, 4)), (512, (2, 4, 4, 4, 4)), (256, (1, 4, 4, 4, 4)), (64, (1, 1, 4, 4, 4))],
)
def test_quantize_dimension(value, expected):
    assert quantize_dimension(value) == expected
    assert math.prod(expected) == value


@pytest.mark.parametrize("value", [0, 3, 1000, 4096])
def test_quantize_dimension_rejects_unsupported_sizes(value):
    with pytest.raises(ValueError):
        quantize_dimension(value)


def test_capped_ranks_never_exceed_unfolding_dimensions():
    assert capped_ranks([9, 8, 16, 16, 16, 16], 128) == [1, 9, 72, 128, 128, 16, 1]
    assert capped_ranks([4] * 10, 64) == [1, 4, 16, 64, 64, 64, 64, 64, 16, 4, 1]


def test_parameter_count_formula_matches_module():
    chain = TensorTrain((3, 2, 4), (2, 5, 1), max_rank=4, target_variance=1.0)
    assert tensor_train_parameter_count((3, 2, 4), (2, 5, 1), 4) == sum(core.numel() for core in chain.cores)


def test_materialize_matches_brute_force_contraction():
    torch.manual_seed(0)
    chain = TensorTrain((2, 3, 2), (3, 1, 2), max_rank=3, target_variance=1.0)
    assert torch.allclose(chain.materialize(), reference_materialize(chain), atol=1e-6)


def test_full_rank_chain_reproduces_an_arbitrary_matrix():
    torch.manual_seed(0)
    target = torch.randn(8, 8)
    chain = TensorTrain((2, 2, 2), (2, 2, 2), max_rank=64, target_variance=1.0)
    optimizer = torch.optim.Adam(chain.parameters(), lr=0.05)
    for _ in range(1500):
        optimizer.zero_grad()
        loss = (chain.materialize() - target).pow(2).mean()
        loss.backward()
        optimizer.step()
    assert loss.item() < 1e-4


def test_initialization_matches_target_variance():
    torch.manual_seed(0)
    chain = TensorTrain((2,) * 10, (2,) * 10, max_rank=64, target_variance=1.0 / 1024)
    assert chain.materialize().var().item() == pytest.approx(1.0 / 1024, rel=0.35)


def test_tt_conv_equals_dense_conv_with_materialized_weight():
    torch.manual_seed(0)
    layer = TTConv2d(8, 16, kernel_size=3, max_rank=8, bias=True)
    with torch.no_grad():
        layer.bias.normal_()
    x = torch.randn(2, 8, 6, 6)
    assert torch.allclose(layer(x), F.conv2d(x, layer.weight(), layer.bias, padding=1), atol=1e-6)
    assert layer.weight().shape == (16, 8, 3, 3)


def test_tt_conv_weight_layout_follows_kernel_then_channel_ordering():
    layer = TTConv2d(4, 4, kernel_size=3, max_rank=64)
    matrix = layer.chain.materialize().reshape(9, 4, 4)
    weight = layer.weight()
    assert torch.equal(weight[3, 1, 2, 0], matrix[2 * 3 + 0, 1, 3])


def test_interleaved_roles_alternate_spatial_and_channel_bits():
    assert interleaved_site_roles(2, 3) == ("spatial", "channel", "spatial", "channel", "channel")


def test_identity_cores_materialize_the_identity_operator():
    operator = AbsoluteAxialOperator(bond_dim=4, spatial_size=8, max_rank=4)
    set_cores(operator.chain, [IDENTITY] * len(operator.site_roles))
    expected = torch.einsum("cd,hj->chjd", torch.eye(4), torch.eye(8))
    assert torch.equal(operator.materialize(), expected)


def test_channel_bit_flip_maps_to_the_channel_axis_only():
    operator = AbsoluteAxialOperator(bond_dim=4, spatial_size=2, max_rank=4)
    most_significant_channel_site = operator.site_roles.index("channel")
    cores = [BIT_FLIP if site == most_significant_channel_site else IDENTITY for site in range(len(operator.site_roles))]
    set_cores(operator.chain, cores)
    flipped_channels = torch.eye(4)[[2, 3, 0, 1]]
    expected = torch.einsum("cd,hj->chjd", flipped_channels, torch.eye(2))
    assert torch.equal(operator.materialize(), expected)


def test_spatial_bit_flip_maps_to_the_spatial_axis_only():
    operator = AbsoluteAxialOperator(bond_dim=2, spatial_size=4, max_rank=4)
    least_significant_spatial_site = len(operator.site_roles) - 1 - operator.site_roles[::-1].index("spatial")
    cores = [BIT_FLIP if site == least_significant_spatial_site else IDENTITY for site in range(len(operator.site_roles))]
    set_cores(operator.chain, cores)
    flipped_rows = torch.eye(4)[[1, 0, 3, 2]]
    expected = torch.einsum("cd,hj->chjd", torch.eye(2), flipped_rows)
    assert torch.equal(operator.materialize(), expected)


def test_materialization_runs_in_float32_under_autocast():
    operator = AbsoluteAxialOperator(bond_dim=4, spatial_size=4, max_rank=4)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        assert operator.materialize().dtype == torch.float32


def relative_kernel_reference(operator: RelativeAxialOperator) -> torch.Tensor:
    full = reference_materialize(operator.chain)
    bond_dim, offsets = operator.bond_dim, 2 * operator.spatial_size
    kernel = torch.zeros(bond_dim, offsets, bond_dim)
    channel_bits = operator.site_roles.count("channel")
    offset_bits = operator.site_roles.count("spatial")
    for c_in, offset, c_out in itertools.product(range(bond_dim), range(offsets), range(bond_dim)):
        channel_digits = iter(format(c_in, f"0{channel_bits}b"))
        offset_digits = iter(format(offset, f"0{offset_bits}b"))
        row_digits = [
            next(channel_digits) if role == "channel" else next(offset_digits) for role in operator.site_roles
        ]
        kernel[c_in, offset, c_out] = full[int("".join(row_digits), 2), c_out]
    return kernel


def test_relative_kernel_matches_brute_force_contraction():
    operator = RelativeAxialOperator(bond_dim=4, spatial_size=4, max_rank=8)
    assert torch.allclose(operator.kernel(), relative_kernel_reference(operator), atol=1e-6)


def test_relative_operator_depends_only_on_the_offset():
    operator = RelativeAxialOperator(bond_dim=4, spatial_size=8, max_rank=16)
    weights, kernel = operator.materialize(), operator.kernel()
    assert weights.shape == (4, 8, 8, 4)
    assert torch.equal(weights[:, :-1, :-1, :], weights[:, 1:, 1:, :])
    for h, j in itertools.product(range(8), repeat=2):
        assert torch.equal(weights[:, h, j, :], kernel[:, j - h + 8, :])


def test_relative_operator_parameter_count_follows_the_offset_and_channel_sites():
    operator = RelativeAxialOperator(bond_dim=32, spatial_size=32, max_rank=64)
    output_modes = tuple(2 if role == "channel" else 1 for role in operator.site_roles)
    expected = tensor_train_parameter_count((2,) * len(operator.site_roles), output_modes, 64)
    assert sum(parameter.numel() for parameter in operator.parameters()) == expected == 41_608
    assert expected < sum(parameter.numel() for parameter in AbsoluteAxialOperator(32, 32, 64).parameters())


def test_relative_materialization_runs_in_float32_under_autocast():
    operator = RelativeAxialOperator(bond_dim=4, spatial_size=4, max_rank=4)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        assert operator.materialize().dtype == torch.float32
