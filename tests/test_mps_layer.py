import pytest
import torch

from models.mps_layer import (
    COLUMN_EQUATION,
    ROW_EQUATION,
    AxialMPSLayer,
    MPSBottleneck,
    MPSVariant,
    SeparablePositionalEncoding2D,
)
from models.tensor_train import MeanBoundaryAxialOperator, RelativeAxialOperator


@pytest.fixture
def sample_tensor() -> torch.Tensor:
    batch_size = 2
    channels = 64
    height = 32
    width = 32
    return torch.randn(batch_size, channels, height, width)


def test_positional_encoding_preserves_shape(sample_tensor: torch.Tensor):
    channels = sample_tensor.size(1)
    height = sample_tensor.size(2)
    width = sample_tensor.size(3)
    
    encoder = SeparablePositionalEncoding2D(channels=channels, height=height, width=width)
    encoded_tensor = encoder(sample_tensor)
    
    assert encoded_tensor.shape == sample_tensor.shape
    assert not torch.allclose(encoded_tensor, sample_tensor)


def test_axial_mps_layer_preserves_shape(sample_tensor: torch.Tensor):
    channels = sample_tensor.size(1)
    height = sample_tensor.size(2)
    width = sample_tensor.size(3)
    bond_dim = 16
    
    layer = AxialMPSLayer(in_channels=channels, bond_dim=bond_dim, height=height, width=width)
    output_tensor = layer(sample_tensor)
    
    assert output_tensor.shape == sample_tensor.shape


def test_mps_bottleneck_preserves_shape(sample_tensor: torch.Tensor):
    channels = sample_tensor.size(1)
    height = sample_tensor.size(2)
    width = sample_tensor.size(3)
    bond_dim = 16
    
    bottleneck = MPSBottleneck(in_channels=channels, bond_dim=bond_dim, height=height, width=width)
    output_tensor = bottleneck(sample_tensor)
    
    assert output_tensor.shape == sample_tensor.shape


def test_mps_bottleneck_gradient_flow(sample_tensor: torch.Tensor):
    channels = sample_tensor.size(1)
    height = sample_tensor.size(2)
    width = sample_tensor.size(3)
    bond_dim = 16
    
    bottleneck = MPSBottleneck(in_channels=channels, bond_dim=bond_dim, height=height, width=width)
    
    output_tensor = bottleneck(sample_tensor)
    dummy_loss = output_tensor.mean()
    dummy_loss.backward()
    
    tensor_train_cores = [
        *bottleneck.axial_mps.row_operator.chain.cores,
        *bottleneck.axial_mps.column_operator.chain.cores,
        *bottleneck.local_bypass.chain.cores,
    ]
    assert all(core.grad is not None for core in tensor_train_cores)
    assert all(torch.any(core.grad != 0) for core in tensor_train_cores)


def shift_response(axial_operator: str, shift: int) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    layer = AxialMPSLayer(8, 4, 16, 16, variant=MPSVariant(axial_operator)).eval()
    with torch.no_grad():
        layer.in_proj.bias.zero_()
        layer.out_proj.bias.zero_()
    blob = torch.zeros(1, 8, 16, 16)
    blob[..., 6:9, 3:6] = torch.randn(1, 8, 3, 3)
    with torch.no_grad():
        return layer(blob), layer(torch.roll(blob, shifts=shift, dims=-1))


def test_relative_layer_output_follows_a_shifted_input():
    original, shifted = shift_response("relative", shift=4)
    assert torch.allclose(shifted[..., 5:15], original[..., 1:11], atol=1e-5)


def test_circular_layer_output_follows_a_shifted_input():
    original, shifted = shift_response("circular", shift=4)
    assert torch.allclose(shifted[..., 5:15], original[..., 1:11], atol=1e-5)


def test_absolute_layer_output_does_not_follow_a_shifted_input():
    original, shifted = shift_response("absolute", shift=4)
    assert not torch.allclose(shifted[..., 5:15], original[..., 1:11], atol=1e-3)


def test_positional_encoding_is_opt_in_and_adds_its_parameters():
    def count(layer):
        return sum(parameter.numel() for parameter in layer.parameters())

    with_encoding = AxialMPSLayer(64, 16, 32, 32, variant=MPSVariant(positional_encoding=True))
    without_encoding = AxialMPSLayer(64, 16, 32, 32)
    assert without_encoding.positional_encoding is None
    assert count(with_encoding) - count(without_encoding) == 64 * (32 + 32)


@pytest.mark.parametrize("axial_operator", ["relative", "circular"])
def test_offset_operator_gradient_reaches_every_core(sample_tensor: torch.Tensor, axial_operator: str):
    bottleneck = MPSBottleneck(64, 16, 32, 32, variant=MPSVariant(axial_operator))
    bottleneck(sample_tensor).mean().backward()
    cores = [*bottleneck.axial_mps.row_operator.chain.cores, *bottleneck.axial_mps.column_operator.chain.cores]
    assert all(core.grad is not None and torch.any(core.grad != 0) for core in cores)


def mixed(x: torch.Tensor, equation: str, axis: int, mean_boundary: bool) -> torch.Tensor:
    torch.manual_seed(0)
    operator_class = MeanBoundaryAxialOperator if mean_boundary else RelativeAxialOperator
    operator = operator_class(bond_dim=4, spatial_size=8, max_rank=16)
    with torch.no_grad():
        return operator.mix(x, equation, axis)


@pytest.mark.parametrize("equation, axis", [(ROW_EQUATION, 2), (COLUMN_EQUATION, 3)])
def test_mean_boundary_maps_a_uniform_input_to_a_position_independent_output(equation, axis):
    uniform = torch.randn(2, 4, 1, 1).expand(2, 4, 8, 8).contiguous()
    output = mixed(uniform, equation, axis, mean_boundary=True)
    assert torch.allclose(output, output.mean(dim=axis, keepdim=True).expand_as(output), atol=1e-5)
    zero_edges = mixed(uniform, equation, axis, mean_boundary=False)
    assert not torch.allclose(zero_edges, zero_edges.mean(dim=axis, keepdim=True).expand_as(zero_edges), atol=1e-3)


def test_mean_boundary_shifts_the_output_of_a_zero_mean_pattern_with_the_input():
    pattern = torch.zeros(1, 4, 8, 8)
    pattern[..., 2:4, :] = torch.randn(1, 4, 2, 8)
    pattern[..., 4:6, :] = -pattern[..., 2:4, :]
    original = mixed(pattern, ROW_EQUATION, 2, mean_boundary=True)
    shifted = mixed(torch.roll(pattern, shifts=2, dims=2), ROW_EQUATION, 2, mean_boundary=True)
    assert torch.allclose(shifted[..., 4:, :], original[..., 2:6, :], atol=1e-5)


def test_zero_boundary_mixing_is_the_plain_einsum():
    torch.manual_seed(0)
    operator = RelativeAxialOperator(bond_dim=4, spatial_size=8, max_rank=16)
    features = torch.randn(2, 4, 8, 8)
    with torch.no_grad():
        expected = torch.einsum(ROW_EQUATION, features, operator.materialize())
        assert torch.equal(operator.mix(features, ROW_EQUATION, 2), expected)


@pytest.mark.parametrize("boundary, flat", [("mean", True), ("zero", False)])
def test_layer_response_to_a_uniform_input_is_flat_only_with_the_mean_boundary(boundary, flat):
    torch.manual_seed(0)
    layer = AxialMPSLayer(8, 4, 16, 16, variant=MPSVariant(boundary=boundary)).eval()
    with torch.no_grad():
        output = layer(torch.ones(1, 8, 16, 16))
    assert torch.allclose(output, output[..., :1, :1].expand_as(output), atol=1e-4) == flat


def test_mean_boundary_gradient_reaches_every_core(sample_tensor: torch.Tensor):
    bottleneck = MPSBottleneck(64, 16, 32, 32, variant=MPSVariant(boundary="mean"))
    bottleneck(sample_tensor).mean().backward()
    cores = [*bottleneck.axial_mps.row_operator.chain.cores, *bottleneck.axial_mps.column_operator.chain.cores]
    assert all(core.grad is not None and torch.any(core.grad != 0) for core in cores)
