import pytest
import torch

from models.mps_layer import AxialMPSLayer, MPSBottleneck, MPSVariant, SeparablePositionalEncoding2D


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


def test_unknown_axial_operator_raises():
    with pytest.raises(ValueError, match="Unknown axial_operator"):
        MPSVariant("dilated")


@pytest.mark.parametrize("axial_operator", ["relative", "circular"])
def test_offset_operator_gradient_reaches_every_core(sample_tensor: torch.Tensor, axial_operator: str):
    bottleneck = MPSBottleneck(64, 16, 32, 32, variant=MPSVariant(axial_operator))
    bottleneck(sample_tensor).mean().backward()
    cores = [*bottleneck.axial_mps.row_operator.chain.cores, *bottleneck.axial_mps.column_operator.chain.cores]
    assert all(core.grad is not None and torch.any(core.grad != 0) for core in cores)
