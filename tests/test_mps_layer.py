import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pytest
import torch

from models.mps_layer import PositionalEncoding2D, AxialMPSLayer, MPSBottleneck


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
    
    encoder = PositionalEncoding2D(channels=channels, height=height, width=width)
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
    
    assert bottleneck.axial_mps.T_mid_H.grad is not None
    assert bottleneck.axial_mps.T_mid_W.grad is not None
    assert bottleneck.local_bypass.weight.grad is not None
    
    assert torch.any(bottleneck.axial_mps.T_mid_H.grad != 0)
