"""Model construction shared by training, evaluation, export and the parameter profiler."""

import torch.nn as nn

from models.mps_layer import MPSBottleneck
from models.unet import UNet

MODEL_TYPES = ("vanilla", "hybrid", "deep_hybrid")
BOTTLENECK_DOWNSAMPLING_FACTOR = 16
PRE_BOTTLENECK_DOWNSAMPLING_FACTOR = 8
BOTTLENECK_CHANNELS = 1024
PRE_BOTTLENECK_CHANNELS = 512


def spatial_size_at(image_size: int, downsampling_factor: int) -> int:
    """Feature-map side length after ``downsampling_factor`` for a square input."""
    if image_size % BOTTLENECK_DOWNSAMPLING_FACTOR != 0:
        raise ValueError(
            f"image_size must be divisible by {BOTTLENECK_DOWNSAMPLING_FACTOR}, got {image_size}"
        )
    return image_size // downsampling_factor


def build_mps_bottleneck(channels: int, bond_dim: int, spatial_size: int) -> MPSBottleneck:
    return MPSBottleneck(in_channels=channels, bond_dim=bond_dim, height=spatial_size, width=spatial_size)


def build_segmentation_model(model_type: str, bond_dim: int, image_size: int = 512) -> nn.Module:
    """Build a segmentation network.

    Args:
        model_type: ``vanilla`` (plain U-Net), ``hybrid`` (MPS bottleneck at 1024 channels)
            or ``deep_hybrid`` (MPS blocks at 512 and 1024 channels).
        bond_dim: Latent width D of the MPS blocks; ignored for ``vanilla``.
        image_size: Square input size. The MPS blocks are sized to the resulting feature
            maps, so hybrid models only accept inputs of this size.
    """
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unknown model_type {model_type!r}. Expected one of {MODEL_TYPES}.")

    if model_type == "vanilla":
        return UNet()

    bottleneck_transform = build_mps_bottleneck(
        BOTTLENECK_CHANNELS, bond_dim, spatial_size_at(image_size, BOTTLENECK_DOWNSAMPLING_FACTOR)
    )
    pre_bottleneck_transform = (
        build_mps_bottleneck(
            PRE_BOTTLENECK_CHANNELS, bond_dim, spatial_size_at(image_size, PRE_BOTTLENECK_DOWNSAMPLING_FACTOR)
        )
        if model_type == "deep_hybrid"
        else None
    )
    return UNet(transform_512=pre_bottleneck_transform, transform_1024=bottleneck_transform)
