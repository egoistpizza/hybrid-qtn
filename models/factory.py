"""Model construction shared by training, evaluation, export and the parameter profiler."""

import torch.nn as nn

from models.mps_layer import MPSBottleneck
from models.variant import MPSVariant
from models.unet import UNet

MODEL_TYPES = ("vanilla", "slim_vanilla", "hybrid", "deep_hybrid")
BOTTLENECK_DOWNSAMPLING_FACTOR = 16
PRE_BOTTLENECK_DOWNSAMPLING_FACTOR = 8
BOTTLENECK_CHANNELS = 1024
PRE_BOTTLENECK_CHANNELS = 512
# Deepest-stage width at which the dense U-Net has the parameter count of the hybrid network
# (15,937,697 against 15,950,210), the control that separates mixing from size.
SLIM_BOTTLENECK_CHANNELS = 232


def spatial_size_at(image_size: int, downsampling_factor: int) -> int:
    """Feature-map side length after ``downsampling_factor`` for a square input."""
    if image_size % BOTTLENECK_DOWNSAMPLING_FACTOR != 0:
        raise ValueError(
            f"image_size must be divisible by {BOTTLENECK_DOWNSAMPLING_FACTOR}, got {image_size}"
        )
    return image_size // downsampling_factor


def build_segmentation_model(
    model_type: str,
    bond_dim: int,
    image_size: int = 512,
    variant: MPSVariant = MPSVariant(),
) -> nn.Module:
    """Build a segmentation network.

    Args:
        model_type: ``vanilla`` (plain U-Net), ``slim_vanilla`` (plain U-Net with a
            :data:`SLIM_BOTTLENECK_CHANNELS`-wide deepest stage), ``hybrid`` (MPS bottleneck at
            1024 channels) or ``deep_hybrid`` (MPS blocks at 512 and 1024 channels).
        bond_dim: Latent width D of the MPS blocks; ignored for ``vanilla``.
        image_size: Square input size. The MPS blocks are sized to the resulting feature
            maps, so hybrid models only accept inputs of this size.
        variant: Axial operator, positional encoding and boundary of every MPS block.

    Raises:
        ValueError: For an unknown model type, or a non-default ``variant`` for a plain U-Net.
    """
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unknown model_type {model_type!r}. Expected one of {MODEL_TYPES}.")

    if model_type in ("vanilla", "slim_vanilla"):
        if variant != MPSVariant():
            raise ValueError("mixer, axial_operator, positional_encoding and boundary only apply to models with MPS blocks")
        return UNet() if model_type == "vanilla" else UNet(bottleneck_channels=SLIM_BOTTLENECK_CHANNELS)

    def mps_block(channels: int, downsampling_factor: int) -> MPSBottleneck:
        spatial_size = spatial_size_at(image_size, downsampling_factor)
        return MPSBottleneck(channels, bond_dim, spatial_size, spatial_size, variant=variant)

    bottleneck_transform = mps_block(BOTTLENECK_CHANNELS, BOTTLENECK_DOWNSAMPLING_FACTOR)
    pre_bottleneck_transform = (
        mps_block(PRE_BOTTLENECK_CHANNELS, PRE_BOTTLENECK_DOWNSAMPLING_FACTOR) if model_type == "deep_hybrid" else None
    )
    return UNet(transform_512=pre_bottleneck_transform, transform_1024=bottleneck_transform)
