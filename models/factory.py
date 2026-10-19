"""Model construction shared by training, evaluation, export and the parameter profiler."""

import argparse

import torch.nn as nn

from models.mps_layer import AXIAL_BOUNDARIES, AXIAL_OPERATORS, MPSBottleneck, MPSVariant
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


def add_mps_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the ``--axial_operator``, ``--positional_encoding`` and ``--axial_boundary`` options, defaulting to :class:`MPSVariant`."""
    default = MPSVariant()
    parser.add_argument(
        "--axial_operator",
        choices=tuple(AXIAL_OPERATORS),
        default=default.axial_operator,
        help="relative: weights depend only on the offset (shift-equivariant); absolute: weights per position pair",
    )
    parser.add_argument(
        "--positional_encoding",
        action=argparse.BooleanOptionalAction,
        default=default.positional_encoding,
        help="Add a learned positional encoding in the MPS blocks",
    )
    parser.add_argument(
        "--axial_boundary",
        choices=AXIAL_BOUNDARIES,
        default=default.boundary,
        help="Positions beyond the edge for relative operators: zero, or the mean along the axis",
    )


def mps_variant(arguments: argparse.Namespace) -> MPSVariant:
    """The :class:`MPSVariant` selected by the options of :func:`add_mps_arguments`."""
    return MPSVariant(arguments.axial_operator, arguments.positional_encoding, arguments.axial_boundary)


def build_segmentation_model(
    model_type: str,
    bond_dim: int,
    image_size: int = 512,
    variant: MPSVariant = MPSVariant(),
) -> nn.Module:
    """Build a segmentation network.

    Args:
        model_type: ``vanilla`` (plain U-Net), ``hybrid`` (MPS bottleneck at 1024 channels)
            or ``deep_hybrid`` (MPS blocks at 512 and 1024 channels).
        bond_dim: Latent width D of the MPS blocks; ignored for ``vanilla``.
        image_size: Square input size. The MPS blocks are sized to the resulting feature
            maps, so hybrid models only accept inputs of this size.
        variant: Axial operator type and positional encoding of every MPS block.

    Raises:
        ValueError: For an unknown model type, or a non-default ``variant`` for ``vanilla``.
    """
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unknown model_type {model_type!r}. Expected one of {MODEL_TYPES}.")

    if model_type == "vanilla":
        if variant != MPSVariant():
            raise ValueError("axial_operator and positional_encoding only apply to models with MPS blocks")
        return UNet()

    def mps_block(channels: int, downsampling_factor: int) -> MPSBottleneck:
        spatial_size = spatial_size_at(image_size, downsampling_factor)
        return MPSBottleneck(channels, bond_dim, spatial_size, spatial_size, variant=variant)

    bottleneck_transform = mps_block(BOTTLENECK_CHANNELS, BOTTLENECK_DOWNSAMPLING_FACTOR)
    pre_bottleneck_transform = (
        mps_block(PRE_BOTTLENECK_CHANNELS, PRE_BOTTLENECK_DOWNSAMPLING_FACTOR) if model_type == "deep_hybrid" else None
    )
    return UNet(transform_512=pre_bottleneck_transform, transform_1024=bottleneck_transform)
