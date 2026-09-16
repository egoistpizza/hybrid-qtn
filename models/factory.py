"""Single model builder shared by train.py, evaluate.py and export.py.

The U-Net itself is fully convolutional; the only input-size-dependent part is
the MPS bottleneck, whose positional encoding and tensor cores are sized to the
feature map they sit on. The 512-channel stage sees ``image_size // 8`` and the
1024-channel stage sees ``image_size // 16`` (64/32 at 512, 32/16 at 256).
"""

from __future__ import annotations

import torch.nn as nn

MODEL_TYPES = ("vanilla", "hybrid", "deep_hybrid")
DEFAULT_IMAGE_SIZE = 512
# Four MaxPool2d(2) stages between the input and the bottleneck.
DOWNSAMPLE_FACTOR = 16


def bottleneck_sizes(image_size: int) -> tuple[int, int]:
    """Spatial size of the 512- and 1024-channel feature maps for a square input."""
    if image_size <= 0 or image_size % DOWNSAMPLE_FACTOR != 0:
        raise ValueError(
            f"image_size must be a positive multiple of {DOWNSAMPLE_FACTOR}, got {image_size}"
        )
    return image_size // 8, image_size // 16


def build_model(
    model_type: str,
    bond_dim: int,
    image_size: int = DEFAULT_IMAGE_SIZE,
    in_channels: int = 3,
    out_channels: int = 1,
) -> nn.Module:
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unknown model type {model_type!r}. Choose from {MODEL_TYPES}")

    size_512, size_1024 = bottleneck_sizes(image_size)

    if model_type == "vanilla":
        from models.unet_classic import UNet
        return UNet(in_channels=in_channels, out_channels=out_channels)

    from models.mps_layer import MPSBottleneck
    from models.unet_hybrid import UNetDeepHybrid

    mps_512 = None
    if model_type == "deep_hybrid":
        mps_512 = MPSBottleneck(in_channels=512, bond_dim=bond_dim, height=size_512, width=size_512)
    mps_1024 = MPSBottleneck(in_channels=1024, bond_dim=bond_dim, height=size_1024, width=size_1024)

    return UNetDeepHybrid(
        in_channels=in_channels,
        out_channels=out_channels,
        transform_512=mps_512,
        transform_1024=mps_1024,
    )


def build_run_name(
    dataset_slug: str,
    model_type: str,
    bond_dim: int,
    loss: str,
    image_size: int,
    tag: str | None = None,
) -> str:
    """Run / checkpoint directory name. train.py and evaluate.py must agree on it."""
    model_part = "vanilla_unet" if model_type == "vanilla" else f"{model_type}_unet_b{bond_dim}_ckpt"
    run_name = f"{dataset_slug}__{model_part}__{loss}__s{image_size}"
    if tag:
        run_name = f"{run_name}__{tag}"
    return run_name
