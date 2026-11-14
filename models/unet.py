"""U-Net (Ronneberger et al., 2015) with bilinear upsampling and optional bottleneck transforms.

Without transforms this is the baseline U-Net. The hybrid variants place a transform after
the 512- and/or 1024-channel encoder stages; such a stage uses a tensorized entry convolution.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from models.tensor_train import TTConv2d

ENTRY_TT_RANK = 128


class DoubleConv(nn.Module):
    """Two ``Conv3x3 -> BatchNorm -> ReLU`` blocks; padding keeps the spatial size."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class UpBlock(nn.Module):
    """Decoder stage: bilinear 2x upsampling and a 1x1 channel reduction, then concatenation
    with the encoder skip features and a :class:`DoubleConv`.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
        )
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Upsample ``x1`` and fuse it with the skip connection ``x2``."""
        return self.conv(torch.cat([x2, self.up(x1)], dim=1))


def encoder_stage(in_channels: int, out_channels: int, tensorized: bool, entry_rank: int) -> nn.Sequential:
    """Downsampling encoder stage.

    A tensorized stage replaces the :class:`DoubleConv` with a single TT-Conv3x3 entry
    convolution followed by BatchNorm and ReLU; the stage's transform is applied after it.
    """
    if not tensorized:
        return nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))
    return nn.Sequential(
        nn.MaxPool2d(2),
        TTConv2d(in_channels, out_channels, kernel_size=3, max_rank=entry_rank),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """Four-level U-Net (64 to 1024 channels) returning per-pixel logits.

    ``vanilla`` models set no transform, ``hybrid`` models set only ``transform_1024`` (the
    bottleneck), and ``deep_hybrid`` models set both.

    Args:
        in_channels: Input image channels.
        out_channels: Output logit channels.
        transform_512: Module applied to the 512-channel features, or ``None``.
        transform_1024: Module applied to the 1024-channel bottleneck features, or ``None``.
        entry_rank: Maximum bond rank of the tensorized entry convolutions.
        checkpoint_transforms: Recompute the transforms in the backward pass during
            training, trading compute for activation memory.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        transform_512: Optional[nn.Module] = None,
        transform_1024: Optional[nn.Module] = None,
        entry_rank: int = ENTRY_TT_RANK,
        checkpoint_transforms: bool = True,
    ):
        super().__init__()
        self.checkpoint_transforms = checkpoint_transforms
        self.transform_512 = transform_512
        self.transform_1024 = transform_1024

        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = encoder_stage(256, 512, transform_512 is not None, entry_rank)
        self.down4 = encoder_stage(512, 1024, transform_1024 is not None, entry_rank)

        self.up1 = UpBlock(1024, 512)
        self.up2 = UpBlock(512, 256)
        self.up3 = UpBlock(256, 128)
        self.up4 = UpBlock(128, 64)
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def _apply_transform(self, transform: Optional[nn.Module], features: torch.Tensor) -> torch.Tensor:
        if transform is None:
            return features
        if self.checkpoint_transforms and self.training:
            return checkpoint.checkpoint(transform, features, use_reentrant=False)
        return transform(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self._apply_transform(self.transform_512, self.down3(x3))
        x5 = self._apply_transform(self.transform_1024, self.down4(x4))

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)
