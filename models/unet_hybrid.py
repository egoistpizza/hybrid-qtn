from typing import Optional

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from models.tensor_train import TTConv2d
from models.unet_classic import DoubleConv, UpBlock

ENTRY_TT_RANK = 128


def encoder_stage(in_channels: int, out_channels: int, tensorized: bool, entry_rank: int) -> nn.Sequential:
    if not tensorized:
        return nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))
    return nn.Sequential(
        nn.MaxPool2d(2),
        TTConv2d(in_channels, out_channels, kernel_size=3, max_rank=entry_rank),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class UNetDeepHybrid(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        transform_512: Optional[nn.Module] = None,
        transform_1024: Optional[nn.Module] = None,
        entry_rank: int = ENTRY_TT_RANK,
    ):
        super().__init__()
        self.use_checkpointing = True
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
        if self.use_checkpointing and self.training:
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
