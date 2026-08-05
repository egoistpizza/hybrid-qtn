import torch
import torch.nn as nn

from models.unet_classic import DoubleConv, UpBlock


class UNetHybrid(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, bottleneck_transform=None):
        super().__init__()

        # 4 block for encoder
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))

        self.bottleneck_transform = bottleneck_transform

        if bottleneck_transform is not None:
            # Keep only: MaxPool → Conv(512→1024) → BN → ReLU
            # MPS replaces the second Conv(1024→1024) → BN → ReLU
            self.down4 = nn.Sequential(
                nn.MaxPool2d(2),
                nn.Conv2d(512, 1024, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(1024),
                nn.ReLU(inplace=True),
            )
        else:
            self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))

        # Decoder
        self.up1 = UpBlock(1024, 512)
        self.up2 = UpBlock(512, 256)
        self.up3 = UpBlock(256, 128)
        self.up4 = UpBlock(128, 64)

        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Bottleneck
        x5 = self.down4(x4)

        if self.bottleneck_transform is not None:
            x5 = self.bottleneck_transform(x5)

        # Decoder path
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        return logits
