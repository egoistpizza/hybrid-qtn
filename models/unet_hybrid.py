import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from models.unet_classic import DoubleConv, UpBlock

class UNetDeepHybrid(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, transform_512=None, transform_1024=None):
        super().__init__()
        
        self.use_checkpointing = True 

        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))

        self.transform_512 = transform_512
        self.transform_1024 = transform_1024

        if self.transform_512 is not None:
            self.down3 = nn.Sequential(
                nn.MaxPool2d(2),
                nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
            )
        else:
            self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))

        if self.transform_1024 is not None:
            self.down4 = nn.Sequential(
                nn.MaxPool2d(2),
                nn.Conv2d(512, 1024, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(1024),
                nn.ReLU(inplace=True),
            )
        else:
            self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))

        self.up1 = UpBlock(1024, 512)
        self.up2 = UpBlock(512, 256)
        self.up3 = UpBlock(256, 128)
        self.up4 = UpBlock(128, 64)
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)

        x4 = self.down3(x3)
        if self.transform_512 is not None:
            if self.use_checkpointing and self.training:
                x4 = checkpoint.checkpoint(self.transform_512, x4, use_reentrant=False)
            else:
                x4 = self.transform_512(x4)

        x5 = self.down4(x4)
        if self.transform_1024 is not None:
            if self.use_checkpointing and self.training:
                x5 = checkpoint.checkpoint(self.transform_1024, x5, use_reentrant=False)
            else:
                x5 = self.transform_1024(x5)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        return logits
