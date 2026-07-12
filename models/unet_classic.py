import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """Her seviyede peş peşe 2 kez yapılan Conv işlemleri"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UNetBaseline(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNetBaseline, self).__init__()
        
        # ENCODER 
        self.down1 = DoubleConv(in_channels, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 28x28 -> 14x14
        
        self.down2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 14x14 -> 7x7
        
        # BOTTLENECK 
        self.bottleneck = DoubleConv(64, 128)
        
        # DECODER
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2) # 7x7 -> 14x14
        self.conv_up2 = DoubleConv(128, 64) # Skip connection ile birleşince 64+64=128 girdi
        
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2) # 14x14 -> 28x28
        self.conv_up1 = DoubleConv(64, 32) # Skip connection ile birleşince 32+32=64 girdi
        
        # Segmentaton Mask
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)
        
    def forward(self, x):
        # Encoder and skip connections 
        x1 = self.down1(x)
        p1 = self.pool1(x1)
        
        x2 = self.down2(p1)
        p2 = self.pool2(x2)
        
        # bottleneck
        b = self.bottleneck(p2)
        
        # Decoder 
        u2 = self.up2(b)
        merge2 = torch.cat([u2, x2], dim=1) # Kanalları yan yana ekle
        d2 = self.conv_up2(merge2)
        
        u1 = self.up1(d2)
        merge1 = torch.cat([u1, x1], dim=1)
        d1 = self.conv_up1(merge1)
        
        return self.final_conv(d1)