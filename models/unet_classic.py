import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """(Conv2d -> BatchNorm -> ReLU) x 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UpBlock(nn.Module):
    """Upscaling (Bilinear) then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        #: Bilinear Upsample + 1x1 Conv2d instead of ConvTranspose2d
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        )
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Skip connection Concat
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    # in_channels=3 for RGB 
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        
        # 4 block for encoder
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        
        # Bottleneck filter aranged as 1024
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))
        
        # Decoder
        self.up1 = UpBlock(1024, 512)
        self.up2 = UpBlock(512, 256)
        self.up3 = UpBlock(256, 128)
        self.up4 = UpBlock(128, 64)
        
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder Path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # Bottleneck
        x5 = self.down4(x4)
        
        # Decoder Path
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        logits = self.outc(x)
        return logits

#  Dummy Tensor Test
if __name__ == "__main__":
    print("🚀 Dummy Tensor Test starting..")
    
    # 1. Create model
    model = UNet(in_channels=3, out_channels=1)
    
    # 2. Sahte (Dummy) input tensor (Batch=1, Channels=3, H=512, W=512)
    dummy_input = torch.randn(1, 3, 512, 512)
    print(f"📥 Input size:  {dummy_input.shape}")
    
    # 3. Tensor to model
    with torch.no_grad():
        output = model(dummy_input)
        
    print(f"📤 Output size: {output.shape}")
    
    # 4. Doğrulama
    assert output.shape == (1, 1, 512, 512), "Error: Shape mismatch!"
    print("✅ SUCCESS: 512x512 RGB images can be processed...")