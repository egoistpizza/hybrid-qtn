import torch
import torch.nn as nn

class SpatialFlattening(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        b, c, h, w = x.shape
        
        # Flatten spatial dims and swap to sequence format: B x (H*W) x C
        x_seq = x.view(b, c, h * w).permute(0, 2, 1).contiguous()
        
        return x_seq, (h, w)

    def reverse(self, x_seq: torch.Tensor, spatial_shape: tuple[int, int]) -> torch.Tensor:
        h, w = spatial_shape
        b, l, c = x_seq.shape
        
        # Revert sequence format back to spatial map: B x C x H x W
        x_out = x_seq.permute(0, 2, 1).contiguous().view(b, c, h, w)
        
        return x_out