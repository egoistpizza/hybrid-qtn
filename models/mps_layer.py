import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class AxialMPSLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        bond_dim: int,
        height: int,
        width: int
    ):
        super().__init__()
        self.bond_dim = bond_dim
        self.in_proj = nn.Conv2d(in_channels, bond_dim, kernel_size=3, padding=1)
        
        self.T_mid_H = nn.Parameter(torch.empty(bond_dim, height, height, bond_dim))
        self.T_mid_W = nn.Parameter(torch.empty(bond_dim, width, width, bond_dim))
        
        self.out_proj = nn.Conv2d(bond_dim, in_channels, kernel_size=3, padding=1)
        
        self._init_weights()

    def _init_weights(self) -> None:
        stdv = 1.0 / math.sqrt(self.bond_dim)
        nn.init.trunc_normal_(self.T_mid_H, std=stdv)
        nn.init.trunc_normal_(self.T_mid_W, std=stdv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected_input = self.in_proj(x)
        
        vertical_contraction = torch.einsum(
            'bchw,chjd->bdjw', 
            projected_input, 
            self.T_mid_H
        )
        
        activated_vertical = F.gelu(vertical_contraction)
        
        horizontal_contraction = torch.einsum(
            'bdjw,dwke->bejk', 
            activated_vertical, 
            self.T_mid_W
        )
        
        restored_output = self.out_proj(horizontal_contraction)
        
        return restored_output

class MPSBottleneck(nn.Module):
    def __init__(
        self,
        in_channels: int,
        bond_dim: int,
        height: int,
        width: int
    ):
        super().__init__()
        self.norm_pre = nn.BatchNorm2d(in_channels)
        
        self.axial_mps = AxialMPSLayer(
            in_channels=in_channels,
            bond_dim=bond_dim,
            height=height,
            width=width
        )
        
        self.local_bypass = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        
        self.norm_post = nn.BatchNorm2d(in_channels)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.norm_pre(x)
        
        mps_out = self.axial_mps(out)
        local_out = self.local_bypass(out)
        
        out = mps_out + local_out
        
        out = self.norm_post(out)
        out = self.activation(out)
        
        out = out + identity
        
        return out
    