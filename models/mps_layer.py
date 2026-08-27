import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding2D(nn.Module):
    def __init__(self, channels: int, height: int, width: int):
        super().__init__()
        self.spatial_embeddings = nn.Parameter(torch.empty(1, channels, height, width))
        nn.init.trunc_normal_(self.spatial_embeddings, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.spatial_embeddings


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
        
        self.positional_encoding = PositionalEncoding2D(in_channels, height, width)
        self.in_proj = nn.Conv2d(in_channels, bond_dim, kernel_size=3, padding=1)
        
        self.T_mid_H = nn.Parameter(torch.empty(bond_dim, height, height, bond_dim))
        self.T_mid_W = nn.Parameter(torch.empty(bond_dim, width, width, bond_dim))
        
        self.out_proj = nn.Conv2d(bond_dim, in_channels, kernel_size=3, padding=1)
        
        self._initialize_tensor_cores()

    def _initialize_tensor_cores(self) -> None:
        standard_deviation = 1.0 / math.sqrt(self.bond_dim)
        nn.init.trunc_normal_(self.T_mid_H, std=standard_deviation)
        nn.init.trunc_normal_(self.T_mid_W, std=standard_deviation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded_input = self.positional_encoding(x)
        projected_input = self.in_proj(encoded_input)
        
        vertical_contraction = torch.einsum(
            'bchw,chjd->bdjw', 
            projected_input, 
            self.T_mid_H
        )
        
        activated_vertical_state = F.gelu(vertical_contraction)
        
        horizontal_contraction = torch.einsum(
            'bdjw,dwke->bejk', 
            activated_vertical_state, 
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
        identity_connection = x
        
        normalized_input = self.norm_pre(x)
        
        mps_features = self.axial_mps(normalized_input)
        local_features = self.local_bypass(normalized_input)
        
        fused_features = mps_features + local_features
        
        normalized_fused_features = self.norm_post(fused_features)
        activated_output = self.activation(normalized_fused_features)
        
        final_residual_output = activated_output + identity_connection
        
        return final_residual_output
