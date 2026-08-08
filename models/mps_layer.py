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
        self.in_proj = nn.Conv2d(in_channels, bond_dim, kernel_size=1)
        
        self.T_mid_H = nn.Parameter(torch.empty(bond_dim, height, height, bond_dim))
        self.T_mid_W = nn.Parameter(torch.empty(bond_dim, width, width, bond_dim))
        
        self.out_proj = nn.Conv2d(bond_dim, in_channels, kernel_size=1)
        
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.T_mid_H, std=0.02)
        nn.init.trunc_normal_(self.T_mid_W, std=0.02)

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