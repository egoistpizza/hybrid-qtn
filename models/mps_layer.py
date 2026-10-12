"""Axial matrix product state (MPS) bottleneck for the hybrid U-Net."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.tensor_train import QuantizedAxialOperator, TTConv2d

AXIAL_OPERATOR_RANK = 64
BYPASS_TT_RANK = 32
PROJECTION_KERNEL_SIZE = 3


class SeparablePositionalEncoding2D(nn.Module):
    """Learned 2D positional encoding as the sum of a row and a column embedding.

    Uses ``channels * (height + width)`` parameters instead of ``channels * height * width``.
    """

    def __init__(self, channels: int, height: int, width: int):
        super().__init__()
        self.row_embedding = nn.Parameter(torch.empty(1, channels, height, 1))
        self.column_embedding = nn.Parameter(torch.empty(1, channels, 1, width))
        nn.init.trunc_normal_(self.row_embedding, std=0.02)
        nn.init.trunc_normal_(self.column_embedding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.row_embedding + self.column_embedding


class AxialMPSLayer(nn.Module):
    """Global context mixing along the height axis, then along the width axis.

    The input is projected from ``in_channels`` to ``bond_dim`` latent channels, mixed by a
    row operator and a column operator (each a :class:`QuantizedAxialOperator` on the joint
    channel-position index) with a GELU in between, and projected back.

    Args:
        in_channels: Channels of the incoming feature map.
        bond_dim: Latent channel width D seen by the axial operators.
        height: Feature-map height; fixes the size of the row operator.
        width: Feature-map width; fixes the size of the column operator.
        operator_rank: Bond dimension (chi) of the axial operators.
    """

    def __init__(
        self,
        in_channels: int,
        bond_dim: int,
        height: int,
        width: int,
        operator_rank: int = AXIAL_OPERATOR_RANK,
    ):
        super().__init__()
        self.positional_encoding = SeparablePositionalEncoding2D(in_channels, height, width)
        self.in_proj = nn.Conv2d(in_channels, bond_dim, kernel_size=PROJECTION_KERNEL_SIZE, padding=PROJECTION_KERNEL_SIZE // 2)
        self.row_operator = QuantizedAxialOperator(bond_dim, height, operator_rank)
        self.column_operator = QuantizedAxialOperator(bond_dim, width, operator_rank)
        self.out_proj = nn.Conv2d(bond_dim, in_channels, kernel_size=PROJECTION_KERNEL_SIZE, padding=PROJECTION_KERNEL_SIZE // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Row step: (c, h) -> (d, j) for every column w. Column step: (d, w) -> (e, k) for every row j.
        projected = self.in_proj(self.positional_encoding(x))
        row_mixed = F.gelu(torch.einsum("bchw,chjd->bdjw", projected, self.row_operator.materialize()))
        column_mixed = torch.einsum("bdjw,dwke->bejk", row_mixed, self.column_operator.materialize())
        return self.out_proj(column_mixed)


class MPSBottleneck(nn.Module):
    """Residual bottleneck block combining the axial MPS layer with a local TT-Conv1x1 bypass.

    Computes ``x + GELU(BN(axial_mps(BN(x)) + bypass(BN(x))))``; input and output shapes
    are identical, so the block can follow any encoder stage.
    """

    def __init__(
        self,
        in_channels: int,
        bond_dim: int,
        height: int,
        width: int,
        operator_rank: int = AXIAL_OPERATOR_RANK,
        bypass_rank: int = BYPASS_TT_RANK,
    ):
        super().__init__()
        self.norm_pre = nn.BatchNorm2d(in_channels)
        self.axial_mps = AxialMPSLayer(in_channels, bond_dim, height, width, operator_rank)
        self.local_bypass = TTConv2d(in_channels, in_channels, kernel_size=1, max_rank=bypass_rank, gain=1.0)
        self.norm_post = nn.BatchNorm2d(in_channels)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.norm_pre(x)
        fused = self.axial_mps(normalized) + self.local_bypass(normalized)
        return self.activation(self.norm_post(fused)) + x
