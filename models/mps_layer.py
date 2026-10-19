"""Axial matrix product state (MPS) bottleneck for the hybrid U-Net."""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.tensor_train import (
    AbsoluteAxialOperator,
    AxialOperator,
    CircularAxialOperator,
    MeanBoundaryAxialOperator,
    RelativeAxialOperator,
    TTConv2d,
)

AXIAL_OPERATORS = {
    "absolute": AbsoluteAxialOperator,
    "relative": RelativeAxialOperator,
    "circular": CircularAxialOperator,
}
OPERATOR_NAME_SUFFIXES = {"absolute": "", "relative": "_rel", "circular": "_circ"}
AXIAL_BOUNDARIES = ("zero", "mean")
ROW_EQUATION = "bchw,chjd->bdjw"
COLUMN_EQUATION = "bdjw,dwke->bejk"
AXIAL_OPERATOR_RANK = 64
BYPASS_TT_RANK = 32
PROJECTION_KERNEL_SIZE = 3


@dataclass(frozen=True)
class MPSVariant:
    """Design choices of the MPS block.

    Attributes:
        axial_operator: ``relative`` (default, :class:`RelativeAxialOperator`, weights depend only
            on the offset), ``absolute`` (:class:`AbsoluteAxialOperator`, separate weights per
            position pair) or ``circular`` (:class:`CircularAxialOperator`, the offset wraps around
            the axis; kept to reproduce a reported ablation).
        positional_encoding: Whether a learned positional encoding is added to the block input.
        boundary: How the block treats positions beyond the edge: ``zero`` (default), or ``mean``
            (kept to reproduce a reported ablation), where relative operators fill them with the
            mean of the features along the mixed axis (:class:`MeanBoundaryAxialOperator`) and the
            3x3 projections pad by replication, so that a uniform input gives a uniform output.
    """

    axial_operator: str = "relative"
    positional_encoding: bool = False
    boundary: str = "zero"

    def __post_init__(self) -> None:
        if self.axial_operator not in AXIAL_OPERATORS:
            raise ValueError(
                f"Unknown axial_operator {self.axial_operator!r}. Expected one of {tuple(AXIAL_OPERATORS)}."
            )
        if self.boundary not in AXIAL_BOUNDARIES:
            raise ValueError(f"Unknown boundary {self.boundary!r}. Expected one of {AXIAL_BOUNDARIES}.")
        if self.boundary != "zero" and self.axial_operator != "relative":
            raise ValueError(f"The {self.boundary} boundary needs the relative axial operator.")

    @property
    def name_suffix(self) -> str:
        """``_rel`` / ``_circ`` for relative / circular operators, ``_mean`` for the mean boundary and ``_nope`` without positional encoding.

        The suffix describes the settings, not the defaults, so names stay comparable across versions.
        """
        return (
            OPERATOR_NAME_SUFFIXES[self.axial_operator]
            + ("_mean" if self.boundary == "mean" else "")
            + ("" if self.positional_encoding else "_nope")
        )


def axial_operator_class(variant: MPSVariant) -> type[AxialOperator]:
    """Operator class of ``variant``: the mean boundary replaces the relative operator's zero edges."""
    return MeanBoundaryAxialOperator if variant.boundary == "mean" else AXIAL_OPERATORS[variant.axial_operator]


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
    row operator and a column operator with a GELU in between, and projected back.

    Args:
        in_channels: Channels of the incoming feature map.
        bond_dim: Latent channel width D seen by the axial operators.
        height: Feature-map height; fixes the size of the row operator.
        width: Feature-map width; fixes the size of the column operator.
        operator_rank: Bond dimension (chi) of the axial operators.
        variant: Axial operator type, positional encoding and boundary.
    """

    def __init__(
        self,
        in_channels: int,
        bond_dim: int,
        height: int,
        width: int,
        operator_rank: int = AXIAL_OPERATOR_RANK,
        variant: MPSVariant = MPSVariant(),
    ):
        super().__init__()
        operator = axial_operator_class(variant)
        self.positional_encoding = (
            SeparablePositionalEncoding2D(in_channels, height, width) if variant.positional_encoding else None
        )
        # Zero padding would turn a uniform input into a border ring that the axis-wide operators spread.
        padding_mode = "replicate" if variant.boundary == "mean" else "zeros"
        self.in_proj = nn.Conv2d(
            in_channels, bond_dim, PROJECTION_KERNEL_SIZE, padding=PROJECTION_KERNEL_SIZE // 2, padding_mode=padding_mode
        )
        self.row_operator = operator(bond_dim, height, operator_rank)
        self.column_operator = operator(bond_dim, width, operator_rank)
        self.out_proj = nn.Conv2d(
            bond_dim, in_channels, PROJECTION_KERNEL_SIZE, padding=PROJECTION_KERNEL_SIZE // 2, padding_mode=padding_mode
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Row step: (c, h) -> (d, j) for every column w. Column step: (d, w) -> (e, k) for every row j.
        if self.positional_encoding is not None:
            x = self.positional_encoding(x)
        projected = self.in_proj(x)
        row_mixed = F.gelu(self.row_operator.mix(projected, ROW_EQUATION, axis=2))
        column_mixed = self.column_operator.mix(row_mixed, COLUMN_EQUATION, axis=3)
        return self.out_proj(column_mixed)


class MPSBottleneck(nn.Module):
    """Residual bottleneck block combining the axial MPS layer with a local TT-Conv1x1 bypass.

    Computes ``x + GELU(BN(axial_mps(BN(x)) + bypass(BN(x))))``; input and output shapes
    are identical, so the block can follow any encoder stage. ``variant`` is passed to
    :class:`AxialMPSLayer`.
    """

    def __init__(
        self,
        in_channels: int,
        bond_dim: int,
        height: int,
        width: int,
        operator_rank: int = AXIAL_OPERATOR_RANK,
        bypass_rank: int = BYPASS_TT_RANK,
        variant: MPSVariant = MPSVariant(),
    ):
        super().__init__()
        self.norm_pre = nn.BatchNorm2d(in_channels)
        self.axial_mps = AxialMPSLayer(in_channels, bond_dim, height, width, operator_rank, variant)
        self.local_bypass = TTConv2d(in_channels, in_channels, kernel_size=1, max_rank=bypass_rank, gain=1.0)
        self.norm_post = nn.BatchNorm2d(in_channels)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.norm_pre(x)
        fused = self.axial_mps(normalized) + self.local_bypass(normalized)
        return self.activation(self.norm_post(fused)) + x
