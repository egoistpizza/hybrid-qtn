"""Tensor-train (TT) factorized layers used by the quantum-inspired bottleneck.

A weight matrix ``W`` of shape ``(prod(n_k), prod(m_k))`` is stored as a chain of
cores ``G_k`` with shape ``(r_{k-1}, m_k, n_k, r_k)``, where ``m_k`` / ``n_k`` are
the input / output modes of site ``k`` and ``r_k`` the bond ranks. This is the
matrix product operator (MPO) form of the weight. Every forward pass contracts
the chain back into a dense weight, so the arithmetic matches a dense layer
while only the cores are stored and optimized.
"""

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

CHANNEL_MODE_COUNT = 5
MAX_CHANNEL_MODE_SIZE = 4


def is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def binary_digits(value: int) -> int:
    """Number of binary digits needed to index ``value`` items; ``value`` must be a power of two."""
    if not is_power_of_two(value):
        raise ValueError(f"Expected a power of two, got {value}")
    return value.bit_length() - 1


def quantize_dimension(
    value: int,
    mode_count: int = CHANNEL_MODE_COUNT,
    max_mode_size: int = MAX_CHANNEL_MODE_SIZE,
) -> tuple[int, ...]:
    """Factor a power-of-two dimension into ``mode_count`` modes of size at most ``max_mode_size``.

    Larger modes are placed last, e.g. ``1024 -> (4, 4, 4, 4, 4)`` and ``512 -> (2, 4, 4, 4, 4)``.

    Raises:
        ValueError: If ``value`` is not a power of two or does not fit into the modes.
    """
    remaining_bits = binary_digits(value)
    bits_per_mode = binary_digits(max_mode_size)
    modes = []
    for _ in range(mode_count):
        taken_bits = min(bits_per_mode, remaining_bits)
        modes.append(1 << taken_bits)
        remaining_bits -= taken_bits
    if remaining_bits:
        raise ValueError(f"{value} does not fit into {mode_count} modes of size ≤ {max_mode_size}")
    return tuple(reversed(modes))


def capped_ranks(site_sizes: Sequence[int], max_rank: int) -> list[int]:
    """Bond ranks of a chain with the given site sizes (``m_k * n_k``).

    Each inner rank is capped by ``max_rank`` and by the sizes of the two unfoldings it
    separates, the largest rank that can carry information; anything above would only
    add redundant parameters. The boundary ranks are 1.
    """
    return [1] + [
        min(max_rank, math.prod(site_sizes[:bond]), math.prod(site_sizes[bond:]))
        for bond in range(1, len(site_sizes))
    ] + [1]


def tensor_train_parameter_count(input_modes: Sequence[int], output_modes: Sequence[int], max_rank: int) -> int:
    """Analytic parameter count of a :class:`TensorTrain` with the same arguments."""
    site_sizes = [m * n for m, n in zip(input_modes, output_modes)]
    ranks = capped_ranks(site_sizes, max_rank)
    return sum(ranks[k] * site_sizes[k] * ranks[k + 1] for k in range(len(site_sizes)))


class TensorTrain(nn.Module):
    """Matrix in tensor-train format mapping ``prod(input_modes)`` to ``prod(output_modes)`` values.

    Args:
        input_modes: Input mode size of each site.
        output_modes: Output mode size of each site.
        max_rank: Upper bound on every bond rank, see :func:`capped_ranks`.
        target_variance: Variance of the materialized dense entries at initialization.
            Each core receives an equal share, so the contracted product matches the
            variance of a conventionally initialized dense layer.
    """

    def __init__(
        self,
        input_modes: Sequence[int],
        output_modes: Sequence[int],
        max_rank: int,
        target_variance: float,
    ):
        super().__init__()
        if len(input_modes) != len(output_modes):
            raise ValueError("input_modes and output_modes must have the same number of sites")
        self.input_modes = tuple(input_modes)
        self.output_modes = tuple(output_modes)
        self.ranks = capped_ranks([m * n for m, n in zip(input_modes, output_modes)], max_rank)
        self.cores = nn.ParameterList(
            nn.Parameter(torch.empty(self.ranks[k], m, n, self.ranks[k + 1]))
            for k, (m, n) in enumerate(zip(self.input_modes, self.output_modes))
        )
        self._initialize(target_variance)

    @property
    def input_size(self) -> int:
        return math.prod(self.input_modes)

    @property
    def output_size(self) -> int:
        return math.prod(self.output_modes)

    def _initialize(self, target_variance: float) -> None:
        per_site_variance = target_variance ** (1.0 / len(self.cores))
        for core in self.cores:
            nn.init.normal_(core, std=math.sqrt(per_site_variance / core.shape[-1]))

    def materialize(self) -> torch.Tensor:
        """Contract the cores into the dense ``(input_size, output_size)`` matrix.

        The contraction runs in float32 even under autocast: the product of many small
        cores loses too much precision in half precision.
        """
        with torch.autocast(device_type=self.cores[0].device.type, enabled=False):
            result = self.cores[0].squeeze(0)
            for core in list(self.cores)[1:]:
                rows, columns, _ = result.shape
                result = torch.einsum("abr,rmns->ambns", result, core).reshape(
                    rows * core.shape[1], columns * core.shape[2], core.shape[3]
                )
            return result.squeeze(-1)


class TTConv2d(nn.Module):
    """2D convolution whose kernel is stored as a tensor train.

    The first site holds the ``kernel_size x kernel_size`` spatial taps (omitted for a
    1x1 kernel); the remaining sites hold the quantized input and output channels.

    Args:
        in_channels: Input channels; must be a power of two.
        out_channels: Output channels; must be a power of two.
        kernel_size: Odd square kernel size; padding keeps the spatial size.
        max_rank: Upper bound on the bond ranks.
        gain: Initialization gain; 2.0 matches Kaiming initialization for ReLU.
        bias: Whether to add a learnable bias.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        max_rank: int,
        gain: float = 2.0,
        bias: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        spatial_modes = (kernel_size * kernel_size,) if kernel_size > 1 else ()
        spatial_outputs = (1,) if kernel_size > 1 else ()
        self.chain = TensorTrain(
            input_modes=spatial_modes + quantize_dimension(in_channels),
            output_modes=spatial_outputs + quantize_dimension(out_channels),
            max_rank=max_rank,
            target_variance=gain / (in_channels * kernel_size * kernel_size),
        )
        self.bias: Optional[nn.Parameter] = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def weight(self) -> torch.Tensor:
        """Dense convolution kernel of shape ``(out_channels, in_channels, k, k)``."""
        kernel_area = self.kernel_size * self.kernel_size
        return (
            self.chain.materialize()
            .reshape(kernel_area, self.in_channels, self.out_channels)
            .permute(2, 1, 0)
            .reshape(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight(), self.bias, padding=self.padding)


def interleaved_site_roles(spatial_bits: int, channel_bits: int) -> tuple[str, ...]:
    """Site order of an axial operator: spatial and channel bits alternate, most significant first.

    Interleaving places the spatial and channel digits of the same scale next to each
    other, so the chain runs from coarse to fine structure in both indices at once.
    """
    roles = []
    for position in range(max(spatial_bits, channel_bits)):
        if position < spatial_bits:
            roles.append("spatial")
        if position < channel_bits:
            roles.append("channel")
    return tuple(roles)


class AbsoluteAxialOperator(nn.Module):
    """Absolute-position linear operator along one image axis on the joint (channel, position) index.

    Every pair of input and output positions has its own weights, so the operator is not
    shift-equivariant; :class:`RelativeAxialOperator` is the shift-equivariant alternative.

    The ``bond_dim * spatial_size`` values of one image column (or row) are indexed by
    the binary digits of the channel and the position, one chain site per digit. The
    operator is a matrix product operator over those sites with bond rank ``max_rank``.

    Args:
        bond_dim: Latent channel width; must be a power of two.
        spatial_size: Length of the mixed axis; must be a power of two.
        max_rank: Bond dimension (chi) of the chain.
    """

    def __init__(self, bond_dim: int, spatial_size: int, max_rank: int):
        super().__init__()
        self.bond_dim = bond_dim
        self.spatial_size = spatial_size
        self.site_roles = interleaved_site_roles(binary_digits(spatial_size), binary_digits(bond_dim))
        site_count = len(self.site_roles)
        self.chain = TensorTrain(
            input_modes=(2,) * site_count,
            output_modes=(2,) * site_count,
            max_rank=max_rank,
            target_variance=1.0 / (bond_dim * spatial_size),
        )
        channel_sites = [site for site, role in enumerate(self.site_roles) if role == "channel"]
        spatial_sites = [site for site, role in enumerate(self.site_roles) if role == "spatial"]
        self.axis_order = (
            *channel_sites,
            *spatial_sites,
            *(site_count + site for site in spatial_sites),
            *(site_count + site for site in channel_sites),
        )

    def materialize(self) -> torch.Tensor:
        """Dense operator of shape ``(in_channel, in_position, out_position, out_channel)``."""
        return (
            self.chain.materialize()
            .reshape((2,) * (2 * len(self.site_roles)))
            .permute(self.axis_order)
            .reshape(self.bond_dim, self.spatial_size, self.spatial_size, self.bond_dim)
        )


class RelativeAxialOperator(nn.Module):
    """Relative-position (Toeplitz) linear operator along one image axis.

    The weight between input position ``h`` and output position ``j`` depends only on the
    offset ``j - h``: ``W[c, h, j, d] = K[c, d, j - h]``. The kernel ``K`` spans every offset
    of the axis, so each output still sees the whole row or column, and shifting the input
    shifts the output. Offsets beyond the axis contribute zero (no wrap-around).

    ``K`` is a tensor train over the binary digits of the offset (``2 * spatial_size``
    slots) interleaved with the digits of the input and output channels; offset sites have
    no output digit. This is the quantized tensor-train form of a Toeplitz operator.

    Args:
        bond_dim: Latent channel width; must be a power of two.
        spatial_size: Length of the mixed axis; must be a power of two.
        max_rank: Bond dimension (chi) of the chain.
    """

    def __init__(self, bond_dim: int, spatial_size: int, max_rank: int):
        super().__init__()
        self.bond_dim = bond_dim
        self.spatial_size = spatial_size
        self.offset_slots = self.offset_slot_count(spatial_size)
        self.site_roles = interleaved_site_roles(binary_digits(self.offset_slots), binary_digits(bond_dim))
        self.chain = TensorTrain(
            input_modes=(2,) * len(self.site_roles),
            output_modes=tuple(2 if role == "channel" else 1 for role in self.site_roles),
            max_rank=max_rank,
            target_variance=1.0 / (bond_dim * spatial_size),
        )
        channel_sites = [site for site, role in enumerate(self.site_roles) if role == "channel"]
        offset_sites = [site for site, role in enumerate(self.site_roles) if role == "spatial"]
        output_digits = [len(self.site_roles) + digit for digit in range(len(channel_sites))]
        self.axis_order = (*channel_sites, *offset_sites, *output_digits)
        self.register_buffer("offset_index", self.offset_table(spatial_size), persistent=False)

    @staticmethod
    def offset_slot_count(spatial_size: int) -> int:
        return 2 * spatial_size

    @staticmethod
    def offset_table(spatial_size: int) -> torch.Tensor:
        """Kernel slot of every ``(in_position, out_position)`` pair: slot ``k`` holds offset
        ``k - spatial_size``; slot 0 (offset ``-spatial_size``) is never used."""
        positions = torch.arange(spatial_size)
        return positions[None, :] - positions[:, None] + spatial_size

    def kernel(self) -> torch.Tensor:
        """Kernel of shape ``(in_channel, offset_slots, out_channel)``, indexed by :meth:`offset_table`."""
        digit_count = len(self.site_roles) + self.site_roles.count("channel")
        return (
            self.chain.materialize()
            .reshape((2,) * digit_count)
            .permute(self.axis_order)
            .reshape(self.bond_dim, self.offset_slots, self.bond_dim)
        )

    def materialize(self) -> torch.Tensor:
        """Dense operator of shape ``(in_channel, in_position, out_position, out_channel)``."""
        return self.kernel()[:, self.offset_index, :]


class CircularAxialOperator(RelativeAxialOperator):
    """Circulant variant of :class:`RelativeAxialOperator`: ``W[c, h, j, d] = K[c, d, (j - h) mod N]``.

    Offsets wrap around the axis, so the operator has no edges from which absolute position
    could be inferred and commutes exactly with circular shifts. The kernel has ``N`` offset
    slots instead of ``2N``.
    """

    @staticmethod
    def offset_slot_count(spatial_size: int) -> int:
        return spatial_size

    @staticmethod
    def offset_table(spatial_size: int) -> torch.Tensor:
        positions = torch.arange(spatial_size)
        return (positions[None, :] - positions[:, None]) % spatial_size
