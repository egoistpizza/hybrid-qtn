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
    if not is_power_of_two(value):
        raise ValueError(f"Expected a power of two, got {value}")
    return value.bit_length() - 1


def quantize_dimension(
    value: int,
    mode_count: int = CHANNEL_MODE_COUNT,
    max_mode_size: int = MAX_CHANNEL_MODE_SIZE,
) -> tuple[int, ...]:
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
    return [1] + [
        min(max_rank, math.prod(site_sizes[:bond]), math.prod(site_sizes[bond:]))
        for bond in range(1, len(site_sizes))
    ] + [1]


def tensor_train_parameter_count(input_modes: Sequence[int], output_modes: Sequence[int], max_rank: int) -> int:
    site_sizes = [m * n for m, n in zip(input_modes, output_modes)]
    ranks = capped_ranks(site_sizes, max_rank)
    return sum(ranks[k] * site_sizes[k] * ranks[k + 1] for k in range(len(site_sizes)))


class TensorTrain(nn.Module):
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
        with torch.autocast(device_type=self.cores[0].device.type, enabled=False):
            result = self.cores[0].squeeze(0)
            for core in list(self.cores)[1:]:
                rows, columns, _ = result.shape
                result = torch.einsum("abr,rmns->ambns", result, core).reshape(
                    rows * core.shape[1], columns * core.shape[2], core.shape[3]
                )
            return result.squeeze(-1)


class TTConv2d(nn.Module):
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
    roles = []
    for position in range(max(spatial_bits, channel_bits)):
        if position < spatial_bits:
            roles.append("spatial")
        if position < channel_bits:
            roles.append("channel")
    return tuple(roles)


class QuantizedAxialOperator(nn.Module):
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
        return (
            self.chain.materialize()
            .reshape((2,) * (2 * len(self.site_roles)))
            .permute(self.axis_order)
            .reshape(self.bond_dim, self.spatial_size, self.spatial_size, self.bond_dim)
        )
