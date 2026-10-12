"""Analytic and measured parameter counts of the standard and the tensorized U-Net bottleneck.

Every analytic total is checked against the instantiated model, so the reported numbers
cannot drift from the implementation. Run as a module from the repository root:

    python -m scripts.bottleneck_profiler --bond_dims 16 32 64
"""

import argparse
from dataclasses import dataclass

import torch.nn as nn

from models.factory import (
    BOTTLENECK_CHANNELS,
    BOTTLENECK_DOWNSAMPLING_FACTOR,
    PRE_BOTTLENECK_CHANNELS,
    build_segmentation_model,
    spatial_size_at,
)
from models.mps_layer import AXIAL_OPERATOR_RANK, BYPASS_TT_RANK
from models.tensor_train import binary_digits, capped_ranks, quantize_dimension, tensor_train_parameter_count
from models.unet_hybrid import ENTRY_TT_RANK

CONV_KERNEL_AREA = 9
BATCHNORM_PARAMETERS_PER_CHANNEL = 2


@dataclass(frozen=True)
class ParameterComponent:
    """One layer of a profile with its parameter formula and count."""

    name: str
    formula: str
    count: int


@dataclass(frozen=True)
class BottleneckProfile:
    """Ordered layer components of one bottleneck variant."""

    title: str
    components: tuple[ParameterComponent, ...]

    @property
    def total(self) -> int:
        return sum(component.count for component in self.components)


def count_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def conv3x3_component(name: str, in_channels: int, out_channels: int, bias: bool = False) -> ParameterComponent:
    bias_count = out_channels if bias else 0
    formula = f"9·{in_channels}·{out_channels}" + (f" + {out_channels}" if bias else "")
    return ParameterComponent(name, formula, CONV_KERNEL_AREA * in_channels * out_channels + bias_count)


def batchnorm_component(name: str, channels: int) -> ParameterComponent:
    return ParameterComponent(name, f"2·{channels}", BATCHNORM_PARAMETERS_PER_CHANNEL * channels)


def standard_bottleneck_profile(in_channels: int, channels: int) -> BottleneckProfile:
    """Vanilla U-Net bottleneck: two bias-free 3x3 convolutions, each followed by BatchNorm."""
    return BottleneckProfile(
        title="Standard U-Net bottleneck (two 3×3 convolutions)",
        components=(
            conv3x3_component("Conv3×3 #1", in_channels, channels),
            batchnorm_component("BatchNorm #1", channels),
            conv3x3_component("Conv3×3 #2", channels, channels),
            batchnorm_component("BatchNorm #2", channels),
        ),
    )


def compact_sequence(values: list[int]) -> str:
    if len(set(values)) == 1:
        return f"{len(values)}×{values[0]}"
    return "(" + ",".join(str(value) for value in values) + ")"


def tensor_train_component(
    name: str, input_modes: tuple[int, ...], output_modes: tuple[int, ...], max_rank: int
) -> ParameterComponent:
    site_sizes = [m * n for m, n in zip(input_modes, output_modes)]
    ranks = capped_ranks(site_sizes, max_rank)
    formula = f"TT sites m·n={compact_sequence(site_sizes)} ranks={compact_sequence(ranks)}"
    return ParameterComponent(name, formula, tensor_train_parameter_count(input_modes, output_modes, max_rank))


def axial_operator_component(name: str, bond_dim: int, spatial_size: int, max_rank: int) -> ParameterComponent:
    site_count = binary_digits(bond_dim) + binary_digits(spatial_size)
    return tensor_train_component(name, (2,) * site_count, (2,) * site_count, max_rank)


def axial_mps_bottleneck_profile(in_channels: int, channels: int, bond_dim: int, spatial_size: int) -> BottleneckProfile:
    """Tensorized bottleneck: TT-Conv3x3 entry stage followed by the MPS bottleneck block."""
    return BottleneckProfile(
        title=f"Tensorized Axial MPS bottleneck (D={bond_dim}, H=W={spatial_size}, χ={AXIAL_OPERATOR_RANK})",
        components=(
            tensor_train_component(
                f"TT-Conv3×3 entry (R={ENTRY_TT_RANK})",
                (CONV_KERNEL_AREA, *quantize_dimension(in_channels)),
                (1, *quantize_dimension(channels)),
                ENTRY_TT_RANK,
            ),
            batchnorm_component("BatchNorm entry", channels),
            batchnorm_component("BatchNorm pre-MPS", channels),
            ParameterComponent(
                "Separable positional encoding",
                f"{channels}·({spatial_size} + {spatial_size})",
                channels * (spatial_size + spatial_size),
            ),
            conv3x3_component("Conv3×3 in_proj (C→D)", channels, bond_dim, bias=True),
            axial_operator_component("Row MPO chain", bond_dim, spatial_size, AXIAL_OPERATOR_RANK),
            axial_operator_component("Column MPO chain", bond_dim, spatial_size, AXIAL_OPERATOR_RANK),
            conv3x3_component("Conv3×3 out_proj (D→C)", bond_dim, channels, bias=True),
            tensor_train_component(
                f"TT-Conv1×1 bypass (R={BYPASS_TT_RANK})",
                quantize_dimension(channels),
                quantize_dimension(channels),
                BYPASS_TT_RANK,
            ),
            batchnorm_component("BatchNorm post-MPS", channels),
        ),
    )


def measured_standard_bottleneck(image_size: int) -> int:
    return count_parameters(build_segmentation_model("vanilla", bond_dim=1, image_size=image_size).down4)


def measured_axial_mps_bottleneck(bond_dim: int, image_size: int) -> int:
    """Parameters of the hybrid model's entry stage plus its bottleneck transform."""
    model = build_segmentation_model("hybrid", bond_dim=bond_dim, image_size=image_size)
    return count_parameters(model.down4) + count_parameters(model.transform_1024)


def verify_against_implementation(profile: BottleneckProfile, measured: int) -> None:
    """Raise if the analytic total differs from the count of the instantiated model."""
    if profile.total != measured:
        raise AssertionError(
            f"Analytic count {profile.total:,} for '{profile.title}' disagrees with the implementation ({measured:,})."
        )


SUMMARY_LABELS = ("Analytic total", "Measured from model")


def render_profile(profile: BottleneckProfile, measured: int) -> str:
    name_width = max(len(label) for label in (*SUMMARY_LABELS, *(component.name for component in profile.components)))
    formula_width = max(len(component.formula) for component in profile.components)
    lines = [profile.title]
    lines.extend(
        f"  {component.name:<{name_width}}  {component.formula:<{formula_width}}  = {component.count:>12,}"
        for component in profile.components
    )
    analytic_label, measured_label = SUMMARY_LABELS
    lines.append(f"  {analytic_label:<{name_width}}  {'':<{formula_width}}  = {profile.total:>12,}")
    lines.append(f"  {measured_label:<{name_width}}  {'':<{formula_width}}  = {measured:>12,}  ✓")
    return "\n".join(lines)


def reduction_percentage(baseline: int, candidate: int) -> float:
    return 100.0 * (baseline - candidate) / baseline


def render_comparison(standard: BottleneckProfile, axial: BottleneckProfile) -> str:
    reduction = reduction_percentage(standard.total, axial.total)
    return (
        f"Bottleneck reduction = ({standard.total:,} − {axial.total:,}) / {standard.total:,} "
        f"= {reduction:+.2f}%"
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analytic and measured parameter profile of the U-Net bottleneck")
    parser.add_argument("--bond_dims", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--image_size", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    spatial_size = spatial_size_at(arguments.image_size, BOTTLENECK_DOWNSAMPLING_FACTOR)

    standard = standard_bottleneck_profile(PRE_BOTTLENECK_CHANNELS, BOTTLENECK_CHANNELS)
    standard_measured = measured_standard_bottleneck(arguments.image_size)
    verify_against_implementation(standard, standard_measured)

    print(f"Input {arguments.image_size}×{arguments.image_size} → bottleneck feature map "
          f"{BOTTLENECK_CHANNELS}×{spatial_size}×{spatial_size}\n")
    print(render_profile(standard, standard_measured))

    for bond_dim in arguments.bond_dims:
        axial = axial_mps_bottleneck_profile(PRE_BOTTLENECK_CHANNELS, BOTTLENECK_CHANNELS, bond_dim, spatial_size)
        axial_measured = measured_axial_mps_bottleneck(bond_dim, arguments.image_size)
        verify_against_implementation(axial, axial_measured)
        print()
        print(render_profile(axial, axial_measured))
        print(render_comparison(standard, axial))


if __name__ == "__main__":
    main()
