import argparse

import pytest

from models.tensor_train import (
    AbsoluteAxialOperator,
    CircularAxialOperator,
    MeanBoundaryAxialOperator,
    RelativeAxialOperator,
)
from models.variant import MPSVariant

VARIANTS = [
    (MPSVariant(), RelativeAxialOperator, "zeros", "_rel_nope", "relative operators, zero boundary, no positional encoding"),
    (MPSVariant("absolute"), AbsoluteAxialOperator, "zeros", "_nope", "absolute operators, no positional encoding"),
    (MPSVariant("absolute", True), AbsoluteAxialOperator, "zeros", "", "absolute operators, with positional encoding"),
    (MPSVariant("circular"), CircularAxialOperator, "zeros", "_circ_nope", "circular operators, no positional encoding"),
    (
        MPSVariant(boundary="mean"),
        MeanBoundaryAxialOperator,
        "replicate",
        "_rel_mean_nope",
        "relative operators, mean boundary, no positional encoding",
    ),
]


@pytest.mark.parametrize("variant, operator_class, padding_mode, suffix, description", VARIANTS)
def test_each_variant_defines_its_operator_padding_name_and_description(
    variant, operator_class, padding_mode, suffix, description
):
    assert variant.operator_class is operator_class
    assert variant.padding_mode == padding_mode
    assert variant.name_suffix == suffix
    assert variant.description == description


@pytest.mark.parametrize("variant", [entry[0] for entry in VARIANTS])
def test_command_line_options_round_trip_every_variant(variant):
    parser = argparse.ArgumentParser()
    MPSVariant.add_arguments(parser)
    arguments = ["--axial_operator", variant.axial_operator, "--axial_boundary", variant.boundary]
    arguments.append("--positional_encoding" if variant.positional_encoding else "--no-positional_encoding")
    assert MPSVariant.from_arguments(parser.parse_args(arguments)) == variant


def test_command_line_defaults_are_the_default_variant():
    parser = argparse.ArgumentParser()
    MPSVariant.add_arguments(parser)
    assert MPSVariant.from_arguments(parser.parse_args([])) == MPSVariant()


def test_default_variant_is_the_relative_operator_with_zero_edges_and_no_encoding():
    assert MPSVariant() == MPSVariant("relative", positional_encoding=False, boundary="zero")


@pytest.mark.parametrize(
    "variant_arguments, message",
    [
        ({"axial_operator": "dilated"}, "Unknown axial_operator"),
        ({"boundary": "reflect"}, "Unknown boundary"),
        ({"axial_operator": "absolute", "boundary": "mean"}, "needs the relative"),
        ({"axial_operator": "circular", "boundary": "mean"}, "needs the relative"),
    ],
)
def test_invalid_variants_raise(variant_arguments, message):
    with pytest.raises(ValueError, match=message):
        MPSVariant(**variant_arguments)
