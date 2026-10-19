"""Design choices of the MPS block: axial operator, positional encoding and boundary.

Each choice is one table entry holding everything it implies (operator class, run-name suffix,
padding, command-line help), so adding or reading a variant is a local change.
"""

import argparse
from dataclasses import dataclass
from typing import NamedTuple

from models.tensor_train import (
    AbsoluteAxialOperator,
    AxialOperator,
    CircularAxialOperator,
    MeanBoundaryAxialOperator,
    RelativeAxialOperator,
)


class OperatorChoice(NamedTuple):
    operator_class: type[AxialOperator]
    name_suffix: str
    help: str


class BoundaryChoice(NamedTuple):
    name_suffix: str
    padding_mode: str
    operator_class: type[AxialOperator] | None
    help: str


AXIAL_OPERATORS = {
    "relative": OperatorChoice(RelativeAxialOperator, "_rel", "weights depend only on the offset (default)"),
    "absolute": OperatorChoice(AbsoluteAxialOperator, "", "weights per position pair"),
    "circular": OperatorChoice(
        CircularAxialOperator, "_circ", "the offset wraps around the axis (reproduces a reported ablation)"
    ),
}

# Zero padding would turn a uniform input into a border ring that the axis-wide operators spread,
# so the mean boundary also pads the block's 3x3 projections by replication.
AXIAL_BOUNDARIES = {
    "zero": BoundaryChoice("", "zeros", None, "positions beyond the edge contribute zero (default)"),
    "mean": BoundaryChoice(
        "_mean",
        "replicate",
        MeanBoundaryAxialOperator,
        "positions beyond the edge take the mean along the axis; relative operators only "
        "(reproduces a reported ablation)",
    ),
}


@dataclass(frozen=True)
class MPSVariant:
    """Design choices of the MPS block.

    Attributes:
        axial_operator: Key of :data:`AXIAL_OPERATORS`; ``relative`` by default.
        positional_encoding: Whether a learned positional encoding is added to the block input.
        boundary: Key of :data:`AXIAL_BOUNDARIES`; ``zero`` by default. ``mean`` needs the
            relative operator.
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
            raise ValueError(f"Unknown boundary {self.boundary!r}. Expected one of {tuple(AXIAL_BOUNDARIES)}.")
        if self.boundary != "zero" and self.axial_operator != "relative":
            raise ValueError(f"The {self.boundary} boundary needs the relative axial operator.")

    @property
    def operator_class(self) -> type[AxialOperator]:
        boundary_operator = AXIAL_BOUNDARIES[self.boundary].operator_class
        return boundary_operator or AXIAL_OPERATORS[self.axial_operator].operator_class

    @property
    def padding_mode(self) -> str:
        """Padding of the block's 3x3 projections."""
        return AXIAL_BOUNDARIES[self.boundary].padding_mode

    @property
    def name_suffix(self) -> str:
        """Run-name suffix, e.g. ``_rel_nope``.

        It describes the settings, not the defaults, so names stay comparable across versions.
        """
        return (
            AXIAL_OPERATORS[self.axial_operator].name_suffix
            + AXIAL_BOUNDARIES[self.boundary].name_suffix
            + ("" if self.positional_encoding else "_nope")
        )

    @property
    def description(self) -> str:
        boundary = f", {self.boundary} boundary" if self.axial_operator == "relative" else ""
        encoding = "with" if self.positional_encoding else "no"
        return f"{self.axial_operator} operators{boundary}, {encoding} positional encoding"

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser) -> None:
        """Add ``--axial_operator``, ``--positional_encoding`` and ``--axial_boundary``, defaulting to :class:`MPSVariant`."""
        default = MPSVariant()
        parser.add_argument(
            "--axial_operator",
            choices=tuple(AXIAL_OPERATORS),
            default=default.axial_operator,
            help="; ".join(f"{name}: {choice.help}" for name, choice in AXIAL_OPERATORS.items()),
        )
        parser.add_argument(
            "--positional_encoding",
            action=argparse.BooleanOptionalAction,
            default=default.positional_encoding,
            help="Add a learned positional encoding in the MPS blocks",
        )
        parser.add_argument(
            "--axial_boundary",
            choices=tuple(AXIAL_BOUNDARIES),
            default=default.boundary,
            help="; ".join(f"{name}: {choice.help}" for name, choice in AXIAL_BOUNDARIES.items()),
        )

    @classmethod
    def from_arguments(cls, arguments: argparse.Namespace) -> "MPSVariant":
        """The variant selected by the options of :meth:`add_arguments`."""
        return cls(arguments.axial_operator, arguments.positional_encoding, arguments.axial_boundary)
