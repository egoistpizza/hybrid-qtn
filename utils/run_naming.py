"""Run and checkpoint naming shared by training and evaluation."""

from typing import Optional


def mps_variant_suffix(axial_operator: str = "absolute", positional_encoding: bool = True) -> str:
    """``_rel`` for relative axial operators and ``_nope`` without positional encoding; empty for the defaults."""
    return ("_rel" if axial_operator == "relative" else "") + ("" if positional_encoding else "_nope")


def build_run_name(
    dataset: str,
    model_type: str,
    bond_dim: int,
    loss: str,
    seed: int,
    tag: Optional[str] = None,
    axial_operator: str = "absolute",
    positional_encoding: bool = True,
) -> str:
    """Name of the form ``<dataset>__<architecture>__<loss>__seed<seed>[__<tag>]``.

    Hybrid architectures carry the :func:`mps_variant_suffix`, so the default settings keep
    their original names.
    """
    variant = mps_variant_suffix(axial_operator, positional_encoding)
    architecture = "vanilla_unet" if model_type == "vanilla" else f"{model_type}_unet_b{bond_dim}{variant}"
    run_name = f"{dataset}__{architecture}__{loss}__seed{seed}"
    return f"{run_name}__{tag}" if tag else run_name
