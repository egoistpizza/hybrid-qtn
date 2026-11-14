"""Run and checkpoint naming shared by training and evaluation."""

from typing import Optional

from models.variant import MPSVariant


def architecture_name(model_type: str, bond_dim: int, variant: MPSVariant = MPSVariant()) -> str:
    """``vanilla_unet``, or ``<model_type>_unet_b<bond_dim>`` followed by :attr:`MPSVariant.name_suffix`."""
    if model_type == "vanilla":
        return "vanilla_unet"
    return f"{model_type}_unet_b{bond_dim}{variant.name_suffix}"


def build_run_name(
    dataset: str,
    model_type: str,
    bond_dim: int,
    loss: str,
    seed: int,
    tag: Optional[str] = None,
    variant: MPSVariant = MPSVariant(),
    fold: Optional[int] = None,
) -> str:
    """Name of the form ``<dataset>__<architecture>__<loss>__seed<seed>[__fold<fold>][__<tag>]``.

    Hybrid architectures end in the :attr:`MPSVariant.name_suffix` of their MPS blocks.
    """
    run_name = f"{dataset}__{architecture_name(model_type, bond_dim, variant)}__{loss}__seed{seed}"
    if fold is not None:
        run_name += f"__fold{fold}"
    return f"{run_name}__{tag}" if tag else run_name
