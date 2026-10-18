"""Run and checkpoint naming shared by training and evaluation."""

from typing import Optional

from models.mps_layer import MPSVariant


def build_run_name(
    dataset: str,
    model_type: str,
    bond_dim: int,
    loss: str,
    seed: int,
    tag: Optional[str] = None,
    variant: MPSVariant = MPSVariant(),
) -> str:
    """Name of the form ``<dataset>__<architecture>__<loss>__seed<seed>[__<tag>]``.

    Hybrid architectures end in the :attr:`MPSVariant.name_suffix` of their MPS blocks.
    """
    if model_type == "vanilla":
        architecture = "vanilla_unet"
    else:
        architecture = f"{model_type}_unet_b{bond_dim}{variant.name_suffix}"
    run_name = f"{dataset}__{architecture}__{loss}__seed{seed}"
    return f"{run_name}__{tag}" if tag else run_name
