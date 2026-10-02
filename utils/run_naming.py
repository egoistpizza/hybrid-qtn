from typing import Optional


def build_run_name(
    dataset: str,
    model_type: str,
    bond_dim: int,
    loss: str,
    seed: int,
    tag: Optional[str] = None,
) -> str:
    architecture = "vanilla_unet" if model_type == "vanilla" else f"{model_type}_unet_b{bond_dim}"
    run_name = f"{dataset}__{architecture}__{loss}__seed{seed}"
    return f"{run_name}__{tag}" if tag else run_name
