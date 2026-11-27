"""Seeding and determinism settings for reproducible runs."""

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)

CUBLAS_DETERMINISTIC_WORKSPACE = ":4096:8"
UINT32_MODULUS = 2**32


def seed_everything(seed: int, deterministic_algorithms: bool = False) -> None:
    """Seed Python, NumPy and PyTorch and fix the cuDNN algorithm choice.

    With ``deterministic_algorithms``, PyTorch also uses deterministic kernels, so a rerun
    on the same GPU type reproduces the weights bit for bit. This makes a training step
    about four times slower, mostly in the bilinear upsampling backward pass.
    """
    if deterministic_algorithms:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_DETERMINISTIC_WORKSPACE
    random.seed(seed)
    np.random.seed(seed % UINT32_MODULUS)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(deterministic_algorithms, warn_only=True)
    logger.info(f"Seed set to {seed}; deterministic algorithms {'on' if deterministic_algorithms else 'off'}")


def seeded_generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def derived_worker_seed() -> int:
    """Seed of the current DataLoader worker, reduced to the 32-bit range NumPy accepts."""
    return torch.initial_seed() % UINT32_MODULUS
