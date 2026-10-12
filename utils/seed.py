"""Seeding and determinism settings for reproducible runs."""

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)

CUBLAS_DETERMINISTIC_WORKSPACE = ":4096:8"
UINT32_MODULUS = 2**32


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch and request deterministic kernels.

    ``warn_only`` keeps training possible where no deterministic kernel exists
    (e.g. the bilinear upsampling backward pass); bit-exact GPU reproducibility is
    therefore not guaranteed.
    """
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_DETERMINISTIC_WORKSPACE
    random.seed(seed)
    np.random.seed(seed % UINT32_MODULUS)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    logger.info(f"Seed set to {seed} with deterministic algorithms enabled")


def seeded_generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def derived_worker_seed() -> int:
    """Seed of the current DataLoader worker, reduced to the 32-bit range NumPy accepts."""
    return torch.initial_seed() % UINT32_MODULUS
