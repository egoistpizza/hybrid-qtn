import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)

CUBLAS_DETERMINISTIC_WORKSPACE = ":4096:8"
UINT32_MODULUS = 2**32


def seed_everything(seed: int) -> None:
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
    return torch.initial_seed() % UINT32_MODULUS
