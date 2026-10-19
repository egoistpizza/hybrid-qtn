"""Checkpoint file names and weight I/O shared by training and evaluation.

Weights are saved without the ``torch.compile`` (``_orig_mod.``) and ``AveragedModel``
(``module.``, ``n_averaged``) wrappers, so every checkpoint loads into a plain model.
Older checkpoints that still carry the wrappers are cleaned on load.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

BEST_LIVE_CHECKPOINT = "best_model.pth"
SWA_CHECKPOINT = "best_swa_model.pth"
INTERRUPTED_CHECKPOINT = "interrupted_model.pth"
WRAPPER_PREFIXES = ("_orig_mod.", "module.")
SWA_COUNTER_KEY = "n_averaged"


def clean_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Strip wrapper prefixes from the keys and drop the SWA counter."""
    cleaned = {}
    for key, value in state_dict.items():
        if key == SWA_COUNTER_KEY:
            continue
        for prefix in WRAPPER_PREFIXES:
            key = key.replace(prefix, "")
        cleaned[key] = value
    return cleaned


def save_weights(module: nn.Module, path: Path) -> Path:
    torch.save(clean_state_dict(module.state_dict()), path)
    return path


def load_weights(model: nn.Module, path: Path, device: torch.device) -> None:
    """Load a checkpoint into ``model``, or into the module it wraps when compiled.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint required but not found at: {path}")
    state_dict = torch.load(path, map_location=device, weights_only=True)
    getattr(model, "_orig_mod", model).load_state_dict(clean_state_dict(state_dict))


def evaluation_checkpoint(run_dir: Path) -> Path:
    """The SWA checkpoint of a run if it exists, otherwise the best live one."""
    swa_checkpoint = run_dir / SWA_CHECKPOINT
    return swa_checkpoint if swa_checkpoint.exists() else run_dir / BEST_LIVE_CHECKPOINT
