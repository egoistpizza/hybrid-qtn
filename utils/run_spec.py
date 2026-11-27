"""Run identity shared by training and evaluation: options, run name, checkpoint directory and model.

Training and evaluation parse the same options through :meth:`RunSpec.add_arguments`, so the
directory evaluation reads is by construction the one training wrote.
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch.nn as nn

from dataset import discover_dataset_configs, parse_dataset_arg, slugify_dataset_arg
from models.factory import MODEL_TYPES, build_segmentation_model
from models.variant import MPSVariant
from utils.run_naming import build_run_name

CHECKPOINT_ROOT = Path("checkpoints")
LOSS_CHOICES = ("focal_tversky", "bce_dice")


@dataclass(frozen=True)
class RunSpec:
    """The settings that identify a run: they determine its name, checkpoints and model."""

    model_type: str = "hybrid"
    datasets: tuple[str, ...] = ("kvasir_seg",)
    loss: str = "focal_tversky"
    seed: int = 42
    bond_dim: int = 32
    variant: MPSVariant = field(default_factory=MPSVariant)
    fold: Optional[int] = None
    tag: Optional[str] = None

    @property
    def dataset_slug(self) -> str:
        return slugify_dataset_arg(list(self.datasets))

    @property
    def run_name(self) -> str:
        return build_run_name(
            self.dataset_slug, self.model_type, self.bond_dim, self.loss, self.seed, self.tag, self.variant, self.fold
        )

    @property
    def checkpoint_dir(self) -> Path:
        return CHECKPOINT_ROOT / self.run_name

    def build_model(self, image_size: int) -> nn.Module:
        return build_segmentation_model(self.model_type, self.bond_dim, image_size, self.variant)

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser) -> None:
        """Add the options that identify a run, defaulting to :class:`RunSpec`."""
        default = RunSpec()
        available = ", ".join(sorted(discover_dataset_configs())) or "(none)"
        parser.add_argument("--model", type=str, default=default.model_type, choices=MODEL_TYPES)
        parser.add_argument(
            "--dataset", type=str, default=",".join(default.datasets),
            help=f"Comma-separated dataset configs. Available: {available}",
        )
        parser.add_argument("--loss", type=str, default=default.loss, choices=LOSS_CHOICES)
        parser.add_argument("--seed", type=int, default=default.seed, help="Training seed; the data split seed stays fixed")
        parser.add_argument("--bond_dim", type=int, default=default.bond_dim)
        MPSVariant.add_arguments(parser)
        parser.add_argument("--fold", type=int, default=default.fold, help="Cross-validation fold; needs cross_validation_folds in the config")
        parser.add_argument("--tag", type=str, default=default.tag, help="Suffix to distinguish runs of the same config")

    @classmethod
    def from_arguments(cls, arguments: argparse.Namespace) -> "RunSpec":
        """The spec selected by the options of :meth:`add_arguments`."""
        return cls(
            model_type=arguments.model,
            datasets=tuple(parse_dataset_arg(arguments.dataset)),
            loss=arguments.loss,
            seed=arguments.seed,
            bond_dim=arguments.bond_dim,
            variant=MPSVariant.from_arguments(arguments),
            fold=arguments.fold,
            tag=arguments.tag,
        )
