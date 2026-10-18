"""Train a segmentation model with live validation, SWA and test-set-only reporting.

Every epoch validates the live weights; the best live checkpoint is selected on
validation Dice. Weights are averaged (SWA) over the final epochs, and both the best
live and the SWA model are evaluated once on each test set after training.
"""

import argparse
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler
from torch.optim.swa_utils import SWALR, AveragedModel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dataset import (
    DEFAULT_TEST_SET,
    DatasetSplits,
    build_loader,
    build_splits,
    discover_dataset_configs,
    parse_dataset_arg,
    slugify_dataset_arg,
)
from models.factory import MODEL_TYPES, add_mps_arguments, build_segmentation_model, mps_variant
from models.mps_layer import MPSVariant
from utils import get_device
from utils.checkpoints import BEST_LIVE_CHECKPOINT, INTERRUPTED_CHECKPOINT, SWA_CHECKPOINT, load_weights, save_weights
from utils.evaluation import evaluate_model, metric_suites
from utils.hardware import reset_peak_vram, vram_metrics
from utils.init_first import init
from utils.loss import (
    FOCAL_TVERSKY_GAMMA,
    TVERSKY_FN_WEIGHT,
    TVERSKY_FP_WEIGHT,
    BCEDiceLoss,
    FocalTverskyLoss,
)
from utils.run_logger import RunLogger, WandbLogger
from utils.run_naming import build_run_name
from utils.seed import seed_everything

logger = logging.getLogger(__name__)

LOSS_CHOICES = ("focal_tversky", "bce_dice")
GRADIENT_CLIP_NORM = 1.0
CHECKPOINT_ROOT = Path("checkpoints")

LoaderFactory = Callable[[Dataset, bool], DataLoader]


@dataclass
class TrainingConfig:
    """Hyperparameters and run settings; logged to W&B as the run config."""

    model_type: str
    dataset: str
    loss: str
    seed: int
    epochs: int
    batch_size: int
    bond_dim: int
    fn_weight: float
    fp_weight: float
    focal_gamma: float
    image_size: int
    mps_variant: MPSVariant = field(default_factory=MPSVariant)
    accumulation_steps: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    swa_start_pct: float = 0.75
    swa_lr: float = 5e-5
    deterministic: bool = False
    checkpoint_dir: str = str(CHECKPOINT_ROOT)

    @property
    def swa_start_epoch(self) -> int:
        return int(self.epochs * self.swa_start_pct)


def prefixed(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    """Prefix every metric name, e.g. ``dice`` -> ``val_dice``."""
    return {f"{prefix}_{name}": value for name, value in metrics.items()}


def qualified_test_prefix(model_prefix: str, test_set: str) -> str:
    """Metric prefix of one test set; the default set keeps the unqualified ``test_*`` keys."""
    return model_prefix if test_set == DEFAULT_TEST_SET else f"{model_prefix}_{test_set}"


class SegmentationTrainer:
    """Training loop with gradient accumulation, AMP, gradient clipping, SWA and run logging.

    ``loader_factory(dataset, shuffle)`` builds every loader, so all splits share one batch
    size and seeding policy.
    """

    def __init__(
        self,
        model: nn.Module,
        splits: DatasetSplits,
        loader_factory: LoaderFactory,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler.LRScheduler,
        device: torch.device,
        config: TrainingConfig,
        run_logger: RunLogger,
    ):
        self.model = model
        self.splits = splits
        self.loader_factory = loader_factory
        self.train_loader = loader_factory(splits.train, True)
        self.val_loader = loader_factory(splits.val.dataset, False)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.run_logger = run_logger

        self.use_amp = device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)
        self.swa_model = AveragedModel(self.model)
        self.swa_scheduler = SWALR(self.optimizer, swa_lr=config.swa_lr)
        self.best_val_dice = float("-inf")
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _images(self, batch: dict[str, Any]) -> torch.Tensor:
        return batch["image"].to(self.device, non_blocking=True, memory_format=torch.channels_last)

    def _masks(self, batch: dict[str, Any]) -> torch.Tensor:
        return batch["mask"].to(self.device, non_blocking=True)

    def _optimizer_step(self) -> None:
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=GRADIENT_CLIP_NORM)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

    def train_epoch(self, epoch: int) -> float:
        """One pass over the training set; returns the mean batch loss."""
        self.model.train()
        accumulation_steps = self.config.accumulation_steps
        total_loss = 0.0
        progress = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config.epochs} [Train]")

        for batch_index, batch in enumerate(progress, start=1):
            with torch.autocast(self.device.type, enabled=self.use_amp):
                loss = self.criterion(self.model(self._images(batch)), self._masks(batch))
            self.scaler.scale(loss / accumulation_steps).backward()

            if batch_index % accumulation_steps == 0 or batch_index == len(self.train_loader):
                self._optimizer_step()

            total_loss += loss.item()
            progress.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / len(self.train_loader)

    def evaluate(
        self,
        model: nn.Module,
        loader: DataLoader,
        description: str,
        compute_hd95: bool,
        foreground_maps: bool = False,
    ) -> dict[str, float]:
        """Image-weighted ``loss`` and per-image metrics of ``model`` on ``loader``.

        With ``foreground_maps``, the full-resolution foreground-map metrics are added;
        batches must then carry ``metadata.mask_path``.
        """
        suites = metric_suites(self.criterion, compute_hd95=compute_hd95, foreground_maps=foreground_maps)
        return evaluate_model(model, loader, self.device, suites, description)

    @torch.no_grad()
    def _recompute_swa_batchnorm(self) -> None:
        """Re-estimate BatchNorm statistics for the averaged weights with one pass over the training set."""
        self.swa_model.train()
        for module in self.swa_model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.reset_running_stats()
                module.momentum = None
        for batch in self.train_loader:
            with torch.autocast(self.device.type, enabled=self.use_amp):
                self.swa_model(self._images(batch))

    def _save_if_best(self, val_dice: float) -> None:
        if val_dice > self.best_val_dice:
            self.best_val_dice = val_dice
            save_weights(self.model, self.checkpoint_dir / BEST_LIVE_CHECKPOINT)
            logger.info(f"New best live model saved. Val Dice: {val_dice:.4f}")

    def _advance_schedule(self, in_swa_phase: bool) -> float:
        if in_swa_phase:
            self.swa_model.update_parameters(self.model)
            self.swa_scheduler.step()
            return self.swa_scheduler.get_last_lr()[0]
        self.scheduler.step()
        return self.optimizer.param_groups[0]["lr"]

    def _run_epoch(self, epoch: int) -> None:
        reset_peak_vram(self.device)
        train_loss = self.train_epoch(epoch)
        in_swa_phase = epoch >= self.config.swa_start_epoch
        learning_rate = self._advance_schedule(in_swa_phase)
        val_metrics = self.evaluate(self.model, self.val_loader, f"Epoch {epoch} [Val]", compute_hd95=False)

        self.run_logger.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                **prefixed("val", val_metrics),
                "learning_rate": learning_rate,
                "is_swa_active": float(in_swa_phase),
                **vram_metrics(self.device),
            }
        )
        self._save_if_best(val_metrics["dice"])

    def _finalize_swa(self) -> Optional[nn.Module]:
        """Recompute BatchNorm, save and validate the SWA model; ``None`` if no epoch was averaged."""
        if self.swa_model.n_averaged.item() == 0:
            return None
        logger.info("Recomputing SWA BatchNorm statistics...")
        self._recompute_swa_batchnorm()
        save_weights(self.swa_model, self.checkpoint_dir / SWA_CHECKPOINT)
        self.run_logger.log(prefixed("swa_val", self.evaluate(self.swa_model, self.val_loader, "SWA [Val]", compute_hd95=True)))
        return self.swa_model

    def _report_test_metrics(self, swa_model: Optional[nn.Module]) -> None:
        """Evaluate the best live and the SWA model once on every test set and store the results in the run summary."""
        load_weights(self.model, self.checkpoint_dir / BEST_LIVE_CHECKPOINT, self.device)
        candidates = {"test_best_live": self.model}
        if swa_model is not None:
            candidates["test_swa"] = swa_model

        for model_prefix, model in candidates.items():
            for set_name, test_set in self.splits.test_sets.items():
                prefix = qualified_test_prefix(model_prefix, set_name)
                test_metrics = self.evaluate(
                    model,
                    self.loader_factory(test_set.dataset, False),
                    f"{prefix} [Test]",
                    compute_hd95=True,
                    foreground_maps=test_set.foreground_maps,
                )
                results = prefixed(prefix, test_metrics)
                self.run_logger.log(results)
                self.run_logger.summarize(results)
                logger.info(" | ".join(f"{name}: {value:.4f}" for name, value in results.items()))

    def fit(self, run_name: str, total_params: int) -> None:
        """Run training, SWA finalization and test reporting; an interrupt saves the current weights."""
        self.run_logger.start(run_name, {**asdict(self.config), "total_params": total_params})
        logger.info("Starting training...")
        try:
            for epoch in range(1, self.config.epochs + 1):
                self._run_epoch(epoch)
            swa_model = self._finalize_swa()
            self._report_test_metrics(swa_model)
        except KeyboardInterrupt:
            logger.warning("Training interrupted. Saving current state...")
            save_weights(self.model, self.checkpoint_dir / INTERRUPTED_CHECKPOINT)
        finally:
            self.run_logger.finish()
            logger.info("Training finished.")


def parse_arguments() -> argparse.Namespace:
    available = ", ".join(sorted(discover_dataset_configs())) or "(none)"
    parser = argparse.ArgumentParser(description="Train a segmentation model")
    parser.add_argument("--model", type=str, default="hybrid", choices=MODEL_TYPES)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--bond_dim", type=int, default=32)
    add_mps_arguments(parser)
    parser.add_argument("--loss", type=str, default="focal_tversky", choices=LOSS_CHOICES)
    parser.add_argument("--fn_weight", type=float, default=TVERSKY_FN_WEIGHT, help="Focal Tversky weight on false negatives")
    parser.add_argument("--fp_weight", type=float, default=TVERSKY_FP_WEIGHT, help="Focal Tversky weight on false positives")
    parser.add_argument("--focal_gamma", type=float, default=FOCAL_TVERSKY_GAMMA, help="Focal Tversky gamma; the loss exponent is 1/gamma")
    parser.add_argument("--seed", type=int, default=42, help="Training seed; the data split seed stays fixed")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic GPU kernels for bit-identical reruns on the same GPU type (about 4x slower)",
    )
    parser.add_argument("--dataset", type=str, default="kvasir_seg", help=f"Comma-separated dataset configs. Available: {available}")
    parser.add_argument("--tag", type=str, default=None, help="Suffix to distinguish runs of the same config")
    return parser.parse_args()


def build_criterion(config: TrainingConfig) -> nn.Module:
    if config.loss == "bce_dice":
        return BCEDiceLoss(bce_weight=0.5)
    return FocalTverskyLoss(fn_weight=config.fn_weight, fp_weight=config.fp_weight, gamma=config.focal_gamma)


def log_parameter_breakdown(model: nn.Module) -> int:
    """Log the parameter count of every top-level module and return the total."""
    total_params = sum(parameter.numel() for parameter in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")
    for name, module in model.named_children():
        logger.info(f"  {name:25s} -> {sum(parameter.numel() for parameter in module.parameters()):>12,}")
    return total_params


def compile_with_warmup(model: nn.Module, config: TrainingConfig, device: torch.device) -> nn.Module:
    """Compile the model and trigger compilation with one forward/backward pass; falls back to eager mode.

    The warm-up gradients are cleared on success and on failure, so they never reach the first step.
    """
    try:
        compiled_model = torch.compile(model)
        # Warm up at the training shape, memory format and precision so the graph used for
        # training is compiled. Neither input nor output is bound: holding the output would
        # keep the whole activation graph alive for the rest of the run.
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            compiled_model(
                torch.randn(config.batch_size, 3, config.image_size, config.image_size, device=device).to(
                    memory_format=torch.channels_last
                )
            ).float().mean().backward()
        logger.info("Compiled the model successfully")
        return compiled_model
    except Exception as error:
        logger.warning(f"torch.compile failed, continuing in eager mode: {error}")
        return model
    finally:
        model.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()


def main() -> None:
    init()
    arguments = parse_arguments()
    dataset_names = parse_dataset_arg(arguments.dataset)
    seed_everything(arguments.seed, deterministic_algorithms=arguments.deterministic)
    device = get_device()
    splits = build_splits(dataset_names)
    dataset_slug = slugify_dataset_arg(dataset_names)
    variant = mps_variant(arguments)
    run_name = build_run_name(
        dataset_slug, arguments.model, arguments.bond_dim, arguments.loss, arguments.seed, arguments.tag, variant
    )

    config = TrainingConfig(
        model_type=arguments.model,
        dataset=dataset_slug,
        loss=arguments.loss,
        seed=arguments.seed,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        bond_dim=arguments.bond_dim,
        fn_weight=arguments.fn_weight,
        fp_weight=arguments.fp_weight,
        focal_gamma=arguments.focal_gamma,
        image_size=splits.image_size,
        mps_variant=variant,
        deterministic=arguments.deterministic,
        checkpoint_dir=str(CHECKPOINT_ROOT / run_name),
    )

    test_sizes = ", ".join(f"{len(test_set.dataset)} {name}" for name, test_set in splits.test_sets.items())
    logger.info(f"Dataset '{config.dataset}': {len(splits.train)} train / {len(splits.val.dataset)} val / test: {test_sizes}")

    model = build_segmentation_model(config.model_type, config.bond_dim, config.image_size, config.mps_variant)
    total_params = log_parameter_breakdown(model)
    model = model.to(device).to(memory_format=torch.channels_last)
    model = compile_with_warmup(model, config, device)

    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    trainer = SegmentationTrainer(
        model=model,
        splits=splits,
        loader_factory=lambda dataset, shuffle: build_loader(dataset, config.batch_size, shuffle, config.seed),
        criterion=build_criterion(config),
        optimizer=optimizer,
        scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs),
        device=device,
        config=config,
        run_logger=WandbLogger(),
    )
    trainer.fit(run_name=run_name, total_params=total_params)


if __name__ == "__main__":
    main()
