import argparse
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.amp import GradScaler
from torch.optim.swa_utils import SWALR, AveragedModel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dataset import (
    build_splits,
    discover_dataset_configs,
    parse_dataset_arg,
    seed_dataloader_worker,
    slugify_dataset_arg,
)
from models.factory import MODEL_TYPES, build_segmentation_model
from utils import get_device
from utils.hardware import reset_peak_vram, vram_metrics
from utils.init_first import init
from utils.loss import (
    FOCAL_TVERSKY_GAMMA,
    TVERSKY_FN_WEIGHT,
    TVERSKY_FP_WEIGHT,
    BCEDiceLoss,
    FocalTverskyLoss,
)
from utils.metrics import MetricAccumulator
from utils.run_naming import build_run_name
from utils.seed import seed_everything, seeded_generator

logger = logging.getLogger(__name__)

LOSS_CHOICES = ("focal_tversky", "bce_dice")
IMAGE_SIZE = 512
DATALOADER_WORKERS = 4
GRADIENT_CLIP_NORM = 1.0
BEST_LIVE_CHECKPOINT = "best_model.pth"
SWA_CHECKPOINT = "best_swa_model.pth"
INTERRUPTED_CHECKPOINT = "interrupted_model.pth"


@dataclass
class TrainingConfig:
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
    image_size: int = IMAGE_SIZE
    accumulation_steps: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    swa_start_pct: float = 0.75
    swa_lr: float = 5e-5
    checkpoint_dir: str = "checkpoints"

    @property
    def swa_start_epoch(self) -> int:
        return int(self.epochs * self.swa_start_pct)


def prefixed(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{name}": value for name, value in metrics.items()}


class SegmentationTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler.LRScheduler,
        device: torch.device,
        config: TrainingConfig,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config

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

    @torch.inference_mode()
    def evaluate(self, model: nn.Module, loader: DataLoader, description: str, compute_hd95: bool) -> tuple[float, dict[str, float]]:
        model.eval()
        accumulator = MetricAccumulator(compute_hd95=compute_hd95)
        weighted_loss, image_count = 0.0, 0

        for batch in tqdm(loader, desc=description):
            masks = self._masks(batch)
            with torch.autocast(self.device.type, enabled=self.use_amp):
                logits = model(self._images(batch))
                loss = self.criterion(logits, masks)
            accumulator.update(logits, masks)
            weighted_loss += loss.item() * masks.size(0)
            image_count += masks.size(0)

        return weighted_loss / image_count, accumulator.compute()

    @torch.no_grad()
    def _recompute_swa_batchnorm(self) -> None:
        self.swa_model.train()
        for module in self.swa_model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.reset_running_stats()
                module.momentum = None
        for batch in self.train_loader:
            with torch.autocast(self.device.type, enabled=self.use_amp):
                self.swa_model(self._images(batch))

    def _save(self, module: nn.Module, filename: str) -> Path:
        path = self.checkpoint_dir / filename
        torch.save(module.state_dict(), path)
        return path

    def _save_if_best(self, val_dice: float) -> None:
        if val_dice > self.best_val_dice:
            self.best_val_dice = val_dice
            self._save(self.model, BEST_LIVE_CHECKPOINT)
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
        val_loss, val_metrics = self.evaluate(self.model, self.val_loader, f"Epoch {epoch} [Val]", compute_hd95=False)

        wandb.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                **prefixed("val", val_metrics),
                "learning_rate": learning_rate,
                "is_swa_active": float(in_swa_phase),
                **vram_metrics(self.device),
            }
        )
        self._save_if_best(val_metrics["dice"])

    def _finalize_swa(self) -> Optional[nn.Module]:
        if self.swa_model.n_averaged.item() == 0:
            return None
        logger.info("Recomputing SWA BatchNorm statistics...")
        self._recompute_swa_batchnorm()
        self._save(self.swa_model, SWA_CHECKPOINT)
        swa_val_loss, swa_val_metrics = self.evaluate(self.swa_model, self.val_loader, "SWA [Val]", compute_hd95=True)
        wandb.log({"swa_val_loss": swa_val_loss, **prefixed("swa_val", swa_val_metrics)})
        return self.swa_model

    def _report_test_metrics(self, swa_model: Optional[nn.Module]) -> None:
        self.model.load_state_dict(torch.load(self.checkpoint_dir / BEST_LIVE_CHECKPOINT, map_location=self.device))
        candidates = {"test_best_live": self.model}
        if swa_model is not None:
            candidates["test_swa"] = swa_model

        for prefix, model in candidates.items():
            test_loss, test_metrics = self.evaluate(model, self.test_loader, f"{prefix} [Test]", compute_hd95=True)
            results = {f"{prefix}_loss": test_loss, **prefixed(prefix, test_metrics)}
            wandb.log(results)
            wandb.run.summary.update(results)
            logger.info(" | ".join(f"{name}: {value:.4f}" for name, value in results.items()))

    def fit(self, run_name: str, total_params: int) -> None:
        wandb.init(project="hybrid-qtn", config={**asdict(self.config), "total_params": total_params}, name=run_name)
        logger.info("Starting training...")
        try:
            for epoch in range(1, self.config.epochs + 1):
                self._run_epoch(epoch)
            swa_model = self._finalize_swa()
            self._report_test_metrics(swa_model)
        except KeyboardInterrupt:
            logger.warning("Training interrupted. Saving current state...")
            self._save(self.model, INTERRUPTED_CHECKPOINT)
        finally:
            wandb.finish()
            logger.info("Training finished.")


def parse_arguments() -> argparse.Namespace:
    available = ", ".join(sorted(discover_dataset_configs())) or "(none)"
    parser = argparse.ArgumentParser(description="Train a segmentation model")
    parser.add_argument("--model", type=str, default="hybrid", choices=MODEL_TYPES)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--bond_dim", type=int, default=32)
    parser.add_argument("--loss", type=str, default="focal_tversky", choices=LOSS_CHOICES)
    parser.add_argument("--fn_weight", type=float, default=TVERSKY_FN_WEIGHT, help="Focal Tversky weight on false negatives")
    parser.add_argument("--fp_weight", type=float, default=TVERSKY_FP_WEIGHT, help="Focal Tversky weight on false positives")
    parser.add_argument("--focal_gamma", type=float, default=FOCAL_TVERSKY_GAMMA, help="Focal Tversky gamma; the loss exponent is 1/gamma")
    parser.add_argument("--seed", type=int, default=42, help="Training seed; the data split seed stays fixed")
    parser.add_argument("--dataset", type=str, default="kvasir_seg", help=f"Comma-separated dataset configs. Available: {available}")
    parser.add_argument("--tag", type=str, default=None, help="Suffix to distinguish runs of the same config")
    return parser.parse_args()


def build_criterion(config: TrainingConfig) -> nn.Module:
    if config.loss == "bce_dice":
        return BCEDiceLoss(bce_weight=0.5)
    return FocalTverskyLoss(fn_weight=config.fn_weight, fp_weight=config.fp_weight, gamma=config.focal_gamma)


def build_loader(dataset: Dataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=DATALOADER_WORKERS,
        pin_memory=True,
        worker_init_fn=seed_dataloader_worker,
        generator=seeded_generator(seed),
    )


def log_parameter_breakdown(model: nn.Module) -> int:
    total_params = sum(parameter.numel() for parameter in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")
    for name, module in model.named_children():
        logger.info(f"  {name:25s} -> {sum(parameter.numel() for parameter in module.parameters()):>12,}")
    return total_params


def compile_with_warmup(model: nn.Module, config: TrainingConfig, device: torch.device) -> nn.Module:
    try:
        compiled_model = torch.compile(model)
        warmup_input = torch.randn(config.batch_size, 3, config.image_size, config.image_size, device=device)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            compiled_model(warmup_input.to(memory_format=torch.channels_last)).float().mean().backward()
        model.zero_grad(set_to_none=True)
        del warmup_input
        if device.type == "cuda":
            torch.cuda.empty_cache()
        logger.info("Compiled the model successfully")
        return compiled_model
    except Exception as error:
        logger.warning(f"torch.compile failed, continuing in eager mode: {error}")
        return model


def main() -> None:
    init()
    arguments = parse_arguments()
    dataset_names = parse_dataset_arg(arguments.dataset)
    seed_everything(arguments.seed)
    device = get_device()

    config = TrainingConfig(
        model_type=arguments.model,
        dataset=slugify_dataset_arg(dataset_names),
        loss=arguments.loss,
        seed=arguments.seed,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        bond_dim=arguments.bond_dim,
        fn_weight=arguments.fn_weight,
        fp_weight=arguments.fp_weight,
        focal_gamma=arguments.focal_gamma,
    )
    run_name = build_run_name(config.dataset, config.model_type, config.bond_dim, config.loss, config.seed, arguments.tag)
    config.checkpoint_dir = os.path.join("checkpoints", run_name)

    splits = build_splits(dataset_names)
    logger.info(f"Dataset '{config.dataset}': {len(splits.train)} train / {len(splits.val)} val / {len(splits.test)} test")

    model = build_segmentation_model(config.model_type, config.bond_dim, config.image_size)
    total_params = log_parameter_breakdown(model)
    model = model.to(device).to(memory_format=torch.channels_last)
    model = compile_with_warmup(model, config, device)

    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    trainer = SegmentationTrainer(
        model=model,
        train_loader=build_loader(splits.train, config.batch_size, shuffle=True, seed=config.seed),
        val_loader=build_loader(splits.val, config.batch_size, shuffle=False, seed=config.seed),
        test_loader=build_loader(splits.test, config.batch_size, shuffle=False, seed=config.seed),
        criterion=build_criterion(config),
        optimizer=optimizer,
        scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs),
        device=device,
        config=config,
    )
    trainer.fit(run_name=run_name, total_params=total_params)


if __name__ == "__main__":
    main()
