import logging
import os
import random
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import wandb

from dataset import load_kvasir_seg
from models.unet_hybrid import UNetHybrid
from utils import get_device
from models.mps_layer import MPSBottleneck

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def seed_everything(seed: int = 42) -> None:
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    logger.info(f"Seed set to {seed}")

class BCEDiceLoss(nn.Module):
    """BCE + Dice Loss for highly imbalanced segmentation tasks."""
    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_score.mean()
        
        return bce_loss + dice_loss


class SegmentationTrainer:
    """Handles the training loop with AMP and gradient accumulation."""
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
        config: Dict[str, Any],
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
    ):
        self.model = model.to(device).to(memory_format=torch.channels_last)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        
        self.use_amp = device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)
        self.best_val_dice = 0.0 
        self.checkpoint_dir = config.get("checkpoint_dir", "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        epoch_loss = 0.0
        acc_steps = self.config.get("accumulation_steps", 1)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config['epochs']} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            images = batch["image"].to(self.device, non_blocking=True, memory_format=torch.channels_last)
            masks = batch["mask"].to(self.device, non_blocking=True)

            with torch.autocast(self.device.type, enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                loss = loss / acc_steps

            self.scaler.scale(loss).backward()

            if ((batch_idx + 1) % acc_steps == 0) or (batch_idx + 1 == len(self.train_loader)):
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            current_loss = loss.item() * acc_steps
            epoch_loss += current_loss
            pbar.set_postfix({'loss': f"{current_loss:.4f}"})

        return epoch_loss / len(self.train_loader)

    @torch.inference_mode()
    def validate_epoch(self, epoch: int) -> tuple[float, float]:
        self.model.eval()
        epoch_loss = 0.0
        epoch_dice = 0.0
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch}/{self.config['epochs']} [Val]")
        
        for batch in pbar:
            images = batch["image"].to(self.device, non_blocking=True, memory_format=torch.channels_last)
            masks = batch["mask"].to(self.device, non_blocking=True)

            with torch.autocast(self.device.type, enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)

            epoch_loss += loss.item()
            
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            intersection = (preds * masks).sum(dim=(2, 3))
            union = preds.sum(dim=(2, 3)) + masks.sum(dim=(2, 3))
            dice = (2.0 * intersection + 1e-5) / (union + 1e-5)
            epoch_dice += dice.mean().item()
            
            pbar.set_postfix({
                'val_loss': f"{loss.item():.4f}", 
                'val_dice': f"{dice.mean().item():.4f}"
            })

        return epoch_loss / len(self.val_loader), epoch_dice / len(self.val_loader)

    def save_checkpoint(self, val_dice: float, filename: str = "best_model.pth") -> None:
        """Save model if validation Dice score improves."""
        if val_dice > self.best_val_dice:
            self.best_val_dice = val_dice
            filepath = os.path.join(self.checkpoint_dir, filename)
            torch.save(self.model.state_dict(), filepath)
            logger.info(f"New best model saved! Val Dice: {val_dice:.4f}")

    def fit(self) -> None:
        """Execute full training and validation loop."""
        # W&B entegrasyonu tamamen duruyor
        total_params = sum(p.numel() for p in self.model.parameters())
        wandb.init(
            entity="hybrid-qtn-team",
            project="hybrid-qtn",
            config={**self.config, "total_params": total_params},
            name="r=32_channel_only_mps_run")
        
        logger.info("Starting training...")
        try:
            for epoch in range(1, self.config['epochs'] + 1):
                train_loss = self.train_epoch(epoch)
                val_loss, val_dice = self.validate_epoch(epoch)
                
                if self.scheduler:
                    self.scheduler.step()
                
                log_data = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_dice": val_dice,
                    "learning_rate": self.optimizer.param_groups[0]['lr'],
                }
                if self.device.type == "cuda":
                    log_data["vram_allocated_mb"] = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
                    log_data["vram_reserved_mb"] = torch.cuda.memory_reserved(self.device) / (1024 ** 2)
                    torch.cuda.reset_peak_memory_stats(self.device)
                wandb.log(log_data)

                self.save_checkpoint(val_dice)
                
        except KeyboardInterrupt:
            logger.warning("Training interrupted. Saving current state...")
            filepath = os.path.join(self.checkpoint_dir, "interrupted_model.pth")
            torch.save(self.model.state_dict(), filepath)
        finally:
            wandb.finish()
            logger.info("Training finished.")

def debug_model_info(model: nn.Module, device: torch.device, config: dict) -> None:
    logger.info("=" * 60)
    logger.info("MODEL DEBUG INFO")
    logger.info("=" * 60)

    logger.info(f"Device:               {device}")
    if device.type == "cuda":
        logger.info(f"GPU:                  {torch.cuda.get_device_name(device)}")
        vram = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
        logger.info(f"VRAM:                 {vram:.1f} GB")
        logger.info(f"CUDA version:         {torch.version.cuda}")
    logger.info(f"PyTorch version:      {torch.__version__}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters:     {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    logger.info("-" * 60)
    logger.info("Per-component parameter counts:")
    for name, module in model.named_children():
        count = sum(p.numel() for p in module.parameters())
        logger.info(f"  {name:25s} → {count:>12,}")

    if model.bottleneck_transform is not None:
        logger.info("-" * 60)
        logger.info("MPS Bottleneck breakdown:")
        bt = model.bottleneck_transform
        for name, param in bt.named_parameters():
            logger.info(f"  {name:25s} → shape {str(list(param.shape)):20s} = {param.numel():>10,}")

    logger.info("-" * 60)
    logger.info("Forward pass shape trace (dummy input):")
    model.to(device)
    dummy = torch.randn(1, 3, config["image_size"], config["image_size"]).to(device)

    with torch.no_grad():
        x1 = model.inc(dummy)
        logger.info(f"  inc  (encoder 0):  {list(dummy.shape)} → {list(x1.shape)}")
        x2 = model.down1(x1)
        logger.info(f"  down1 (encoder 1): {list(x1.shape)} → {list(x2.shape)}")
        x3 = model.down2(x2)
        logger.info(f"  down2 (encoder 2): {list(x2.shape)} → {list(x3.shape)}")
        x4 = model.down3(x3)
        logger.info(f"  down3 (encoder 3): {list(x3.shape)} → {list(x4.shape)}")
        x5 = model.down4(x4)
        logger.info(f"  down4 (bottleneck conv): {list(x4.shape)} → {list(x5.shape)}")

        if model.bottleneck_transform is not None:
            x5_mps = model.bottleneck_transform(x5)
            logger.info(f"  MPS transform:     {list(x5.shape)} → {list(x5_mps.shape)}")
            x5 = x5_mps

        x = model.up1(x5, x4)
        logger.info(f"  up1 (decoder 1):   {list(x5.shape)} → {list(x.shape)}")
        x = model.up2(x, x3)
        logger.info(f"  up2 (decoder 2):   → {list(x.shape)}")
        x = model.up3(x, x2)
        logger.info(f"  up3 (decoder 3):   → {list(x.shape)}")
        x = model.up4(x, x1)
        logger.info(f"  up4 (decoder 4):   → {list(x.shape)}")
        out = model.outc(x)
        logger.info(f"  outc (final):      → {list(out.shape)}")

    logger.info("=" * 60)


if __name__ == "__main__":
    seed_everything(42)
    device = get_device()

    config = {
        "epochs": 1,
        "batch_size": 4,
        "accumulation_steps": 8,
        "learning_rate": 2e-4,
        "weight_decay": 1e-4,
        "image_size": 512,
        "bond_dim": 32,
        "checkpoint_dir": "./checkpoints"
    }

    full_dataset = load_kvasir_seg("configs/kvasir_seg.yaml")
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_ds, val_ds = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, num_workers=4, pin_memory=True)

    # Model
    bottleneck = MPSBottleneck(channels=1024, bond_dim=config["bond_dim"])
    model = UNetHybrid(in_channels=3, out_channels=1, bottleneck_transform=bottleneck)

    debug_model_info(model, device, config)

    if int(torch.__version__.split('.')[0]) >= 2:
        try:
            model = torch.compile(model)
        except Exception:
            pass
    
    criterion = BCEDiceLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])

    trainer = SegmentationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        config=config,
        scheduler=scheduler
    )
    
    trainer.fit()