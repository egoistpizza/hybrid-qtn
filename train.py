import argparse
import logging
import os
import random
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import wandb

from dataset import load_kvasir_seg
from utils import get_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    logger.info(f"Seed set to {seed}")

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, gamma: float = 0.75, smooth: float = 1e-5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        
        tp = (probs * targets).sum(dim=(2, 3))
        fp = (probs * (1.0 - targets)).sum(dim=(2, 3))
        fn = ((1.0 - probs) * targets).sum(dim=(2, 3))
        
        tversky_index = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        focal_tversky = (1.0 - tversky_index) ** self.gamma
        
        return focal_tversky.mean()

def get_vram_metrics() -> Dict[str, float]:
    if not torch.cuda.is_available():
        return {"vram_allocated_mb": 0.0, "vram_reserved_mb": 0.0}
        
    allocated_bytes = torch.cuda.memory_allocated()
    reserved_bytes = torch.cuda.memory_reserved()
    
    megabyte_conversion_factor = 1024 ** 2
    
    return {
        "vram_allocated_mb": allocated_bytes / megabyte_conversion_factor,
        "vram_reserved_mb": reserved_bytes / megabyte_conversion_factor
    }

def debug_model_info(model: nn.Module, device: torch.device, config: Dict[str, Any]) -> None:
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
    for name, module in model.named_children():
        count = sum(p.numel() for p in module.parameters())
        logger.info(f"  {name:25s} -> {count:>12,}")

    if hasattr(model, 'transform_512') and model.transform_512 is not None:
        logger.info("-" * 60)
        for name, param in model.transform_512.named_parameters():
            logger.info(f"  {name:25s} -> shape {str(list(param.shape)):20s} = {param.numel():>10,}")

    if hasattr(model, 'transform_1024') and model.transform_1024 is not None:
        logger.info("-" * 60)
        for name, param in model.transform_1024.named_parameters():
            logger.info(f"  {name:25s} -> shape {str(list(param.shape)):20s} = {param.numel():>10,}")

    logger.info("-" * 60)
    model.to(device)
    dummy = torch.randn(1, 3, config["image_size"], config["image_size"]).to(device)

    with torch.no_grad():
        x1 = model.inc(dummy)
        x2 = model.down1(x1)
        x3 = model.down2(x2)
        
        x4 = model.down3(x3)
        if hasattr(model, 'transform_512') and model.transform_512 is not None:
            x4 = model.transform_512(x4)
            
        x5 = model.down4(x4)
        if hasattr(model, 'transform_1024') and model.transform_1024 is not None:
            x5 = model.transform_1024(x5)

        x = model.up1(x5, x4)
        x = model.up2(x, x3)
        x = model.up3(x, x2)
        x = model.up4(x, x1)
        out = model.outc(x)

    logger.info("=" * 60)

class SegmentationTrainer:
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

        self.swa_start_epoch = int(self.config["epochs"] * self.config.get("swa_start_pct", 0.75))
        self.swa_model = AveragedModel(self.model)
        self.swa_scheduler = SWALR(self.optimizer, swa_lr=self.config.get("swa_lr", 5e-5))

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
    def validate_epoch(self, epoch: int, use_swa: bool = False) -> Tuple[float, float]:
        eval_model = self.swa_model if use_swa else self.model
        eval_model.eval()
        
        epoch_loss = 0.0
        epoch_dice = 0.0
        
        mode_str = "Val-SWA" if use_swa else "Val"
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch}/{self.config['epochs']} [{mode_str}]")
        
        for batch in pbar:
            images = batch["image"].to(self.device, non_blocking=True, memory_format=torch.channels_last)
            masks = batch["mask"].to(self.device, non_blocking=True)

            with torch.autocast(self.device.type, enabled=self.use_amp):
                outputs = eval_model(images)
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

    @torch.no_grad()
    def _update_swa_bn(self) -> None:
        self.swa_model.train()
        for module in self.swa_model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.reset_running_stats()
                module.momentum = None
        
        for batch in self.train_loader:
            images = batch["image"].to(self.device, non_blocking=True, memory_format=torch.channels_last)
            with torch.autocast(self.device.type, enabled=self.use_amp):
                self.swa_model(images)

    def save_checkpoint(self, val_dice: float, filename: str = "best_model.pth") -> None:
        if val_dice > self.best_val_dice:
            self.best_val_dice = val_dice
            filepath = os.path.join(self.checkpoint_dir, filename)
            torch.save(self.model.state_dict(), filepath)
            logger.info(f"New best model saved! Val Dice: {val_dice:.4f}")

    def fit(self, run_name: str) -> None:
        total_params = sum(p.numel() for p in self.model.parameters())
        wandb.init(
            project="hybrid-qtn",
            config={**self.config, "total_params": total_params},
            name=run_name
        )
        
        logger.info("Starting training...")
        try:
            for epoch in range(1, self.config['epochs'] + 1):
                train_loss = self.train_epoch(epoch)
                
                use_swa = epoch >= self.swa_start_epoch
                if use_swa:
                    self.swa_model.update_parameters(self.model)
                    self.swa_scheduler.step()
                elif self.scheduler:
                    self.scheduler.step()
                
                val_loss, val_dice = self.validate_epoch(epoch, use_swa=use_swa)
                
                log_data = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_dice": val_dice,
                    "learning_rate": self.optimizer.param_groups[0]['lr'] if not use_swa else self.swa_scheduler.get_last_lr()[0],
                    "is_swa_active": float(use_swa),
                    **get_vram_metrics()
                }
                
                wandb.log(log_data)
                
                if not use_swa:
                    self.save_checkpoint(val_dice)
                else:
                    filepath = os.path.join(self.checkpoint_dir, "swa_latest.pth")
                    torch.save(self.swa_model.state_dict(), filepath)

            if self.config['epochs'] >= self.swa_start_epoch:
                logger.info("Training complete. Updating SWA BatchNorm statistics...")
                self._update_swa_bn()
                final_val_loss, final_val_dice = self.validate_epoch(self.config['epochs'], use_swa=True)
                
                logger.info(f"Final SWA Model - Val Loss: {final_val_loss:.4f} | Val Dice: {final_val_dice:.4f}")
                wandb.log({"final_swa_val_loss": final_val_loss, "final_swa_val_dice": final_val_dice})
                
                filepath = os.path.join(self.checkpoint_dir, "best_swa_model.pth")
                torch.save(self.swa_model.state_dict(), filepath)
                
        except KeyboardInterrupt:
            logger.warning("Training interrupted. Saving current state...")
            filepath = os.path.join(self.checkpoint_dir, "interrupted_model.pth")
            torch.save(self.model.state_dict(), filepath)
        finally:
            wandb.finish()
            logger.info("Training finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Segmentation Model")
    parser.add_argument("--model", type=str, default="hybrid", choices=["vanilla", "hybrid", "deep_hybrid"])
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--bond_dim", type=int, default=32, help="Bond dimension for MPS layer")
    args = parser.parse_args()

    seed_everything(42)
    device = get_device()

    config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "accumulation_steps": 8,
        "learning_rate": 2e-4,
        "weight_decay": 1e-4,
        "image_size": 512,
        "bond_dim": args.bond_dim,
        "swa_start_pct": 0.75,
        "swa_lr": 5e-5,
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

    if args.model == "deep_hybrid":
        logger.info("Initializing DEEP Hybrid UNet with Dual MPS Bottlenecks...")
        from models.unet_hybrid import UNetDeepHybrid
        from models.mps_layer import MPSBottleneck
        
        mps_512 = MPSBottleneck(in_channels=512, bond_dim=config["bond_dim"], height=64, width=64)
        mps_1024 = MPSBottleneck(in_channels=1024, bond_dim=config["bond_dim"], height=32, width=32)
        
        model = UNetDeepHybrid(
            in_channels=3, 
            out_channels=1, 
            transform_512=mps_512, 
            transform_1024=mps_1024
        )
        run_name = f"deep_hybrid_unet_b{config['bond_dim']}_ckpt"
        debug_model_info(model, device, config)

    elif args.model == "hybrid":
        logger.info("Initializing Standard Hybrid UNet...")
        from models.unet_hybrid import UNetDeepHybrid
        from models.mps_layer import MPSBottleneck
        
        mps_1024 = MPSBottleneck(in_channels=1024, bond_dim=config["bond_dim"], height=32, width=32)
        model = UNetDeepHybrid(
            in_channels=3, 
            out_channels=1, 
            transform_512=None, 
            transform_1024=mps_1024
        )
        run_name = f"hybrid_unet_b{config['bond_dim']}_ckpt"
        debug_model_info(model, device, config) 
        
    else:
        logger.info("Initializing Vanilla UNet...")
        from models.unet_classic import UNet
        
        model = UNet(in_channels=3, out_channels=1)
        run_name = "vanilla_unet"

    config["model_type"] = args.model

    model = model.to(device).to(memory_format=torch.channels_last)
    if int(torch.__version__.split('.')[0]) >= 2:
        try:
            model = torch.compile(model)
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}")
    
    criterion = FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=0.75)
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
    
    trainer.fit(run_name=run_name)
