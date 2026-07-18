import logging
import os
import random
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import wandb

from utils.seed import seed_everything
from utils.init_first import init

from dataset import load_kvasir_seg
from models.unet_classic import UNet

# Outside of the main function (where init_first is called which calls init_logger_basicconfig) --- calling
# init_logger_basicconfig explicitly
from utils.init_first import init_logger_basicconfig
init_logger_basicconfig()
logger = logging.getLogger(__name__)



# BCE (Binary Cross Entropy) Loss for pixelwise comparison + Dice Loss for imbalanced classes
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
        
        self.scaler = GradScaler()
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

            with autocast():
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

            with autocast():
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
        # TODO: Logger pollutes the STDOUT (with an unicode encode error and its stack trace) due to wandb probably
        #       trying to print a unicode character to its own logger instance (when seleted [3: don't visualize])
        #       so suppress that message for now.
        #       That happens when the directory path contains non-ASCII characters including the uppercase "İ" (as in "İnzva")
        wandb.init(project="hybrid-qtn", config=self.config)
        
        logger.info("Starting training...")
        try:
            for epoch in range(1, self.config['epochs'] + 1):
                train_loss = self.train_epoch(epoch)
                val_loss, val_dice = self.validate_epoch(epoch)
                
                if self.scheduler:
                    self.scheduler.step()
                
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_dice": val_dice,
                    "learning_rate": self.optimizer.param_groups[0]['lr']
                })

                self.save_checkpoint(val_dice)
                
        except KeyboardInterrupt:
            logger.warning("Training interrupted. Saving current state...")
            filepath = os.path.join(self.checkpoint_dir, "interrupted_model.pth")
            torch.save(self.model.state_dict(), filepath)
        finally:
            wandb.finish()
            logger.info("Training finished.")


if __name__ == "__main__":
    init()
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # TODO: Maybe get this from a YAML file in /configs
    
    config = {
        "epochs": 100,
        "batch_size": 4,           
        "accumulation_steps": 8,   
        "learning_rate": 2e-4,
        "weight_decay": 1e-4,
        "image_size": 512,
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
    model = UNet(in_channels=3, out_channels=1) 
    
    if int(torch.__version__.split('.')[0]) >= 2:
        try:
            c_model = torch.compile(model)
            # torch.compile is lazy, it will wrap and successfully get out of the try-catch, and will try to compile only
            # when run something forward/backward.
            # Try actually attempting to run anything through it just to see if it compiles successfully.
            dummy_input = torch.randn(1, 3, config["image_size"], config["image_size"]).to(device)
            _ = c_model(dummy_input)
            # Replace it with the compiled only after passing the test
            model = c_model
        except Exception as e:
            logger.warn("[!] torch.compile(model)(input) failed")
            # logger.warn(e)
    
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