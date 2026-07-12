import os
import random
import logging
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

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
        
        self.scaler = GradScaler()
        self.best_val_dice = 0.0 
        self.checkpoint_dir = config.get("checkpoint_dir", "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        epoch_loss = 0.0
        acc_steps = self.config.get("accumulation_steps", 1)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config['epochs']} [Train]")
        
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True, memory_format=torch.channels_last)
            masks = masks.to(self.device, non_blocking=True)

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
        
        for images, masks in pbar:
            images = images.to(self.device, non_blocking=True, memory_format=torch.channels_last)
            masks = masks.to(self.device, non_blocking=True)

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
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config = {
        "epochs": 100,
        "batch_size": 4,           
        "accumulation_steps": 8,   
        "learning_rate": 2e-4,
        "weight_decay": 1e-4,
        "image_size": 512,
        "checkpoint_dir": "./checkpoints"
    }

    # Placeholders for integration
    model = nn.Identity() 
    train_loader = []     
    val_loader = []       

    if int(torch.__version__.split('.')[0]) >= 2:
        logger.info("PyTorch >= 2.0 detected. Compiling model...")
        model = torch.compile(model)
    
    criterion = BCEDiceLoss()
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config["learning_rate"], 
        weight_decay=config["weight_decay"]
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"]
    )

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
    
    # trainer.fit()