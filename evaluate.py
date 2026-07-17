import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset import load_kvasir_seg
from models.unet_classic import UNet

def calculate_metrics(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-5) -> tuple[float, float]:
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    
    intersection = (preds * targets).sum(dim=(2, 3))
    union = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    iou = (intersection + smooth) / (union - intersection + smooth)
    
    return dice.mean().item(), iou.mean().item()

def denormalize(tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(tensor.device)
    tensor = tensor * std + mean
    return torch.clamp(tensor, 0, 1).cpu().permute(1, 2, 0).numpy()

def save_visualizations(images: torch.Tensor, masks: torch.Tensor, logits: torch.Tensor, 
                        sample_ids: list[str], output_dir: str = "outputs"):
    os.makedirs(output_dir, exist_ok=True)
    preds = (torch.sigmoid(logits) > 0.5).float()
    
    for i in range(images.size(0)):
        img_np = denormalize(images[i])
        gt_np = masks[i].cpu().squeeze().numpy()
        pred_np = preds[i].cpu().squeeze().numpy()
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img_np)
        axes[0].set_title(f"Original RGB\n({sample_ids[i]})")
        axes[0].axis('off')
        
        axes[1].imshow(gt_np, cmap='gray')
        axes[1].set_title("Ground Truth Mask")
        axes[1].axis('off')
        
        axes[2].imshow(pred_np, cmap='gray')
        axes[2].set_title("Model Prediction")
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"pred_{sample_ids[i]}.png"), bbox_inches='tight', dpi=150)
        plt.close(fig)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")
    
    full_dataset = load_kvasir_seg("configs/kvasir_seg.yaml")
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    _, val_ds = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=4)
    
    model = UNet(in_channels=3, out_channels=1).to(device)
    
    checkpoint_path = "checkpoints/best_model.pth"
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)

        clean_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
            
        model.load_state_dict(clean_state_dict)
        print("Weights successfully loaded and adapted!")
    
    model.eval()
    total_dice, total_iou = 0.0, 0.0
    viz_saved = 0
    
    with torch.inference_mode():
        for batch in tqdm(val_loader, desc="Evaluating"):
            images, masks = batch["image"].to(device), batch["mask"].to(device)
            outputs = model(images)
            
            dice, iou = calculate_metrics(outputs, masks)
            total_dice += dice
            total_iou += iou
            
            if viz_saved < 2:
                save_visualizations(images, masks, outputs, batch["metadata"]["sample_id"])
                viz_saved += 1
                
    print(f"Avg Dice: {total_dice / len(val_loader):.4f} | Avg IoU: {total_iou / len(val_loader):.4f}")

if __name__ == "__main__":
    main()