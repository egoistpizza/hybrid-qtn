import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset import (
    build_train_val,
    discover_dataset_configs,
    parse_dataset_arg,
    slugify_dataset_arg,
)
from utils import get_device

IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


def calculate_metrics(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-5) -> tuple[float, float]:
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    
    intersection = (preds * targets).sum(dim=(2, 3))
    union = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    iou = (intersection + smooth) / (union - intersection + smooth)
    
    return dice.mean().item(), iou.mean().item()


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    device = tensor.device
    mean = torch.tensor(IMAGE_MEAN, device=device).view(3, 1, 1)
    std = torch.tensor(IMAGE_STD, device=device).view(3, 1, 1)
    
    tensor = tensor * std + mean
    return torch.clamp(tensor, 0.0, 1.0).cpu().permute(1, 2, 0).numpy()


def save_visualizations(
    images: torch.Tensor, 
    masks: torch.Tensor, 
    logits: torch.Tensor, 
    sample_ids: list[str], 
    output_dir: Path, 
    start_index: int = 0
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    preds = (torch.sigmoid(logits) > 0.5).float()
    
    for i in range(images.size(0)):
        img_np = denormalize(images[i])
        gt_np = masks[i].cpu().squeeze().numpy()
        pred_np = preds[i].cpu().squeeze().numpy()

        dice, iou = calculate_metrics(logits[i:i+1], masks[i:i+1])
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img_np)
        axes[0].set_title(f"Original RGB\n({sample_ids[i]})")
        axes[0].axis("off")
        
        axes[1].imshow(gt_np, cmap="gray")
        axes[1].set_title("Ground Truth Mask")
        axes[1].axis("off")
        
        axes[2].imshow(pred_np, cmap="gray")
        axes[2].set_title(f"Prediction\nDice: {dice:.4f} | IoU: {iou:.4f}")
        axes[2].axis("off")
        
        plt.tight_layout()
        filename = f"viz_{start_index + i + 1:03d}_{sample_ids[i]}.png"
        plt.savefig(output_dir / filename, bbox_inches="tight", dpi=300)
        plt.close(fig)


def load_model_weights(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint required but not found at: {checkpoint_path}")
        
    print(f"Loading weights from: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    clean_state_dict = {}
    for key, value in state_dict.items():
        if "n_averaged" in key:
            continue
            
        clean_key = key.replace("_orig_mod.", "").replace("module.", "")
        clean_state_dict[clean_key] = value
        
    model.load_state_dict(clean_state_dict)
    print("Weights loaded successfully!")


def build_model(model_type: str, bond_dim: int) -> torch.nn.Module:
    if model_type == "deep_hybrid":
        from models.unet_hybrid import UNetDeepHybrid
        from models.mps_layer import MPSBottleneck
        
        mps_512 = MPSBottleneck(in_channels=512, bond_dim=bond_dim, height=64, width=64)
        mps_1024 = MPSBottleneck(in_channels=1024, bond_dim=bond_dim, height=32, width=32)
        return UNetDeepHybrid(in_channels=3, out_channels=1, transform_512=mps_512, transform_1024=mps_1024)
        
    if model_type == "hybrid":
        from models.unet_hybrid import UNetDeepHybrid
        from models.mps_layer import MPSBottleneck
        
        mps_1024 = MPSBottleneck(in_channels=1024, bond_dim=bond_dim, height=32, width=32)
        return UNetDeepHybrid(in_channels=3, out_channels=1, transform_512=None, transform_1024=mps_1024)
        
    from models.unet_classic import UNet
    return UNet(in_channels=3, out_channels=1)


def parse_args() -> argparse.Namespace:
    available_datasets = ", ".join(sorted(discover_dataset_configs())) or "(none)"
    
    parser = argparse.ArgumentParser(description="Evaluate Segmentation Models")
    parser.add_argument("--model", type=str, default="vanilla", choices=["vanilla", "hybrid", "deep_hybrid"])
    parser.add_argument("--bond_dim", type=int, default=32)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="kvasir_seg", help=f"Available: {available_datasets}")
    parser.add_argument("--num_samples", type=int, default=10, choices=range(5, 11))
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--run", type=str, default=None)
    
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    device = get_device()
    use_amp = device.type == "cuda"
    
    print(f"Evaluating on device: {device} | AMP Enabled: {use_amp}")

    # full_dataset = load_kvasir_seg("configs/kvasir_seg.yaml")
    # full_dataset = load_kvasir_seg(os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs/kvasir_seg.yaml"))
    
    dataset_names = parse_dataset_arg(args.dataset)
    dataset_slug = slugify_dataset_arg(dataset_names)

    _, val_ds = build_train_val(dataset_names)
    print(f"Dataset '{dataset_slug}': {len(val_ds)} validation samples")
    
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

    viz_generator = torch.Generator().manual_seed(123)
    viz_indices = torch.randperm(len(val_ds), generator=viz_generator)[:args.num_samples]
    viz_subset = Subset(val_ds, viz_indices.tolist())
    viz_loader = DataLoader(viz_subset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    
    model = build_model(args.model, args.bond_dim)
    model = model.to(device).to(memory_format=torch.channels_last)
    
    run_suffix = f"{args.model}_unet" if args.model == "vanilla" else f"{args.model}_unet_b{args.bond_dim}_ckpt"
    run_name = args.run if args.run else f"{dataset_slug}__{run_suffix}"
    if args.tag:
        run_name = f"{run_name}__{args.tag}"
        
    checkpoint_dir = Path("checkpoints") / run_name
    swa_checkpoint_path = checkpoint_dir / "best_swa_model.pth"
    standard_checkpoint_path = checkpoint_dir / "best_model.pth"
    
    load_path = swa_checkpoint_path if swa_checkpoint_path.exists() else standard_checkpoint_path
    load_model_weights(model, load_path, device)
    
    model.eval()
    total_dice, total_iou = 0.0, 0.0
    
    with torch.inference_mode():
        for batch in tqdm(val_loader, desc="Evaluating"):
            images = batch["image"].to(device, non_blocking=True, memory_format=torch.channels_last)
            masks = batch["mask"].to(device, non_blocking=True)
            
            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images)
            
            dice, iou = calculate_metrics(outputs, masks)
            total_dice += dice
            total_iou += iou
            
    avg_dice = total_dice / len(val_loader)
    avg_iou = total_iou / len(val_loader)
    print(f"Avg Dice: {avg_dice:.4f} | Avg IoU: {avg_iou:.4f}")

    output_dir = Path(args.output_dir) if args.output_dir else Path("outputs") / "visualizations" / run_name
    print(f"\nGenerating {args.num_samples} visualizations...")
    
    viz_count = 0
    with torch.inference_mode():
        for batch in tqdm(viz_loader, desc="Saving visualizations"):
            images = batch["image"].to(device, non_blocking=True, memory_format=torch.channels_last)
            masks = batch["mask"].to(device, non_blocking=True)
            
            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images)
                
            save_visualizations(images, masks, outputs, batch["metadata"]["sample_id"], output_dir, viz_count)
            viz_count += images.size(0)
            
    print(f"Visualizations saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
