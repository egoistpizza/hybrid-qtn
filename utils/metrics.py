import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure

def calculate_mae(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return torch.abs(preds - targets).mean(dim=(1, 2, 3))

def calculate_f2_score(preds: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-5) -> torch.Tensor:
    tp = (preds * targets).sum(dim=(2, 3))
    fp = (preds * (1.0 - targets)).sum(dim=(2, 3))
    fn = ((1.0 - preds) * targets).sum(dim=(2, 3))
    
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    
    return (5.0 * precision * recall) / (4.0 * precision + recall + smooth)

def calculate_hd95(preds: torch.Tensor, targets: torch.Tensor) -> float:
    preds_np = preds.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    batch_size = preds_np.shape[0]
    hd95_scores = []
    
    struct = generate_binary_structure(2, 1)

    for i in range(batch_size):
        pred_mask = preds_np[i, 0].astype(bool)
        target_mask = targets_np[i, 0].astype(bool)

        if not np.any(pred_mask) and not np.any(target_mask):
            continue
            
        if not np.any(pred_mask) or not np.any(target_mask):
            max_dist = np.sqrt(pred_mask.shape[0]**2 + pred_mask.shape[1]**2)
            hd95_scores.append(max_dist)
            continue

        pred_border = pred_mask ^ binary_erosion(pred_mask, structure=struct)
        target_border = target_mask ^ binary_erosion(target_mask, structure=struct)

        pred_edt = distance_transform_edt(~pred_mask)
        target_edt = distance_transform_edt(~target_mask)

        dist_1 = target_edt[pred_border]
        dist_2 = pred_edt[target_border]

        if dist_1.size == 0 or dist_2.size == 0:
            continue

        hd95_scores.append(np.maximum(np.percentile(dist_1, 95), np.percentile(dist_2, 95)))

    return float(np.mean(hd95_scores)) if hd95_scores else 0.0

def calculate_all_metrics(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-5, compute_hd95: bool = False) -> dict:
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()

    intersection = (preds * targets).sum(dim=(2, 3))
    union = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))

    dice = (2.0 * intersection + smooth) / (union + smooth)
    iou = (intersection + smooth) / (union - intersection + smooth)
    
    mae = calculate_mae(preds, targets)
    f2 = calculate_f2_score(preds, targets, smooth)

    metrics = {
        "dice": dice.mean().item(),
        "iou": iou.mean().item(),
        "mae": mae.mean().item(),
        "f2": f2.mean().item(),
        "hd95": 0.0
    }

    if compute_hd95:
        metrics["hd95"] = calculate_hd95(preds, targets)

    return metrics
