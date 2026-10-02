import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure

METRIC_NAMES = ("dice", "iou", "mae", "f2", "hd95")
PREDICTION_THRESHOLD = 0.5
HD95_PERCENTILE = 95


def binarize(logits: torch.Tensor) -> torch.Tensor:
    return (torch.sigmoid(logits.float()) > PREDICTION_THRESHOLD).float()


def calculate_mae(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return torch.abs(preds - targets).mean(dim=(1, 2, 3))


def calculate_f2_score(preds: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-5) -> torch.Tensor:
    tp = (preds * targets).sum(dim=(2, 3))
    fp = (preds * (1.0 - targets)).sum(dim=(2, 3))
    fn = ((1.0 - preds) * targets).sum(dim=(2, 3))
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    return (5.0 * precision * recall) / (4.0 * precision + recall + smooth)


def _border(mask: np.ndarray, structure: np.ndarray) -> np.ndarray:
    return mask ^ binary_erosion(mask, structure=structure)


def hd95_per_image(preds: torch.Tensor, targets: torch.Tensor) -> list[float]:
    preds_np = preds.detach().cpu().numpy().astype(bool)
    targets_np = targets.detach().cpu().numpy().astype(bool)
    structure = generate_binary_structure(2, 1)
    scores: list[float] = []

    for pred_mask, target_mask in zip(preds_np[:, 0], targets_np[:, 0]):
        if not pred_mask.any() and not target_mask.any():
            continue
        if not pred_mask.any() or not target_mask.any():
            scores.append(float(np.hypot(*pred_mask.shape)))
            continue

        pred_border = _border(pred_mask, structure)
        target_border = _border(target_mask, structure)
        pred_to_target = distance_transform_edt(~target_border)[pred_border]
        target_to_pred = distance_transform_edt(~pred_border)[target_border]

        if pred_to_target.size == 0 or target_to_pred.size == 0:
            continue
        scores.append(
            float(max(np.percentile(pred_to_target, HD95_PERCENTILE), np.percentile(target_to_pred, HD95_PERCENTILE)))
        )
    return scores


def calculate_hd95(preds: torch.Tensor, targets: torch.Tensor) -> float:
    scores = hd95_per_image(preds, targets)
    return float(np.mean(scores)) if scores else 0.0


def per_image_overlap_metrics(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-5) -> dict[str, torch.Tensor]:
    preds = binarize(logits)
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return {
        "dice": (2.0 * intersection + smooth) / (union + smooth),
        "iou": (intersection + smooth) / (union - intersection + smooth),
        "mae": calculate_mae(preds, targets),
        "f2": calculate_f2_score(preds, targets, smooth).mean(dim=1),
    }


def calculate_all_metrics(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-5, compute_hd95: bool = False) -> dict:
    per_image = per_image_overlap_metrics(logits, targets, smooth)
    metrics = {name: values.mean().item() for name, values in per_image.items()}
    metrics["hd95"] = calculate_hd95(binarize(logits), targets) if compute_hd95 else 0.0
    return metrics


class MetricAccumulator:
    def __init__(self, compute_hd95: bool = False) -> None:
        self.compute_hd95 = compute_hd95
        self._overlap_totals = {name: 0.0 for name in METRIC_NAMES if name != "hd95"}
        self._hd95_scores: list[float] = []
        self._image_count = 0

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        for name, values in per_image_overlap_metrics(logits, targets).items():
            self._overlap_totals[name] += values.sum().item()
        self._image_count += logits.size(0)
        if self.compute_hd95:
            self._hd95_scores.extend(hd95_per_image(binarize(logits), targets))

    def compute(self) -> dict[str, float]:
        if self._image_count == 0:
            raise ValueError("MetricAccumulator.compute() called before any update()")
        metrics = {name: total / self._image_count for name, total in self._overlap_totals.items()}
        metrics["hd95"] = float(np.mean(self._hd95_scores)) if self._hd95_scores else 0.0
        return metrics
