import torch
import torch.nn as nn

TVERSKY_FN_WEIGHT = 0.3
TVERSKY_FP_WEIGHT = 0.7
FOCAL_TVERSKY_GAMMA = 4.0 / 3.0
FOCAL_TVERSKY_SMOOTH = 1e-5
SPATIAL_DIMENSIONS = (2, 3)


class FocalTverskyLoss(nn.Module):
    def __init__(
        self,
        fn_weight: float = TVERSKY_FN_WEIGHT,
        fp_weight: float = TVERSKY_FP_WEIGHT,
        gamma: float = FOCAL_TVERSKY_GAMMA,
        smooth: float = FOCAL_TVERSKY_SMOOTH,
    ):
        super().__init__()
        if gamma <= 0.0:
            raise ValueError(f"gamma must be positive, got {gamma}")
        self.fn_weight = fn_weight
        self.fp_weight = fp_weight
        self.gamma = gamma
        self.smooth = smooth

    def tversky_index(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probabilities = torch.sigmoid(logits.float())
        targets = targets.float()
        true_positives = (probabilities * targets).sum(dim=SPATIAL_DIMENSIONS)
        false_negatives = ((1.0 - probabilities) * targets).sum(dim=SPATIAL_DIMENSIONS)
        false_positives = (probabilities * (1.0 - targets)).sum(dim=SPATIAL_DIMENSIONS)
        return (true_positives + self.smooth) / (
            true_positives + self.fn_weight * false_negatives + self.fp_weight * false_positives + self.smooth
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return ((1.0 - self.tversky_index(logits, targets)) ** (1.0 / self.gamma)).mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, smooth: float = 1e-5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = 1.0 - bce_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        probabilities = torch.sigmoid(logits)
        intersection = (probabilities * targets).sum(dim=(2, 3))
        union = probabilities.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice_loss = 1.0 - ((2.0 * intersection + self.smooth) / (union + self.smooth))
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss.mean()
