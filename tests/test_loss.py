import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.loss import BCEDiceLoss, FocalTverskyLoss

CONFIDENT_LOGIT = 20.0


@pytest.fixture
def setup_tensors():
    logits = torch.randn(2, 1, 64, 64, requires_grad=True)
    targets = torch.randint(0, 2, (2, 1, 64, 64)).float()
    return logits, targets


def _single_pixel_error(false_negative: bool) -> tuple[torch.Tensor, torch.Tensor]:
    targets = torch.zeros(1, 1, 4, 4)
    targets[..., :2, :] = 1.0
    logits = torch.where(targets.bool(), CONFIDENT_LOGIT, -CONFIDENT_LOGIT)
    if false_negative:
        logits[0, 0, 0, 0] = -CONFIDENT_LOGIT
    else:
        logits[0, 0, 3, 3] = CONFIDENT_LOGIT
    return logits, targets


def test_focal_tversky_loss(setup_tensors):
    logits, targets = setup_tensors
    loss = FocalTverskyLoss()(logits, targets)
    assert not torch.isnan(loss)
    assert loss.item() >= 0
    loss.backward()
    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any()


def test_focal_tversky_defaults_follow_abraham_and_khan():
    criterion = FocalTverskyLoss()
    assert (criterion.alpha, criterion.beta) == (0.7, 0.3)
    assert criterion.gamma == pytest.approx(4.0 / 3.0)


def test_alpha_penalizes_false_negatives_more_than_false_positives():
    criterion = FocalTverskyLoss(smooth=0.0)
    false_negative_loss = criterion(*_single_pixel_error(false_negative=True))
    false_positive_loss = criterion(*_single_pixel_error(false_negative=False))
    assert false_negative_loss > false_positive_loss


def test_loss_equals_one_minus_tversky_to_the_inverse_gamma():
    criterion = FocalTverskyLoss(smooth=0.0)
    logits, targets = _single_pixel_error(false_negative=True)
    true_positives, false_negatives = 7.0, 1.0
    tversky = true_positives / (true_positives + 0.7 * false_negatives)
    assert criterion(logits, targets).item() == pytest.approx((1.0 - tversky) ** 0.75, rel=1e-4)


def test_perfect_prediction_yields_zero_loss():
    logits, targets = _single_pixel_error(false_negative=True)
    logits = torch.where(targets.bool(), CONFIDENT_LOGIT, -CONFIDENT_LOGIT)
    assert FocalTverskyLoss(smooth=0.0)(logits, targets).item() == pytest.approx(0.0, abs=1e-6)


def test_non_positive_gamma_is_rejected():
    with pytest.raises(ValueError, match="gamma must be positive"):
        FocalTverskyLoss(gamma=0.0)


def test_bce_dice_loss(setup_tensors):
    logits, targets = setup_tensors
    loss = BCEDiceLoss(bce_weight=0.5)(logits, targets)
    assert not torch.isnan(loss)
    assert loss.item() >= 0
    loss.backward()
    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
