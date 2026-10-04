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


def test_focal_tversky_defaults_weight_false_positives():
    criterion = FocalTverskyLoss()
    assert (criterion.fn_weight, criterion.fp_weight) == (0.3, 0.7)
    assert criterion.gamma == pytest.approx(4.0 / 3.0)
    assert criterion.smooth == pytest.approx(1e-5)


def test_fp_weight_penalizes_false_positives_more_than_false_negatives():
    criterion = FocalTverskyLoss(smooth=0.0)
    false_negative_loss = criterion(*_single_pixel_error(false_negative=True))
    false_positive_loss = criterion(*_single_pixel_error(false_negative=False))
    assert false_positive_loss > false_negative_loss


def test_loss_equals_one_minus_tversky_to_the_inverse_gamma():
    criterion = FocalTverskyLoss(smooth=0.0)
    logits, targets = _single_pixel_error(false_negative=True)
    true_positives, false_negatives = 7.0, 1.0
    tversky = true_positives / (true_positives + 0.3 * false_negatives)
    assert criterion(logits, targets).item() == pytest.approx((1.0 - tversky) ** 0.75, rel=1e-4)


def test_tversky_index_is_computed_per_image():
    perfect_logits, perfect_targets = _single_pixel_error(false_negative=True)
    perfect_logits = torch.where(perfect_targets.bool(), CONFIDENT_LOGIT, -CONFIDENT_LOGIT)
    erroneous_logits, erroneous_targets = _single_pixel_error(false_negative=False)
    criterion = FocalTverskyLoss(smooth=0.0)
    batch_loss = criterion(torch.cat([perfect_logits, erroneous_logits]), torch.cat([perfect_targets, erroneous_targets]))
    expected = (criterion(perfect_logits, perfect_targets) + criterion(erroneous_logits, erroneous_targets)) / 2
    assert criterion.tversky_index(erroneous_logits, erroneous_targets).shape == (1, 1)
    assert batch_loss.item() == pytest.approx(expected.item(), rel=1e-6)


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
