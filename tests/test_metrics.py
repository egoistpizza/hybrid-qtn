import numpy as np
import pytest
import torch

from utils.metrics import MetricAccumulator, calculate_f2_score


def calculate_hd95(preds: torch.Tensor, targets: torch.Tensor) -> float:
    accumulator = MetricAccumulator(compute_hd95=True)
    accumulator.update(torch.where(preds > 0, 10.0, -10.0), targets)
    return accumulator.compute()["hd95"]

def test_f2_score_perfect_match():
    preds = torch.ones((2, 1, 64, 64))
    targets = torch.ones((2, 1, 64, 64))
    f2 = calculate_f2_score(preds, targets)
    assert torch.isclose(f2, torch.tensor(1.0), atol=1e-4).all()

def test_f2_score_empty_masks():
    preds = torch.zeros((2, 1, 64, 64))
    targets = torch.zeros((2, 1, 64, 64))
    f2 = calculate_f2_score(preds, targets)
    assert not torch.isnan(f2).any()

def test_hd95_empty_masks():
    preds = torch.zeros((2, 1, 64, 64))
    targets = torch.zeros((2, 1, 64, 64))
    hd95 = calculate_hd95(preds, targets)
    assert hd95 == 0.0

def test_hd95_no_prediction():
    preds = torch.zeros((2, 1, 64, 64))
    targets = torch.ones((2, 1, 64, 64))
    hd95 = calculate_hd95(preds, targets)
    max_dist = np.sqrt(64**2 + 64**2)
    assert np.isclose(hd95, max_dist, atol=1e-4)

def test_hd95_partial_overlap():
    preds = torch.zeros((1, 1, 64, 64))
    targets = torch.zeros((1, 1, 64, 64))
    
    targets[0, 0, :, :32] = 1
    
    preds[0, 0, :, :30] = 1
    
    hd95 = calculate_hd95(preds, targets)
    
    assert np.isclose(hd95, 2.0, atol=1e-2)

def test_hd95_mutual_offset_no_nesting():
    preds = torch.zeros((1, 1, 64, 64))
    targets = torch.zeros((1, 1, 64, 64))

    targets[0, 0, :, 10:40] = 1
    preds[0, 0, :, 20:50] = 1

    hd95 = calculate_hd95(preds, targets)
    assert np.isclose(hd95, 10.0, atol=1.0)



def test_accumulator_averages_per_image_not_per_batch():
    perfect = torch.full((1, 1, 8, 8), 10.0)
    wrong = torch.full((1, 1, 8, 8), -10.0)
    targets = torch.ones((1, 1, 8, 8))
    accumulator = MetricAccumulator()
    accumulator.update(torch.cat([perfect, perfect, perfect]), targets.repeat(3, 1, 1, 1))
    accumulator.update(wrong, targets)
    assert accumulator.compute()["dice"] == pytest.approx(0.75, abs=1e-4)


def test_accumulator_is_independent_of_batching():
    logits = torch.randn(4, 1, 16, 16)
    targets = (torch.rand(4, 1, 16, 16) > 0.5).float()
    whole, split = MetricAccumulator(compute_hd95=True), MetricAccumulator(compute_hd95=True)
    whole.update(logits, targets)
    split.update(logits[:1], targets[:1])
    split.update(logits[1:], targets[1:])
    assert split.compute() == pytest.approx(whole.compute(), rel=1e-5)


def test_hd95_is_omitted_unless_computed():
    accumulator = MetricAccumulator()
    accumulator.update(torch.randn(2, 1, 8, 8), torch.ones(2, 1, 8, 8))
    assert set(accumulator.compute()) == {"dice", "iou", "mae", "f2"}


def test_accumulator_requires_updates():
    with pytest.raises(ValueError, match="before any update"):
        MetricAccumulator().compute()


def test_perfect_prediction_scores_one():
    accumulator = MetricAccumulator()
    accumulator.update(torch.full((1, 1, 2, 2), 10.0), torch.ones(1, 1, 2, 2))
    scores = accumulator.compute()
    assert scores["dice"] == pytest.approx(1.0, abs=1e-3)
    assert scores["iou"] == pytest.approx(1.0, abs=1e-3)


def test_partial_overlap_ranks_dice_above_iou():
    accumulator = MetricAccumulator()
    accumulator.update(torch.tensor([[[[10.0, -10.0], [10.0, -10.0]]]]), torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]]))
    scores = accumulator.compute()
    assert 0.0 < scores["iou"] < scores["dice"] < 1.0
