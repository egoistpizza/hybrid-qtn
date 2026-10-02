import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pytest
import torch

from utils.metrics import MetricAccumulator, calculate_all_metrics, calculate_f2_score, calculate_hd95

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


def test_accumulator_matches_single_batch_metrics():
    logits = torch.randn(4, 1, 16, 16)
    targets = (torch.rand(4, 1, 16, 16) > 0.5).float()
    accumulator = MetricAccumulator(compute_hd95=True)
    accumulator.update(logits, targets)
    expected = calculate_all_metrics(logits, targets, compute_hd95=True)
    assert accumulator.compute() == pytest.approx(expected, rel=1e-5)


def test_accumulator_requires_updates():
    with pytest.raises(ValueError, match="before any update"):
        MetricAccumulator().compute()
