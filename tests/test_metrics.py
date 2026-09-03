import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pytest
import torch

from utils.metrics import calculate_f2_score, calculate_hd95

def test_f2_score_perfect_match():
    preds = torch.ones((2, 1, 64, 64))
    targets = torch.ones((2, 1, 64, 64))
    f2 = calculate_f2_score(preds, targets)
    # Use .all() to assert every item in the batch tensor is close to 1.0
    assert torch.isclose(f2, torch.tensor(1.0), atol=1e-4).all()

def test_f2_score_empty_masks():
    preds = torch.zeros((2, 1, 64, 64))
    targets = torch.zeros((2, 1, 64, 64))
    f2 = calculate_f2_score(preds, targets)
    # Use .any() to assert that absolutely no item in the batch tensor is NaN
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
