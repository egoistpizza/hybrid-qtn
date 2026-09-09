import os
import sys
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.loss import FocalTverskyLoss, BCEDiceLoss

@pytest.fixture
def setup_tensors():
    logits = torch.randn(2, 1, 64, 64, requires_grad=True)
    targets = torch.randint(0, 2, (2, 1, 64, 64)).float()
    return logits, targets

def test_focal_tversky_loss(setup_tensors):
    logits, targets = setup_tensors
    criterion = FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=0.75)
    
    loss = criterion(logits, targets)
    
    assert not torch.isnan(loss)
    assert loss.item() >= 0
    
    loss.backward()
    
    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any()

def test_bce_dice_loss(setup_tensors):
    logits, targets = setup_tensors
    criterion = BCEDiceLoss(bce_weight=0.5)
    
    loss = criterion(logits, targets)
    
    assert not torch.isnan(loss)
    assert loss.item() >= 0
    
    loss.backward()
    
    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any()

if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
