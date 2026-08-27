import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from train import FocalTverskyLoss

def test_focal_tversky_loss():
    logits = torch.randn(2, 1, 64, 64, requires_grad=True)
    targets = torch.randint(0, 2, (2, 1, 64, 64)).float()
    
    criterion = FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=0.75)
    
    loss = criterion(logits, targets)
    print(f"Calculated Loss: {loss.item():.4f}")
    
    assert not torch.isnan(loss)
    assert loss.item() >= 0
    
    loss.backward()
    
    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any()
    
    print("SUCCESS: Focal Tversky Loss passed all checks!")

if __name__ == "__main__":
    test_focal_tversky_loss()
