import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock

from evaluate import (
    calculate_metrics,
    denormalize,
    save_visualizations,
    load_model_weights,
    build_model
)

import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from evaluate import (
    calculate_metrics,
    denormalize,
    save_visualizations,
    load_model_weights,
    build_model
)


@pytest.fixture
def dummy_logits_and_targets() -> tuple[torch.Tensor, torch.Tensor]:
    logits = torch.tensor([[[[10.0, -10.0], [10.0, -10.0]]]])
    targets = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    return logits, targets


def test_calculate_metrics_perfect_match():
    logits = torch.tensor([[[[10.0, 10.0], [10.0, 10.0]]]])
    targets = torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]])
    
    dice, iou = calculate_metrics(logits, targets)
    
    assert pytest.approx(dice, 0.001) == 1.0
    assert pytest.approx(iou, 0.001) == 1.0


def test_calculate_metrics_partial_match(dummy_logits_and_targets):
    logits, targets = dummy_logits_and_targets
    
    dice, iou = calculate_metrics(logits, targets)
    
    assert 0.0 < dice < 1.0
    assert 0.0 < iou < 1.0
    assert dice > iou


def test_denormalize_bounds_and_shape():
    tensor = torch.randn(3, 32, 32)
    
    denorm = denormalize(tensor)
    
    assert isinstance(denorm, np.ndarray)
    assert denorm.shape == (32, 32, 3)
    assert np.min(denorm) >= 0.0
    assert np.max(denorm) <= 1.0


@patch("evaluate.plt.savefig")
def test_save_visualizations_creates_files(mock_savefig, tmp_path):
    images = torch.randn(2, 3, 32, 32)
    masks = torch.zeros(2, 1, 32, 32)
    logits = torch.randn(2, 1, 32, 32)
    sample_ids = ["sample_A", "sample_B"]
    
    save_visualizations(images, masks, logits, sample_ids, tmp_path)
    
    assert mock_savefig.call_count == 2
    assert tmp_path.exists()


def test_load_model_weights_raises_file_not_found(tmp_path):
    dummy_model = MagicMock()
    non_existent_path = tmp_path / "does_not_exist.pth"
    
    with pytest.raises(FileNotFoundError, match="Checkpoint required but not found"):
        load_model_weights(dummy_model, non_existent_path, torch.device("cpu"))


@patch("evaluate.torch.load")
def test_load_model_weights_cleans_state_dict(mock_torch_load, tmp_path):
    fake_path = tmp_path / "fake_model.pth"
    fake_path.touch()
    
    mock_state_dict = {
        "_orig_mod.conv1.weight": torch.tensor([1.0]),
        "module.conv2.weight": torch.tensor([2.0]),
        "clean_layer.weight": torch.tensor([3.0])
    }
    mock_torch_load.return_value = mock_state_dict
    
    dummy_model = MagicMock()
    
    load_model_weights(dummy_model, fake_path, torch.device("cpu"))
    
    dummy_model.load_state_dict.assert_called_once()
    passed_dict = dummy_model.load_state_dict.call_args[0][0]
    
    assert "conv1.weight" in passed_dict
    assert "conv2.weight" in passed_dict
    assert "clean_layer.weight" in passed_dict
    assert "_orig_mod.conv1.weight" not in passed_dict
    assert "module.conv2.weight" not in passed_dict


def test_build_model_vanilla():
    model = build_model("vanilla", bond_dim=32)
    assert model.__class__.__name__ == "UNet"
    assert not hasattr(model, "transform_1024")


def test_build_model_hybrid():
    model = build_model("hybrid", bond_dim=32)
    assert model.__class__.__name__ == "UNetDeepHybrid"
    assert getattr(model, "transform_512", None) is None
    assert getattr(model, "transform_1024", None) is not None


def test_build_model_deep_hybrid():
    model = build_model("deep_hybrid", bond_dim=64)
    assert model.__class__.__name__ == "UNetDeepHybrid"
    assert getattr(model, "transform_512", None) is not None
    assert getattr(model, "transform_1024", None) is not None
