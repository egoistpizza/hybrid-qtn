import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from PIL import Image
from torch.utils.data import DataLoader

from dataset import DatasetSplits
from evaluate import (
    clean_state_dict,
    denormalize,
    evaluate_split,
    evaluation_sets,
    load_model_weights,
    save_probability_maps,
    save_visualizations,
)
from utils.metrics import calculate_all_metrics

@pytest.fixture
def dummy_logits_and_targets() -> tuple[torch.Tensor, torch.Tensor]:
    logits = torch.tensor([[[[10.0, -10.0], [10.0, -10.0]]]])
    targets = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    return logits, targets

def test_calculate_metrics_perfect_match():
    logits = torch.tensor([[[[10.0, 10.0], [10.0, 10.0]]]])
    targets = torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]])
    
    metrics = calculate_all_metrics(logits, targets, compute_hd95=False)
    
    assert pytest.approx(metrics["dice"], 0.001) == 1.0
    assert pytest.approx(metrics["iou"], 0.001) == 1.0

def test_calculate_metrics_partial_match(dummy_logits_and_targets):
    logits, targets = dummy_logits_and_targets
    
    metrics = calculate_all_metrics(logits, targets, compute_hd95=False)
    
    assert 0.0 < metrics["dice"] < 1.0
    assert 0.0 < metrics["iou"] < 1.0
    assert metrics["dice"] > metrics["iou"]

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

def test_clean_state_dict_drops_swa_metadata():
    cleaned = clean_state_dict({"n_averaged": torch.tensor(3), "module._orig_mod.head.weight": torch.tensor([1.0])})
    assert list(cleaned) == ["head.weight"]


def test_test_split_evaluates_every_named_set():
    splits = DatasetSplits(train=[1], val=[2], test_sets={"cod10k": [3], "nc4k": [4]})
    assert evaluation_sets(splits, "test") == {"cod10k": [3], "nc4k": [4]}
    assert evaluation_sets(splits, "val") == {"val": [2]}


def test_probability_maps_are_saved_per_sample(tmp_path):
    maps = [np.full((5, 7), 200, np.uint8), np.zeros((3, 4), np.uint8)]
    save_probability_maps(maps, ["first", "second"], tmp_path / "maps")
    assert np.array_equal(np.array(Image.open(tmp_path / "maps" / "first.png")), maps[0])
    assert np.array(Image.open(tmp_path / "maps" / "second.png")).shape == (3, 4)


class MaskFileSamples(torch.utils.data.Dataset):
    def __init__(self, directory: Path) -> None:
        self.paths = []
        for index, size in enumerate([(30, 40), (22, 26)]):
            path = directory / f"sample_{index}.png"
            ground_truth = np.zeros(size, np.uint8)
            ground_truth[5:15, 6:16] = 255
            Image.fromarray(ground_truth).save(path)
            self.paths.append(str(path))

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict:
        mask = torch.zeros(1, 16, 16)
        mask[:, 4:10, 4:10] = 1.0
        return {
            "image": mask.repeat(3, 1, 1),
            "mask": mask,
            "metadata": {"mask_path": self.paths[index], "sample_id": f"sample_{index}"},
        }


def test_evaluate_split_adds_foreground_map_metrics_and_saves_maps(tmp_path):
    model = torch.nn.Conv2d(3, 1, kernel_size=1)
    loader = DataLoader(MaskFileSamples(tmp_path), batch_size=2)
    metrics = evaluate_split(model, loader, torch.device("cpu"), foreground_maps=True, map_dir=tmp_path / "maps")
    assert {"dice", "hd95", "s_measure", "e_measure", "wf_measure", "map_mae"} <= set(metrics)
    assert np.array(Image.open(tmp_path / "maps" / "sample_1.png")).shape == (22, 26)


def test_evaluate_split_skips_foreground_map_metrics_when_disabled(tmp_path):
    model = torch.nn.Conv2d(3, 1, kernel_size=1)
    metrics = evaluate_split(model, DataLoader(MaskFileSamples(tmp_path), batch_size=2), torch.device("cpu"), foreground_maps=False)
    assert "s_measure" not in metrics
    assert not (tmp_path / "maps").exists()
