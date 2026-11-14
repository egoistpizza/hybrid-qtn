from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from PIL import Image
from torch.utils.data import DataLoader

from dataset import DatasetSplits, Denormalization, EvaluationSet
from evaluate import PanelWriter, ProbabilityMapWriter, evaluation_sets
from tests.synthetic import MaskFileSamples
from utils.evaluation import evaluate_model


IDENTITY = Denormalization(offset=(0.0,) * 3, scale=(1.0,) * 3)


def test_test_split_evaluates_every_named_set():
    val, cod10k, nc4k = (EvaluationSet([index], True, IDENTITY) for index in range(3))
    splits = DatasetSplits(train=[1], val=val, test_sets={"cod10k": cod10k, "nc4k": nc4k}, image_size=16)
    assert evaluation_sets(splits, "test") == {"cod10k": cod10k, "nc4k": nc4k}
    assert evaluation_sets(splits, "val") == {"val": val}


def test_probability_maps_are_saved_at_ground_truth_resolution(tmp_path):
    loader = DataLoader(MaskFileSamples(tmp_path), batch_size=2)
    evaluate_model(torch.nn.Conv2d(3, 1, kernel_size=1), loader, torch.device("cpu"), [ProbabilityMapWriter(tmp_path / "maps")])
    assert np.array(Image.open(tmp_path / "maps" / "sample_0.png")).shape == (30, 40)
    assert np.array(Image.open(tmp_path / "maps" / "sample_1.png")).shape == (22, 26)


@patch("evaluate.plt.savefig")
def test_panels_are_numbered_in_loader_order(mock_savefig, tmp_path):
    loader = DataLoader(MaskFileSamples(tmp_path), batch_size=2)
    writer = PanelWriter(tmp_path / "panels", denormalization=IDENTITY)
    assert evaluate_model(torch.nn.Conv2d(3, 1, kernel_size=1), loader, torch.device("cpu"), [writer]) == {}
    assert [call.args[0].name for call in mock_savefig.call_args_list] == [
        "viz_001_sample_0.png",
        "viz_002_sample_1.png",
        "viz_003_sample_2.png",
    ]
