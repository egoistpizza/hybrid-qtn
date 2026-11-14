import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from tests.synthetic import MaskFileSamples
from utils.evaluation import evaluate_model, metric_suites
from utils.loss import FocalTverskyLoss


@pytest.fixture
def model() -> torch.nn.Module:
    torch.manual_seed(0)
    return torch.nn.Conv2d(3, 1, kernel_size=1)


def evaluate(model, loader, **suite_options) -> dict[str, float]:
    return evaluate_model(model, loader, torch.device("cpu"), metric_suites(**suite_options))


def test_segmentation_metrics_are_reported_by_default(model, tmp_path):
    scores = evaluate(model, DataLoader(MaskFileSamples(tmp_path), batch_size=2))
    assert set(scores) == {"dice", "iou", "mae", "f2", "hd95"}


def test_hd95_is_omitted_when_not_computed(model, tmp_path):
    scores = evaluate(model, DataLoader(MaskFileSamples(tmp_path), batch_size=2), compute_hd95=False)
    assert "hd95" not in scores


def test_foreground_map_metrics_are_added_on_request(model, tmp_path):
    scores = evaluate(model, DataLoader(MaskFileSamples(tmp_path), batch_size=2), foreground_maps=True)
    assert {"s_measure", "e_measure", "wf_measure", "map_mae"} <= set(scores)


def test_loss_is_the_image_weighted_mean_over_batches(model, tmp_path):
    criterion = FocalTverskyLoss()
    dataset = MaskFileSamples(tmp_path)
    scores = evaluate(model, DataLoader(dataset, batch_size=2), criterion=criterion)
    with torch.no_grad():
        per_image = [criterion(model(sample["image"][None]), sample["mask"][None]).item() for sample in dataset]
    expected = (2 * np.mean(per_image[:2]) + per_image[2]) / 3
    assert scores["loss"] == pytest.approx(expected, rel=1e-5)


def test_scores_do_not_depend_on_the_batch_size(model, tmp_path):
    dataset = MaskFileSamples(tmp_path)
    assert evaluate(model, DataLoader(dataset, batch_size=1)) == pytest.approx(evaluate(model, DataLoader(dataset, batch_size=3)))


def test_evaluation_leaves_the_model_in_eval_mode(model, tmp_path):
    model.train()
    evaluate(model, DataLoader(MaskFileSamples(tmp_path), batch_size=2))
    assert not model.training
