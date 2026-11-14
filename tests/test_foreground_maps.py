import numpy as np
import pytest
import torch
from PIL import Image

from utils.foreground_maps import (
    FOREGROUND_MAP_METRIC_NAMES,
    ForegroundMapAccumulator,
    binarize_ground_truth,
    full_resolution_maps,
    normalize_prediction,
    probability_map,
    structure_measure,
)

HEIGHT, WIDTH = 48, 64
REFERENCE_TOLERANCE = 1e-12

PYSODMETRICS_1_6_2_REFERENCE = {
    "noisy_disc": (0.7453278726225067, 0.6430504896007484, 0.3809454280311941, 0.16844873366013072),
    "shifted_pair": (0.6160101746390243, 0.782986517526445, 0.49837284135918797, 0.14757838031045753),
    "empty_target": (0.5015586703431372, 0.49976213977531747, 0.0, 0.49844132965686283),
    "full_target": (0.49844132965686283, 0.5005634870563335, 0.778598993005892, 0.5015586703431372),
    "constant_prediction": (0.4494699528598962, 0.2500814067079128, 0.10167638170903355, 0.3428321589052287),
}
PYSODMETRICS_1_6_2_AGGREGATE = (0.5621616000242854, 0.5352888081333514, 0.3519187288210615, 0.3317718545751634)


def disc(center_row: int, center_column: int, radius: int) -> np.ndarray:
    rows, columns = np.ogrid[:HEIGHT, :WIDTH]
    return (rows - center_row) ** 2 + (columns - center_column) ** 2 <= radius**2


def texture() -> np.ndarray:
    rows, columns = np.indices((HEIGHT, WIDTH))
    return ((rows * 37 + columns * 11) % 101) / 100.0


def as_uint8(values: np.ndarray) -> np.ndarray:
    return (np.clip(values, 0.0, 1.0) * 255).astype(np.uint8)


def golden_cases() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    target = disc(20, 30, 10)
    pair = disc(12, 14, 7) | disc(34, 50, 9)
    return {
        "noisy_disc": (as_uint8(0.6 * target + 0.5 * texture() - 0.1), as_uint8(target)),
        "shifted_pair": (as_uint8(0.8 * np.roll(pair, 3, axis=1) + 0.2 * texture()), as_uint8(pair)),
        "empty_target": (as_uint8(texture()), np.zeros((HEIGHT, WIDTH), np.uint8)),
        "full_target": (as_uint8(texture()), np.full((HEIGHT, WIDTH), 255, np.uint8)),
        "constant_prediction": (np.full((HEIGHT, WIDTH), 77, np.uint8), as_uint8(target)),
    }


@pytest.mark.parametrize("name", sorted(PYSODMETRICS_1_6_2_REFERENCE))
def test_single_image_scores_match_the_reference_implementation(name):
    prediction, ground_truth = golden_cases()[name]
    accumulator = ForegroundMapAccumulator()
    accumulator.update(prediction, ground_truth)
    scores = accumulator.compute()
    expected = dict(zip(FOREGROUND_MAP_METRIC_NAMES, PYSODMETRICS_1_6_2_REFERENCE[name]))
    for metric, value in expected.items():
        assert scores[metric] == pytest.approx(value, abs=REFERENCE_TOLERANCE)


def test_dataset_scores_match_the_reference_aggregation():
    accumulator = ForegroundMapAccumulator()
    for prediction, ground_truth in golden_cases().values():
        accumulator.update(prediction, ground_truth)
    expected = dict(zip(FOREGROUND_MAP_METRIC_NAMES, PYSODMETRICS_1_6_2_AGGREGATE))
    for metric, value in expected.items():
        assert accumulator.compute()[metric] == pytest.approx(value, abs=REFERENCE_TOLERANCE)


def test_perfect_prediction_scores_are_ideal():
    target = as_uint8(disc(20, 30, 10))
    accumulator = ForegroundMapAccumulator()
    accumulator.update(target, target)
    scores = accumulator.compute()
    assert scores["s_measure"] == pytest.approx(1.0, abs=1e-6)
    assert scores["map_mae"] == 0.0
    assert scores["wf_measure"] == pytest.approx(1.0, abs=1e-6)


def test_foreground_confined_to_the_last_column_stays_finite():
    target = np.zeros((HEIGHT, WIDTH), dtype=bool)
    target[5:20, WIDTH - 1] = True
    score = structure_measure(normalize_prediction(as_uint8(texture())), target)
    assert np.isfinite(score)


def test_ground_truth_threshold_follows_the_benchmark_convention():
    assert binarize_ground_truth(np.array([128, 129], dtype=np.uint8)).tolist() == [False, True]


def test_constant_prediction_is_not_rescaled():
    assert np.all(normalize_prediction(np.full((4, 4), 51, np.uint8)) == pytest.approx(0.2))


def test_mismatched_shapes_are_rejected():
    with pytest.raises(ValueError, match="differs from ground truth shape"):
        ForegroundMapAccumulator().update(np.zeros((4, 4), np.uint8), np.zeros((4, 5), np.uint8))


def test_compute_without_updates_raises():
    with pytest.raises(ValueError, match="before any update"):
        ForegroundMapAccumulator().compute()


def test_probability_map_resizes_logits_to_the_requested_size():
    logits = torch.full((1, 8, 8), 20.0)
    prediction = probability_map(logits, (30, 50))
    assert prediction.shape == (30, 50)
    assert prediction.dtype == np.uint8
    assert np.all(prediction == 255)


def test_full_resolution_maps_follow_each_ground_truth_size(tmp_path):
    sizes = [(30, 40), (25, 60)]
    paths = []
    for index, size in enumerate(sizes):
        path = tmp_path / f"mask_{index}.png"
        Image.fromarray(np.zeros(size, np.uint8)).save(path)
        paths.append(str(path))
    maps = list(full_resolution_maps(torch.zeros(2, 1, 16, 16), paths))
    assert [prediction.shape for prediction, _ in maps] == sizes
    assert [ground_truth.shape for _, ground_truth in maps] == sizes
