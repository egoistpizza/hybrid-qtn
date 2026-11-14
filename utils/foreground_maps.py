"""Foreground-map metrics of the salient and camouflaged object detection benchmarks.

S-measure (Fan et al., ICCV 2017), mean E-measure (Fan et al., IJCAI 2018), weighted
F-measure (Margolin et al., CVPR 2014) and MAE are computed per image on the uint8
probability map at the original ground-truth resolution, and averaged over images.
The implementation reproduces the reference MATLAB toolbox conventions as implemented
in PySODMetrics 1.6.2: the prediction is min-max normalized per image, the ground truth
is thresholded at 128, and the E-measure curve uses 256 thresholds.
"""

from __future__ import annotations

from typing import Iterator, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import convolve, distance_transform_edt

EPSILON = np.spacing(1)
INTENSITY_LEVELS = 255
GROUND_TRUTH_THRESHOLD = 128
STRUCTURE_OBJECT_WEIGHT = 0.5
ALIGNMENT_THRESHOLD_COUNT = 256
WEIGHTED_F_BETA_SQUARED = 1.0
ERROR_SMOOTHING_KERNEL_SIZE = 7
ERROR_SMOOTHING_SIGMA = 5.0
BACKGROUND_IMPORTANCE_HALF_DISTANCE = 5.0
FOREGROUND_MAP_METRIC_NAMES = ("s_measure", "e_measure", "wf_measure", "map_mae")


def probability_map(logits: torch.Tensor, size: tuple[int, int]) -> np.ndarray:
    """Resize ``(1, h, w)`` logits bilinearly to ``size`` and return the sigmoid map as uint8."""
    resized = F.interpolate(logits.float()[None], size=size, mode="bilinear", align_corners=False)[0, 0]
    return np.rint(torch.sigmoid(resized).cpu().numpy() * INTENSITY_LEVELS).astype(np.uint8)


def read_ground_truth(path: str) -> np.ndarray:
    """Ground-truth mask as an uint8 grayscale array, at its original resolution."""
    with Image.open(path) as mask:
        return np.array(mask.convert("L"), dtype=np.uint8)


def full_resolution_maps(logits: torch.Tensor, mask_paths: Sequence[str]) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(prediction, ground_truth)`` uint8 pairs, each prediction resized to its mask."""
    for image_logits, mask_path in zip(logits, mask_paths, strict=True):
        ground_truth = read_ground_truth(mask_path)
        yield probability_map(image_logits, ground_truth.shape), ground_truth


def normalize_prediction(prediction: np.ndarray) -> np.ndarray:
    """Scale an uint8 map to [0, 1] and min-max normalize it; constant maps are only scaled."""
    scaled = prediction.astype(np.float64) / INTENSITY_LEVELS
    lowest, highest = scaled.min(), scaled.max()
    return (scaled - lowest) / (highest - lowest) if highest != lowest else scaled


def binarize_ground_truth(ground_truth: np.ndarray) -> np.ndarray:
    """Foreground mask of an uint8 ground truth, using the benchmark threshold."""
    return ground_truth > GROUND_TRUTH_THRESHOLD


def mean_absolute_error(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    return float(np.mean(np.abs(prediction - ground_truth)))


def _object_similarity(values: np.ndarray) -> float:
    mean = np.mean(values)
    deviation = np.std(values, ddof=1) if values.size > 1 else 0.0
    return 2.0 * mean / (mean**2 + 1.0 + deviation + EPSILON)


def _object_score(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    foreground_ratio = np.mean(ground_truth)
    foreground = _object_similarity(prediction[ground_truth]) * foreground_ratio
    background = _object_similarity((1.0 - prediction)[~ground_truth]) * (1.0 - foreground_ratio)
    return foreground + background


def _structural_similarity(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    if prediction.size == 0:
        return 0.0
    target = ground_truth.astype(np.float64)
    denominator = prediction.size - 1 + EPSILON
    prediction_mean, target_mean = np.mean(prediction), np.mean(target)
    prediction_variance = np.sum((prediction - prediction_mean) ** 2) / denominator
    target_variance = np.sum((target - target_mean) ** 2) / denominator
    covariance = np.sum((prediction - prediction_mean) * (target - target_mean)) / denominator

    numerator = 4.0 * prediction_mean * target_mean * covariance
    spread = (prediction_mean**2 + target_mean**2) * (prediction_variance + target_variance)
    if numerator != 0:
        return numerator / (spread + EPSILON)
    return 1.0 if spread == 0 else 0.0


def _region_score(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    height, width = ground_truth.shape
    area = height * width
    centroid_row, centroid_column = np.argwhere(ground_truth).mean(axis=0).round()
    # One-based like the MATLAB reference, whose quadrant ranges 1:X include X.
    row, column = int(centroid_row) + 1, int(centroid_column) + 1

    quadrants = (
        (slice(0, row), slice(0, column), row * column / area),
        (slice(0, row), slice(column, width), row * (width - column) / area),
        (slice(row, height), slice(0, column), (height - row) * column / area),
    )
    remaining_weight = 1.0 - sum(weight for _, _, weight in quadrants)
    quadrants += ((slice(row, height), slice(column, width), remaining_weight),)
    return sum(
        _structural_similarity(prediction[rows, columns], ground_truth[rows, columns]) * weight
        for rows, columns, weight in quadrants
    )


def structure_measure(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    """S-measure: equal blend of object-aware and region-aware structural similarity.

    Empty quadrants carry zero weight and contribute nothing, where the reference
    implementation returns NaN.
    """
    foreground_ratio = np.mean(ground_truth)
    if foreground_ratio == 0:
        return float(1.0 - np.mean(prediction))
    if foreground_ratio == 1:
        return float(np.mean(prediction))
    object_part = STRUCTURE_OBJECT_WEIGHT * _object_score(prediction, ground_truth)
    region_part = (1.0 - STRUCTURE_OBJECT_WEIGHT) * _region_score(prediction, ground_truth)
    return float(max(0.0, object_part + region_part))


def _enhanced_alignment(prediction_value: np.ndarray, target_value: float) -> np.ndarray:
    alignment = 2.0 * prediction_value * target_value / (prediction_value**2 + target_value**2 + EPSILON)
    return (alignment + 1.0) ** 2 / 4.0


def enhanced_alignment_curve(prediction: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
    """E-measure at each of the 256 thresholds, from the highest threshold to the lowest.

    Pixel counts per threshold come from cumulative histograms, so all thresholds are
    evaluated in one pass; the mean of the curve is the reported mean E-measure.
    """
    levels = (prediction * INTENSITY_LEVELS).astype(np.uint8)
    bin_edges = np.linspace(0, ALIGNMENT_THRESHOLD_COUNT, ALIGNMENT_THRESHOLD_COUNT + 1)
    true_positives = np.cumsum(np.flip(np.histogram(levels[ground_truth], bins=bin_edges)[0]))
    false_positives = np.cumsum(np.flip(np.histogram(levels[~ground_truth], bins=bin_edges)[0]))

    pixel_count = ground_truth.size
    foreground_count = np.count_nonzero(ground_truth)
    predicted_foreground = true_positives + false_positives
    predicted_background = pixel_count - predicted_foreground

    if foreground_count == 0:
        enhanced_sum = predicted_background
    elif foreground_count == pixel_count:
        enhanced_sum = predicted_foreground
    else:
        false_negatives = foreground_count - true_positives
        true_negatives = predicted_background - false_negatives
        prediction_mean = predicted_foreground / pixel_count
        target_mean = foreground_count / pixel_count
        parts = (
            (true_positives, 1.0 - prediction_mean, 1.0 - target_mean),
            (false_positives, 1.0 - prediction_mean, -target_mean),
            (false_negatives, -prediction_mean, 1.0 - target_mean),
            (true_negatives, -prediction_mean, -target_mean),
        )
        enhanced_sum = sum(count * _enhanced_alignment(predicted, target) for count, predicted, target in parts)
    return enhanced_sum / (pixel_count - 1 + EPSILON)


def _gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    radius = (size - 1) / 2
    rows, columns = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    kernel = np.exp(-(columns * columns + rows * rows) / (2.0 * sigma * sigma))
    kernel[kernel < np.finfo(kernel.dtype).eps * kernel.max()] = 0
    return kernel / kernel.sum()


def weighted_f_measure(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    """Weighted F-measure with distance-dependent error weights; 0 for an empty ground truth.

    Background errors take the error of the nearest foreground pixel and are smoothed with
    a Gaussian, and errors far from the object are weighted more heavily.
    """
    if not ground_truth.any():
        return 0.0
    background = ~ground_truth
    distance, nearest = distance_transform_edt(background, return_indices=True)

    error = np.abs(prediction - ground_truth)
    propagated_error = error.copy()
    propagated_error[background] = error[nearest[0][background], nearest[1][background]]
    smoothed_error = convolve(
        propagated_error,
        weights=_gaussian_kernel(ERROR_SMOOTHING_KERNEL_SIZE, ERROR_SMOOTHING_SIGMA),
        mode="constant",
        cval=0,
    )
    dependent_error = np.where(ground_truth & (smoothed_error < error), smoothed_error, error)
    importance = np.where(background, 2.0 - np.exp(np.log(0.5) / BACKGROUND_IMPORTANCE_HALF_DISTANCE * distance), 1.0)
    weighted_error = dependent_error * importance

    weighted_true_positives = np.sum(ground_truth) - np.sum(weighted_error[ground_truth])
    weighted_false_positives = np.sum(weighted_error[background])
    recall = 1.0 - np.mean(weighted_error[ground_truth])
    precision = weighted_true_positives / (weighted_true_positives + weighted_false_positives + EPSILON)
    return float(
        (1.0 + WEIGHTED_F_BETA_SQUARED) * recall * precision / (recall + WEIGHTED_F_BETA_SQUARED * precision + EPSILON)
    )


class ForegroundMapAccumulator:
    """Accumulates S-measure, E-measure, weighted F-measure and MAE over uint8 map pairs."""

    def __init__(self) -> None:
        self._structure_scores: list[float] = []
        self._alignment_curves: list[np.ndarray] = []
        self._weighted_f_scores: list[float] = []
        self._absolute_errors: list[float] = []

    def update(self, prediction: np.ndarray, ground_truth: np.ndarray) -> None:
        if prediction.shape != ground_truth.shape:
            raise ValueError(f"Prediction shape {prediction.shape} differs from ground truth shape {ground_truth.shape}")
        normalized = normalize_prediction(prediction)
        target = binarize_ground_truth(ground_truth)
        self._structure_scores.append(structure_measure(normalized, target))
        self._alignment_curves.append(enhanced_alignment_curve(normalized, target))
        self._weighted_f_scores.append(weighted_f_measure(normalized, target))
        self._absolute_errors.append(mean_absolute_error(normalized, target))

    def compute(self) -> dict[str, float]:
        if not self._structure_scores:
            raise ValueError("ForegroundMapAccumulator.compute() called before any update()")
        return {
            "s_measure": float(np.mean(self._structure_scores)),
            "e_measure": float(np.mean(np.mean(self._alignment_curves, axis=0))),
            "wf_measure": float(np.mean(self._weighted_f_scores)),
            "map_mae": float(np.mean(self._absolute_errors)),
        }
