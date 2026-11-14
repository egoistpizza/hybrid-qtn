"""Model evaluation over a loader, scored by pluggable metric suites.

:func:`evaluate_model` runs inference once per batch (eval mode, ``inference_mode``,
autocast on CUDA, channels-last inputs) and hands the logits to every suite. A suite
accumulates over batches and returns named scores; a suite with side effects, such as
writing prediction maps, returns no scores.
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.foreground_maps import ForegroundMapAccumulator, full_resolution_maps
from utils.metrics import MetricAccumulator

Batch = Mapping[str, Any]


class MetricSuite(Protocol):
    def update(self, logits: torch.Tensor, masks: torch.Tensor, batch: Batch) -> None:
        """Consume the logits and device masks of one batch; ``batch`` is the loader's CPU batch."""

    def compute(self) -> dict[str, float]:
        """Scores accumulated so far."""


class LossMeter:
    """Image-weighted mean of ``criterion``, reported as ``loss``."""

    def __init__(self, criterion: nn.Module) -> None:
        self.criterion = criterion
        self._weighted_total = 0.0
        self._image_count = 0

    def update(self, logits: torch.Tensor, masks: torch.Tensor, batch: Batch) -> None:
        with torch.autocast(logits.device.type, enabled=logits.device.type == "cuda"):
            loss = self.criterion(logits, masks)
        self._weighted_total += loss.item() * masks.size(0)
        self._image_count += masks.size(0)

    def compute(self) -> dict[str, float]:
        return {"loss": self._weighted_total / self._image_count}


class SegmentationMetrics:
    """Per-image Dice, IoU, MAE, F2 and, with ``compute_hd95``, HD95 at network resolution."""

    def __init__(self, compute_hd95: bool) -> None:
        self._accumulator = MetricAccumulator(compute_hd95=compute_hd95)

    def update(self, logits: torch.Tensor, masks: torch.Tensor, batch: Batch) -> None:
        self._accumulator.update(logits, masks)

    def compute(self) -> dict[str, float]:
        return self._accumulator.compute()


class ForegroundMapMetrics:
    """Foreground-map metrics at ground-truth resolution; batches must carry ``metadata.mask_path``."""

    def __init__(self) -> None:
        self._accumulator = ForegroundMapAccumulator()

    def update(self, logits: torch.Tensor, masks: torch.Tensor, batch: Batch) -> None:
        for prediction, ground_truth in full_resolution_maps(logits, batch["metadata"]["mask_path"]):
            self._accumulator.update(prediction, ground_truth)

    def compute(self) -> dict[str, float]:
        return self._accumulator.compute()


def metric_suites(
    criterion: nn.Module | None = None, compute_hd95: bool = True, foreground_maps: bool = False
) -> list[MetricSuite]:
    """The scoring suites of one evaluation: optional loss, segmentation metrics, optional foreground maps."""
    suites: list[MetricSuite] = [LossMeter(criterion)] if criterion is not None else []
    suites.append(SegmentationMetrics(compute_hd95))
    if foreground_maps:
        suites.append(ForegroundMapMetrics())
    return suites


@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    suites: Sequence[MetricSuite],
    description: str = "Evaluating",
) -> dict[str, float]:
    """Run ``model`` over ``loader`` once and return the merged scores of all suites."""
    model.eval()
    for batch in tqdm(loader, desc=description):
        images = batch["image"].to(device, non_blocking=True, memory_format=torch.channels_last)
        masks = batch["mask"].to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            logits = model(images)
        for suite in suites:
            suite.update(logits, masks, batch)

    scores: dict[str, float] = {}
    for suite in suites:
        scores.update(suite.compute())
    return scores
