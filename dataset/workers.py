"""Per-worker seeding of DataLoader workers, including the Albumentations RNG."""

from __future__ import annotations

import random
from typing import Iterator

import numpy as np
from torch.utils.data import ConcatDataset, Dataset, Subset, get_worker_info

from utils.seed import derived_worker_seed

from .base import GenericSegmentationDataset


def iter_segmentation_datasets(dataset: Dataset) -> Iterator[GenericSegmentationDataset]:
    """Yield every :class:`GenericSegmentationDataset` nested in ``Subset`` / ``ConcatDataset`` wrappers."""
    if isinstance(dataset, GenericSegmentationDataset):
        yield dataset
    elif isinstance(dataset, Subset):
        yield from iter_segmentation_datasets(dataset.dataset)
    elif isinstance(dataset, ConcatDataset):
        for child in dataset.datasets:
            yield from iter_segmentation_datasets(child)


def seed_dataloader_worker(worker_id: int) -> None:
    """``worker_init_fn`` that derives each worker's seed from the loader's seeded generator.

    Albumentations keeps its own RNG, so the datasets in the worker are reseeded as well;
    otherwise every worker would replay the same augmentations.
    """
    worker_seed = derived_worker_seed()
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    worker_info = get_worker_info()
    if worker_info is None:
        return
    for segmentation_dataset in iter_segmentation_datasets(worker_info.dataset):
        segmentation_dataset.reseed_augmentations(worker_seed)
