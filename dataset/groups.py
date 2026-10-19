"""Group-aware splitting for datasets whose samples are not independent.

CVC-ClinicDB contains 612 frames from 29 colonoscopy sequences. Frames of one sequence
show the same polyp from nearly the same pose, so a frame-level random split places
near-duplicates on both sides: on the seed-42 split, every validation frame had a
same-sequence sibling in training. A dataset opts in with a ``group_map`` block::

    group_map:
      csv: configs/cvc_clinicdb_sequences.csv
      sample_column: sample_id
      group_column: sequence_id

Datasets without the block keep the plain random split. A top-level
``cross_validation_folds: K`` additionally enables grouped K-fold cross-validation
(:func:`grouped_kfold_indices`), selected per run with ``--fold``.
"""

from __future__ import annotations

import csv
import random
from typing import Iterable, Sequence

from .config import GroupMap


def load_group_map(group_map: GroupMap) -> dict[str, str]:
    """Read the ``sample_id -> group_id`` mapping from the group-map CSV."""
    if not group_map.csv.is_file():
        raise FileNotFoundError(
            f"group_map csv not found: {group_map.csv}. Sequence-aware splitting is "
            "mandatory for this dataset — falling back to a frame-level random "
            "split would leak the same polyp into both halves."
        )

    mapping: dict[str, str] = {}
    with group_map.csv.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        missing = {group_map.sample_column, group_map.group_column} - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{group_map.csv} is missing column(s) {sorted(missing)}; found {reader.fieldnames}")
        for row in reader:
            mapping[row[group_map.sample_column].strip()] = row[group_map.group_column].strip()

    if not mapping:
        raise ValueError(f"{group_map.csv} contains no rows")
    return mapping


def groups_for_samples(sample_ids: Iterable[str], group_map: GroupMap) -> list[str]:
    """Group of every sample, in dataset order.

    Raises:
        KeyError: If a sample has no group; falling back to frame-level assignment would leak.
    """
    mapping = load_group_map(group_map)
    sample_ids = list(sample_ids)
    unknown = [sample_id for sample_id in sample_ids if sample_id not in mapping]
    if unknown:
        raise KeyError(
            f"{len(unknown)} sample(s) have no group in {group_map.csv} "
            f"(first: {unknown[:5]}). Refusing to split — an ungrouped frame "
            "would silently revert to leaky frame-level assignment."
        )
    return [mapping[sample_id] for sample_id in sample_ids]


def _closest_prefix_length(order: Sequence[str], members: dict[str, list[int]], target: float) -> int:
    running_total = 0
    best_length, best_gap = 1, float("inf")
    for length, group in enumerate(order, start=1):
        running_total += len(members[group])
        gap = abs(running_total - target)
        if gap < best_gap:
            best_gap, best_length = gap, length
    return best_length


def grouped_partition_indices(
    groups: Sequence[str],
    holdout_fractions: Sequence[float],
    seed: int,
) -> tuple[list[int], ...]:
    """Partition indices so that every group lands in exactly one partition.

    Groups are shuffled from sorted order (deterministic across platforms); each holdout
    partition takes the prefix of remaining groups whose sample count is closest to its
    target. Because whole groups move together, realized fractions only approximate the
    targets, and every partition keeps at least one group.

    Args:
        groups: Group id of every sample.
        holdout_fractions: Target fractions of the holdout partitions, e.g. ``(val, test)``.
        seed: Shuffle seed.

    Returns:
        ``(train, *holdouts)`` index lists, each sorted.
    """
    if not holdout_fractions or any(not 0.0 < fraction < 1.0 for fraction in holdout_fractions):
        raise ValueError(f"holdout fractions must each lie in (0, 1), got {list(holdout_fractions)}")
    if sum(holdout_fractions) >= 1.0:
        raise ValueError(f"holdout fractions must leave a training share, got {list(holdout_fractions)}")

    members: dict[str, list[int]] = {}
    for index, group in enumerate(groups):
        members.setdefault(group, []).append(index)

    remaining = sorted(members)
    if len(remaining) < len(holdout_fractions) + 1:
        raise ValueError(f"grouped split needs at least {len(holdout_fractions) + 1} groups, got {len(remaining)}")
    random.Random(seed).shuffle(remaining)

    holdout_groups: list[list[str]] = []
    for position, fraction in enumerate(holdout_fractions):
        groups_still_required = len(holdout_fractions) - position
        candidates = remaining[: len(remaining) - groups_still_required]
        prefix_length = _closest_prefix_length(candidates, members, fraction * len(groups))
        holdout_groups.append(remaining[:prefix_length])
        remaining = remaining[prefix_length:]

    partitions = [remaining, *holdout_groups]
    index_partitions = tuple(sorted(index for group in partition for index in members[group]) for partition in partitions)

    assert sum(len(partition) for partition in index_partitions) == len(groups)
    return index_partitions


def _assign_groups_to_folds(members: dict[str, list[int]], folds: int, seed: int) -> list[list[str]]:
    order = sorted(members)
    random.Random(seed).shuffle(order)
    order.sort(key=lambda group: len(members[group]), reverse=True)
    assignment: list[list[str]] = [[] for _ in range(folds)]
    loads = [0] * folds
    for group in order:
        lightest = min(range(folds), key=lambda fold: (loads[fold], fold))
        assignment[lightest].append(group)
        loads[lightest] += len(members[group])
    return assignment


def grouped_kfold_indices(
    groups: Sequence[str],
    folds: int,
    fold: int,
    val_fraction: float,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    """Train, validation and test indices of one fold of a grouped K-fold split.

    Groups are shuffled from sorted order (deterministic across platforms), then assigned
    largest first to the fold with the fewest samples, so folds are balanced by sample
    count and every group is tested in exactly one fold. Validation takes whole groups
    from the remaining ones, about ``val_fraction`` of all samples (see
    :func:`grouped_partition_indices`); the rest is training.

    Raises:
        ValueError: If ``folds`` < 2, ``fold`` is out of range, or there are too few groups.
    """
    if folds < 2:
        raise ValueError(f"cross-validation needs at least 2 folds, got {folds}")
    if not 0 <= fold < folds:
        raise ValueError(f"fold must lie in [0, {folds - 1}], got {fold}")
    members: dict[str, list[int]] = {}
    for index, group in enumerate(groups):
        members.setdefault(group, []).append(index)
    if len(members) < folds + 2:
        raise ValueError(f"{folds}-fold cross-validation needs at least {folds + 2} groups, got {len(members)}")

    test_groups = set(_assign_groups_to_folds(members, folds, seed)[fold])
    pool = [index for index, group in enumerate(groups) if group not in test_groups]
    pool_train, pool_val = grouped_partition_indices(
        [groups[index] for index in pool], (val_fraction * len(groups) / len(pool),), seed
    )
    test = sorted(index for group in test_groups for index in members[group])
    return sorted(pool[i] for i in pool_train), sorted(pool[i] for i in pool_val), test
