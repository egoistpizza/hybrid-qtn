"""Group-aware splitting for datasets whose samples are not independent.

CVC-ClinicDB is 612 frames drawn from 29 colonoscopy video sequences. Frames
within a sequence show the same polyp from nearly the same pose, so a
frame-level random split places near-duplicates in both halves: measured on
the seed-42 split, 123/123 validation frames had a same-sequence sibling in
training. The resulting Dice scores memorisation, not generalisation.

A dataset opts in by adding a ``group_map`` block to its YAML::

    group_map:
      csv: configs/cvc_clinicdb_sequences.csv
      sample_column: sample_id
      group_column: sequence_id

Datasets without the block (Kvasir-SEG, whose 1000 images are independent
captures) keep the plain random split, bit-identical to before.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_group_map(spec: dict[str, Any], root: str | Path | None = None) -> dict[str, str]:
    """Read ``sample_id -> group_id`` from the CSV named by a ``group_map`` spec."""
    root = Path(root) if root is not None else REPO_ROOT
    csv_path = Path(spec["csv"])
    if not csv_path.is_absolute():
        csv_path = root / csv_path

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"group_map csv not found: {csv_path}. Sequence-aware splitting is "
            "mandatory for this dataset — falling back to a frame-level random "
            "split would leak the same polyp into both halves."
        )

    sample_col = spec.get("sample_column", "sample_id")
    group_col = spec.get("group_column", "group_id")

    mapping: dict[str, str] = {}
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        missing = {sample_col, group_col} - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"{csv_path} is missing column(s) {sorted(missing)}; "
                f"found {reader.fieldnames}"
            )
        for row in reader:
            mapping[row[sample_col].strip()] = row[group_col].strip()

    if not mapping:
        raise ValueError(f"{csv_path} contains no rows")
    return mapping


def groups_for_samples(
    sample_ids: Iterable[str],
    spec: dict[str, Any],
    root: str | Path | None = None,
) -> list[str]:
    """Group id per sample, in dataset order. Raises if any sample is unmapped."""
    mapping = load_group_map(spec, root)
    sample_ids = list(sample_ids)

    unknown = [s for s in sample_ids if s not in mapping]
    if unknown:
        raise KeyError(
            f"{len(unknown)} sample(s) have no group in {spec['csv']} "
            f"(first: {unknown[:5]}). Refusing to split — an ungrouped frame "
            "would silently revert to leaky frame-level assignment."
        )
    return [mapping[s] for s in sample_ids]


def grouped_split_indices(
    groups: Sequence[str],
    val_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Split indices so that no group contributes to both halves.

    Groups are shuffled from a sorted order (deterministic across platforms),
    then the prefix whose cumulative frame count lands closest to
    ``val_fraction * len(groups)`` becomes validation. Because whole sequences
    move together the realised fraction only approximates the target — with 29
    sequences of 6-26 frames, seed 42 gives 120/612 = 19.6%.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")

    members: dict[str, list[int]] = {}
    for index, group in enumerate(groups):
        members.setdefault(group, []).append(index)

    order = sorted(members)  # deterministic starting order, before shuffling
    if len(order) < 2:
        raise ValueError(
            f"grouped split needs at least 2 groups, got {len(order)}"
        )
    random.Random(seed).shuffle(order)

    target = val_fraction * len(groups)
    running = 0
    best_k, best_gap = 1, float("inf")
    for k, name in enumerate(order, start=1):
        running += len(members[name])
        gap = abs(running - target)
        if gap < best_gap:
            best_gap, best_k = gap, k

    val_names, train_names = order[:best_k], order[best_k:]
    if not val_names or not train_names:
        raise ValueError(
            f"val_fraction={val_fraction} leaves one half empty for "
            f"{len(order)} groups; adjust the fraction or the grouping."
        )

    val_idx = sorted(i for name in val_names for i in members[name])
    train_idx = sorted(i for name in train_names for i in members[name])

    assert not set(train_idx) & set(val_idx)
    return train_idx, val_idx