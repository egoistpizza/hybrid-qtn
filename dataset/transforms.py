"""Albumentations pipelines built from YAML transform specs.

Each spec is ``{name: <Albumentations class>, **kwargs}``; ``ToTensorV2`` is also accepted.
Training and evaluation use separate pipelines (``train_transforms`` / ``eval_transforms``),
with ``transforms`` as a shared fallback.
"""

from __future__ import annotations

from typing import Any

import albumentations as A
from albumentations.pytorch import ToTensorV2

TRAIN_STAGE = "train"
EVAL_STAGE = "eval"
STAGES = (TRAIN_STAGE, EVAL_STAGE)
SHARED_TRANSFORMS_KEY = "transforms"
DEFAULT_AUGMENTATION_SEED = 0

DETERMINISTIC_TRANSFORMS = frozenset(
    {
        "Resize",
        "LongestMaxSize",
        "SmallestMaxSize",
        "PadIfNeeded",
        "CenterCrop",
        "Normalize",
        "ToTensorV2",
    }
)

_EXTRA_TRANSFORMS: dict[str, Any] = {"ToTensorV2": ToTensorV2}


def _resolve_transform(name: str) -> Any:
    if name in _EXTRA_TRANSFORMS:
        return _EXTRA_TRANSFORMS[name]
    transform_class = getattr(A, name, None)
    if transform_class is None:
        raise ValueError(
            f"Unknown Albumentations transform: {name!r}. "
            "Use the exact class name from `albumentations` "
            "(or 'ToTensorV2' for the tensor conversion)."
        )
    return transform_class


def _spec_name(spec: dict) -> str:
    if "name" not in spec:
        raise ValueError(f"Transform spec missing 'name' key: {spec!r}")
    return spec["name"]


def build_transforms_from_config(
    specs: list[dict] | None,
    seed: int = DEFAULT_AUGMENTATION_SEED,
) -> A.Compose:
    """Compose the specs into one image/mask pipeline; an empty list gives the identity.

    ``seed`` initializes the Albumentations RNG, which is independent of the global seeds.
    """
    pipeline = [
        _resolve_transform(_spec_name(spec))(**{key: value for key, value in spec.items() if key != "name"})
        for spec in specs or []
    ]
    return A.Compose(pipeline, seed=seed)


def stage_transform_specs(config: dict[str, Any], stage: str) -> list[dict]:
    """Specs of ``stage``, falling back to the shared ``transforms`` key."""
    if stage not in STAGES:
        raise ValueError(f"Unknown stage {stage!r}. Expected one of {STAGES}.")
    return config.get(f"{stage}_transforms", config.get(SHARED_TRANSFORMS_KEY)) or []


def assert_deterministic_pipeline(specs: list[dict]) -> None:
    """Reject random transforms, so validation and test scores are reproducible.

    Raises:
        ValueError: If a spec is not in ``DETERMINISTIC_TRANSFORMS``.
    """
    stochastic = [_spec_name(spec) for spec in specs if _spec_name(spec) not in DETERMINISTIC_TRANSFORMS]
    if stochastic:
        raise ValueError(
            f"Evaluation pipelines must be deterministic; found stochastic transform(s) {stochastic}. "
            f"Allowed: {sorted(DETERMINISTIC_TRANSFORMS)}."
        )
