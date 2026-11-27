"""Albumentations pipelines built from YAML transform specs.

Each spec is ``{name: <Albumentations class>, **kwargs}``; ``ToTensorV2`` is also accepted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2

TRAIN_STAGE = "train"
EVAL_STAGE = "eval"
STAGES = (TRAIN_STAGE, EVAL_STAGE)
DEFAULT_AUGMENTATION_SEED = 0
TENSOR_CONVERSION = "ToTensorV2"
SIZING_TRANSFORMS = ("Resize", "RandomCrop", "CenterCrop")
NORMALIZATION = "Normalize"
STANDARD_NORMALIZATION = "standard"
NORMALIZE_DEFAULTS = {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225), "max_pixel_value": 255.0}
PIXEL_RANGE = 255.0

DETERMINISTIC_TRANSFORMS = frozenset(
    {
        "Resize",
        "LongestMaxSize",
        "SmallestMaxSize",
        "PadIfNeeded",
        "CenterCrop",
        NORMALIZATION,
        TENSOR_CONVERSION,
    }
)

_EXTRA_TRANSFORMS: dict[str, Any] = {TENSOR_CONVERSION: ToTensorV2}


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


def spec_name(spec: Mapping[str, Any]) -> str:
    if "name" not in spec:
        raise ValueError(f"Transform spec missing 'name' key: {spec!r}")
    return spec["name"]


def build_transforms_from_config(
    specs: Sequence[Mapping[str, Any]] | None,
    seed: int = DEFAULT_AUGMENTATION_SEED,
) -> A.Compose:
    """Compose the specs into one image/mask pipeline; an empty list gives the identity.

    ``seed`` initializes the Albumentations RNG, which is independent of the global seeds.
    """
    pipeline = [
        _resolve_transform(spec_name(spec))(**{key: value for key, value in spec.items() if key != "name"})
        for spec in specs or []
    ]
    return A.Compose(pipeline, seed=seed)


def output_size(specs: Sequence[Mapping[str, Any]]) -> tuple[int, int]:
    """``(height, width)`` set by the last sizing transform of a pipeline.

    Raises:
        ValueError: If no transform in ``SIZING_TRANSFORMS`` fixes the output size.
    """
    for spec in reversed(specs):
        if spec_name(spec) in SIZING_TRANSFORMS:
            return int(spec["height"]), int(spec["width"])
    raise ValueError(f"A pipeline needs one of {SIZING_TRANSFORMS} to fix its output size")


def assert_deterministic_pipeline(specs: Sequence[Mapping[str, Any]]) -> None:
    """Reject random transforms, so validation and test scores are reproducible.

    Raises:
        ValueError: If a spec is not in ``DETERMINISTIC_TRANSFORMS``.
    """
    stochastic = [spec_name(spec) for spec in specs if spec_name(spec) not in DETERMINISTIC_TRANSFORMS]
    if stochastic:
        raise ValueError(
            f"Evaluation pipelines must be deterministic; found stochastic transform(s) {stochastic}. "
            f"Allowed: {sorted(DETERMINISTIC_TRANSFORMS)}."
        )


@dataclass(frozen=True)
class Denormalization:
    """Maps a ``(3, H, W)`` network input back to an ``(H, W, 3)`` image in [0, 1]: ``x * scale + offset``."""

    offset: tuple[float, ...]
    scale: tuple[float, ...]

    def __call__(self, image: torch.Tensor) -> np.ndarray:
        offset = torch.tensor(self.offset, device=image.device).view(-1, 1, 1)
        scale = torch.tensor(self.scale, device=image.device).view(-1, 1, 1)
        return torch.clamp(image * scale + offset, 0.0, 1.0).cpu().permute(1, 2, 0).numpy()


def denormalization(specs: Sequence[Mapping[str, Any]]) -> Denormalization:
    """Inverse of the pipeline's ``Normalize`` step; without one, the 0-255 input is only rescaled.

    Raises:
        ValueError: If ``Normalize`` uses a mode other than ``standard``.
    """
    normalize = next((spec for spec in specs if spec_name(spec) == NORMALIZATION), None)
    if normalize is None:
        return Denormalization(offset=(0.0, 0.0, 0.0), scale=(1.0 / PIXEL_RANGE,) * 3)
    if normalize.get("normalization", STANDARD_NORMALIZATION) != STANDARD_NORMALIZATION:
        raise ValueError(f"Only {STANDARD_NORMALIZATION!r} normalization can be inverted, got {normalize['normalization']!r}")
    parameters = {**NORMALIZE_DEFAULTS, **normalize}
    pixel_scale = float(parameters["max_pixel_value"]) / PIXEL_RANGE
    return Denormalization(
        offset=tuple(mean * pixel_scale for mean in _per_channel(parameters["mean"])),
        scale=tuple(std * pixel_scale for std in _per_channel(parameters["std"])),
    )


def _per_channel(value: float | Sequence[float]) -> tuple[float, ...]:
    return tuple(float(item) for item in value) if isinstance(value, Sequence) else (float(value),) * 3
