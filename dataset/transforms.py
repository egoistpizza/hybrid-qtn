"""Build an Albumentations pipeline from a YAML-friendly list of specs.

Config shape::

    transforms:
      - name: Resize
        height: 256
        width: 256
      - name: HorizontalFlip
        p: 0.5
      - name: Normalize
        mean: [0.485, 0.456, 0.406]
        std:  [0.229, 0.224, 0.225]
      - name: ToTensorV2
"""

from __future__ import annotations

from typing import Any

import albumentations as A
from albumentations.pytorch import ToTensorV2


_EXTRA_TRANSFORMS: dict[str, Any] = {
    "ToTensorV2": ToTensorV2,
}


def _resolve_transform(name: str) -> Any:
    if name in _EXTRA_TRANSFORMS:
        return _EXTRA_TRANSFORMS[name]
    cls = getattr(A, name, None)
    if cls is None:
        raise ValueError(
            f"Unknown Albumentations transform: {name!r}. "
            "Use the exact class name from `albumentations` "
            "(or 'ToTensorV2' for the tensor conversion)."
        )
    return cls


def build_transforms_from_config(
    specs: list[dict] | None,
) -> A.Compose:
    """Build an ``A.Compose`` from a list of ``{name, ...kwargs}`` dicts.

    The pipeline is constructed with ``additional_targets={"mask": "mask"}`` implicit
    to Albumentations — call sites should pass ``transform(image=img, mask=msk)``.
    An empty or missing list produces an identity pipeline.
    """
    specs = specs or []
    pipeline: list[Any] = []
    for spec in specs:
        if "name" not in spec:
            raise ValueError(f"Transform spec missing 'name' key: {spec!r}")
        kwargs = {k: v for k, v in spec.items() if k != "name"}
        cls = _resolve_transform(spec["name"])
        pipeline.append(cls(**kwargs))
    return A.Compose(pipeline)
