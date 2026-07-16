"""Image ↔ mask filename pairing."""

from __future__ import annotations

import warnings
from pathlib import Path


def _normalize_ext(ext: str) -> str:
    ext = ext.strip().lower()
    if not ext.startswith("."):
        ext = "." + ext
    return ext


def _index_by_stem(directory: Path, ext: str) -> dict[str, Path]:
    ext = _normalize_ext(ext)
    index: dict[str, Path] = {}
    for path in directory.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() != ext:
            continue
        index[path.stem] = path
    return index


def pair_by_stem(
    images_dir: str | Path,
    masks_dir: str | Path,
    image_ext: str,
    mask_ext: str,
    strict_pairing: bool = True,
) -> list[tuple[str, str, str]]:
    """Pair image and mask files by filename stem.

    Returns a deterministic, stem-sorted list of ``(image_path, mask_path, sample_id)``.
    Matching is case-insensitive on the extension only; stems are matched as-is.

    If ``strict_pairing`` is True (default), any orphan on either side raises
    ``FileNotFoundError``. If False, orphans emit a warning and are skipped.
    """
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)

    if not images_dir.is_dir():
        raise NotADirectoryError(f"images_dir does not exist: {images_dir}")
    if not masks_dir.is_dir():
        raise NotADirectoryError(f"masks_dir does not exist: {masks_dir}")

    images = _index_by_stem(images_dir, image_ext)
    masks = _index_by_stem(masks_dir, mask_ext)

    image_stems = set(images)
    mask_stems = set(masks)
    missing_masks = sorted(image_stems - mask_stems)
    orphan_masks = sorted(mask_stems - image_stems)

    if missing_masks or orphan_masks:
        msg_parts = []
        if missing_masks:
            msg_parts.append(
                f"{len(missing_masks)} image(s) without matching mask "
                f"(first: {missing_masks[:5]})"
            )
        if orphan_masks:
            msg_parts.append(
                f"{len(orphan_masks)} mask(s) without matching image "
                f"(first: {orphan_masks[:5]})"
            )
        message = "; ".join(msg_parts)
        if strict_pairing:
            raise FileNotFoundError(f"Unpaired files in dataset: {message}")
        warnings.warn(f"Skipping unpaired files: {message}", stacklevel=2)

    paired_stems = sorted(image_stems & mask_stems)
    return [
        (str(images[stem]), str(masks[stem]), stem) for stem in paired_stems
    ]
