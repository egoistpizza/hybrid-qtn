"""Download the Massachusetts Roads dataset from Kaggle into data/mass_roads/{images,masks}.

The Kaggle archive stores aerial tiles as ``.tiff`` and road masks as ``.tif`` across its
train / val / test folders; both are flattened into one directory each, every other file
is skipped, and the result must hold all 1,171 pairs. Requires Kaggle credentials
(``KAGGLE_USERNAME`` / ``KAGGLE_KEY`` or ``~/.kaggle/kaggle.json``).
"""

from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath
from typing import Optional

from scripts.fetch import extract_zip, fetch_kaggle_dataset, pairing_problems, require_no_problems

KAGGLE_DATASET = "balraj98/massachusetts-roads-dataset"
DEFAULT_DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "mass_roads"
IMAGE_SUFFIX = ".tiff"
MASK_SUFFIX = ".tif"
EXPECTED_PAIRS = 1171


def destination_for(member: str) -> Optional[PurePosixPath]:
    """``images/<name>`` for tiles, ``masks/<name>`` for road masks, ``None`` for anything else."""
    path = PurePosixPath(member)
    suffix = path.suffix.lower()
    if suffix == IMAGE_SUFFIX:
        return PurePosixPath("images", path.name)
    if suffix == MASK_SUFFIX:
        return PurePosixPath("masks", path.name)
    return None


def download_mass_roads(data_root: Path) -> None:
    archive = fetch_kaggle_dataset(KAGGLE_DATASET, data_root / "archives")
    extract_zip(archive, data_root, destination_for)
    problems = pairing_problems(
        "Massachusetts Roads", data_root / "images", data_root / "masks", IMAGE_SUFFIX, MASK_SUFFIX, EXPECTED_PAIRS
    )
    require_no_problems("Massachusetts Roads", problems)
    print(f"Massachusetts Roads: {EXPECTED_PAIRS} image/mask pairs at {data_root}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the Massachusetts Roads dataset from Kaggle")
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    return parser.parse_args()


if __name__ == "__main__":
    download_mass_roads(parse_arguments().data_root)
