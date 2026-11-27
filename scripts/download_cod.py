"""Download and arrange the camouflaged object detection (COD) benchmark.

Fetches the training and testing archives released with SINet-V2 (Fan et al., TPAMI 2022)
and arranges them as::

    data/cod/train/{cod10k,camo}/{images,masks}
    data/cod/test/{cod10k,camo,chameleon,nc4k}/{images,masks}

The training archive mixes COD10K-train and CAMO-train in one folder; samples are
assigned to their source by filename prefix. Existing archives of the expected size
are reused, and the final layout is checked against the official image counts.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Mapping, Optional

from scripts.fetch import extract_zip, fetch, pairing_problems, require_no_problems

DEFAULT_DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "cod"
DRIVE_DOWNLOAD_URL = "https://drive.usercontent.google.com/download"
IMAGE_SUFFIX = ".jpg"
MASK_SUFFIX = ".png"
KIND_FOLDERS = {"Imgs": ("images", IMAGE_SUFFIX), "GT": ("masks", MASK_SUFFIX)}
TRAIN_SOURCE_PREFIXES = {"COD10K-": "cod10k", "camourflage_": "camo"}
TEST_SOURCE_FOLDERS = {"COD10K": "cod10k", "CAMO": "camo", "CHAMELEON": "chameleon", "NC4K": "nc4k"}
EXPECTED_COUNTS = {
    "train/cod10k": 3040,
    "train/camo": 1000,
    "test/cod10k": 2026,
    "test/camo": 250,
    "test/chameleon": 76,
    "test/nc4k": 4121,
}
USAGE_NOTICE = "COD10K, CAMO, CHAMELEON and NC4K are distributed for non-commercial research use only."


@dataclass(frozen=True)
class Archive:
    filename: str
    drive_id: str
    size_bytes: int
    root: str


TRAIN_ARCHIVE = Archive("COD-TrainDataset.zip", "1M8-Ivd33KslvyehLK9_IUBGJ_Kf52bWG", 1_089_873_536, "COD-TrainDataset")
TEST_ARCHIVE = Archive("COD-TestDataset.zip", "1V0iSEdYJrT0Y_DHZfVGMg6TySFRNTy4o", 1_310_372_023, "COD-TestDataset")


def download_archive(archive: Archive, directory: Path) -> Path:
    """Download ``archive`` from Google Drive into ``directory`` unless a complete copy exists."""
    parameters = {"id": archive.drive_id, "export": "download", "confirm": "t"}
    return fetch(DRIVE_DOWNLOAD_URL, directory / archive.filename, archive.size_bytes, params=parameters)


def _train_source(filename: str) -> str:
    for prefix, source in TRAIN_SOURCE_PREFIXES.items():
        if filename.startswith(prefix):
            return source
    raise ValueError(f"Training file {filename!r} matches none of the prefixes {sorted(TRAIN_SOURCE_PREFIXES)}")


def destination_for(member: str) -> Optional[PurePosixPath]:
    """Relative target path of an archive member, or ``None`` for members that are not kept.

    Directories, macOS metadata, edge maps and instance masks are skipped.

    Raises:
        ValueError: For a training file of unknown source or a file of unexpected type.
    """
    parts = PurePosixPath(member).parts
    if member.endswith("/") or not parts or parts[0] not in (TRAIN_ARCHIVE.root, TEST_ARCHIVE.root):
        return None

    if parts[0] == TRAIN_ARCHIVE.root and len(parts) == 3:
        split, source, kind, filename = "train", _train_source(parts[2]), parts[1], parts[2]
    elif parts[0] == TEST_ARCHIVE.root and len(parts) == 4 and parts[1] in TEST_SOURCE_FOLDERS:
        split, source, kind, filename = "test", TEST_SOURCE_FOLDERS[parts[1]], parts[2], parts[3]
    else:
        return None

    if kind not in KIND_FOLDERS:
        return None
    folder, suffix = KIND_FOLDERS[kind]
    if not filename.endswith(suffix):
        raise ValueError(f"{member}: expected a {suffix} file in {kind}")
    return PurePosixPath(split, source, folder, filename)


def verify_layout(data_root: Path, expected_counts: Mapping[str, int] = EXPECTED_COUNTS) -> None:
    """Check pairing and image counts of every set.

    Raises:
        RuntimeError: Listing every set with unpaired files or an unexpected count.
    """
    problems = [
        problem
        for relative, expected in expected_counts.items()
        for problem in pairing_problems(
            relative, data_root / relative / "images", data_root / relative / "masks", IMAGE_SUFFIX, MASK_SUFFIX, expected
        )
    ]
    require_no_problems("COD", problems)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and arrange the camouflaged object detection benchmark")
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    print(USAGE_NOTICE)
    archive_directory = arguments.data_root / "archives"
    for archive in (TRAIN_ARCHIVE, TEST_ARCHIVE):
        extract_zip(download_archive(archive, archive_directory), arguments.data_root, destination_for)
    verify_layout(arguments.data_root)
    for relative, expected in EXPECTED_COUNTS.items():
        print(f"{relative:16s} {expected:5d} image/mask pairs")


if __name__ == "__main__":
    main()
