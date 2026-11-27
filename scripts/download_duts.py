"""Download and arrange the DUTS salient object detection benchmark.

Fetches the two official archives from https://saliencydetection.net/duts/ (Wang et al.,
CVPR 2017) and arranges them like the COD benchmark::

    data/duts/train/duts_tr/{images,masks}   10,553 DUTS-TR pairs
    data/duts/test/duts_te/{images,masks}     5,019 DUTS-TE pairs

Archives already in ``<data_root>/archives`` with the pinned size and SHA-256 are reused,
so manually downloaded copies can be placed there. The final layout is checked against
the official image counts.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Mapping, Optional

from scripts.fetch import extract_zip, fetch, pairing_problems, require_no_problems

DEFAULT_DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "duts"
DOWNLOAD_URL = "https://saliencydetection.net/duts/download"
IMAGE_SUFFIX = ".jpg"
MASK_SUFFIX = ".png"
EXPECTED_COUNTS = {
    "train/duts_tr": 10553,
    "test/duts_te": 5019,
}


@dataclass(frozen=True)
class Archive:
    filename: str
    size_bytes: int
    sha256: str
    root: str
    destination: str

    @property
    def kind_folders(self) -> dict[str, tuple[str, str]]:
        return {f"{self.root}-Image": ("images", IMAGE_SUFFIX), f"{self.root}-Mask": ("masks", MASK_SUFFIX)}


TRAIN_ARCHIVE = Archive(
    "DUTS-TR.zip", 270_997_309, "ce61e023c8f59d022b4d46981cf16813b83d089242e6489a45630d83962ea058", "DUTS-TR", "train/duts_tr"
)
TEST_ARCHIVE = Archive(
    "DUTS-TE.zip", 139_799_089, "d9905c33307e1d35c397604a701ed87f019acf060033076eec633f13a2eb8104", "DUTS-TE", "test/duts_te"
)
ARCHIVES = (TRAIN_ARCHIVE, TEST_ARCHIVE)


def download_archive(archive: Archive, directory: Path) -> Path:
    """Download ``archive`` into ``directory`` unless a verified copy exists."""
    return fetch(f"{DOWNLOAD_URL}/{archive.filename}", directory / archive.filename, archive.size_bytes, archive.sha256)


def destination_for(member: str) -> Optional[PurePosixPath]:
    """Relative target path of an archive member, or ``None`` for members that are not kept.

    Directories and stray files such as the ``.DS_Store`` in DUTS-TE are skipped.

    Raises:
        ValueError: For a file of unexpected type in an image or mask folder.
    """
    parts = PurePosixPath(member).parts
    if member.endswith("/") or len(parts) != 3:
        return None
    archive = next((archive for archive in ARCHIVES if archive.root == parts[0]), None)
    if archive is None or parts[1] not in archive.kind_folders:
        return None
    folder, suffix = archive.kind_folders[parts[1]]
    if not parts[2].endswith(suffix):
        raise ValueError(f"{member}: expected a {suffix} file in {parts[1]}")
    return PurePosixPath(archive.destination, folder, parts[2])


def verify_layout(data_root: Path, expected_counts: Mapping[str, int] = EXPECTED_COUNTS) -> None:
    """Check pairing and image counts of both sets.

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
    require_no_problems("DUTS", problems)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and arrange the DUTS salient object detection benchmark")
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    archive_directory = arguments.data_root / "archives"
    for archive in ARCHIVES:
        extract_zip(download_archive(archive, archive_directory), arguments.data_root, destination_for)
    verify_layout(arguments.data_root)
    for relative, expected in EXPECTED_COUNTS.items():
        print(f"{relative:14s} {expected:5d} image/mask pairs")


if __name__ == "__main__":
    main()
