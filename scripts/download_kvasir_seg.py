"""Download and extract the Kvasir-SEG dataset (1,000 polyp images with masks) from Simula.

Result::

    data/kvasir-seg/kvasir-seg.zip
    data/kvasir-seg/Kvasir-SEG/images/*.jpg
    data/kvasir-seg/Kvasir-SEG/masks/*.jpg

datasets.simula.no omits an intermediate certificate from its TLS chain, so a verified
download currently fails; ``--insecure`` skips certificate verification for this one
request. The archive's size and SHA-256 are pinned, so its content is checked either way.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.fetch import extract_zip, fetch, pairing_problems, require_no_problems

KVASIR_SEG_URL = "https://datasets.simula.no/downloads/kvasir-seg.zip"
ARCHIVE_NAME = "kvasir-seg.zip"
ARCHIVE_SIZE_BYTES = 46_227_172
ARCHIVE_SHA256 = "03b30e21d584e04facf49397a2576738fd626815771afbbf788f74a7153478f7"
EXTRACTED_MARKER = "Kvasir-SEG"
EXPECTED_PAIRS = 1000


def download_kvasir_seg(destination: Path, insecure: bool = False, force: bool = False) -> Path:
    """Download, extract and verify Kvasir-SEG under ``destination``; returns the dataset directory."""
    destination = destination.expanduser().resolve()
    extracted = destination / EXTRACTED_MARKER
    if extracted.is_dir() and not force:
        print(f"Already present: {extracted} (use --force to redo)")
    else:
        archive = fetch(
            KVASIR_SEG_URL, destination / ARCHIVE_NAME, ARCHIVE_SIZE_BYTES, ARCHIVE_SHA256, insecure=insecure
        )
        extract_zip(archive, destination)
    problems = pairing_problems("Kvasir-SEG", extracted / "images", extracted / "masks", ".jpg", ".jpg", EXPECTED_PAIRS)
    require_no_problems("Kvasir-SEG", problems)
    print(f"Kvasir-SEG: {EXPECTED_PAIRS} image/mask pairs at {extracted}")
    return extracted


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and extract Kvasir-SEG")
    parser.add_argument("--dest", type=Path, default=Path("data/kvasir-seg"))
    parser.add_argument("--insecure", action="store_true", help="Skip TLS certificate verification for the download")
    parser.add_argument("--force", action="store_true", help="Re-extract even if the dataset is present")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    download_kvasir_seg(arguments.dest, insecure=arguments.insecure, force=arguments.force)
