"""Download and extract the Kvasir-SEG dataset.

Kvasir-SEG is hosted by Simula Research Laboratory:
    https://datasets.simula.no/kvasir-seg/

Default layout after running (matches ``configs/kvasir_seg.yaml``):
    data/kvasir-seg/kvasir-seg.zip
    data/kvasir-seg/Kvasir-SEG/images/*.jpg
    data/kvasir-seg/Kvasir-SEG/masks/*.jpg

Usage:
    python scripts/download_kvasir_seg.py [--dest data/kvasir-seg] [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import ssl
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path


def _ssl_context() -> ssl.SSLContext:
    """Build an SSL context using certifi's CA bundle when available."""
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _download_with_curl(url: str, dest: Path) -> bool:
    """Try curl first — it does AIA chasing, which some servers rely on
    (notably datasets.simula.no, which serves an incomplete TLS chain)."""
    curl = shutil.which("curl")
    if curl is None:
        return False
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"Downloading via curl {url}\n           -> {dest}")
    try:
        subprocess.run(
            [curl, "-fL", "--retry", "3", "-o", str(tmp), url],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        tmp.unlink(missing_ok=True)
        print(f"  curl failed (exit {exc.returncode}); will try urllib fallback")
        return False
    tmp.replace(dest)
    return True

KVASIR_SEG_URL = "https://datasets.simula.no/downloads/kvasir-seg.zip"
ARCHIVE_NAME = "kvasir-seg.zip"
EXTRACTED_MARKER = "Kvasir-SEG"


def _download(url: str, dest: Path, chunk: int = 1 << 20) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)

    if _download_with_curl(url, dest):
        return

    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"Downloading via urllib {url}\n           -> {dest}")

    with urllib.request.urlopen(url, context=_ssl_context()) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        digest = hashlib.sha256()
        read = 0
        with tmp.open("wb") as f:
            while True:
                block = resp.read(chunk)
                if not block:
                    break
                f.write(block)
                digest.update(block)
                read += len(block)
                if total:
                    pct = 100.0 * read / total
                    sys.stdout.write(
                        f"\r  {read / 1e6:8.2f} / {total / 1e6:.2f} MB "
                        f"({pct:5.1f}%)"
                    )
                else:
                    sys.stdout.write(f"\r  {read / 1e6:8.2f} MB")
                sys.stdout.flush()
    sys.stdout.write("\n")
    tmp.replace(dest)
    print(f"  sha256: {digest.hexdigest()}")


def _extract(archive: Path, dest: Path) -> None:
    print(f"Extracting {archive.name} -> {dest}")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest)


def download_kvasir_seg(dest: Path, force: bool = False) -> Path:
    dest = dest.expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / ARCHIVE_NAME
    extracted = dest / EXTRACTED_MARKER

    if extracted.is_dir() and not force:
        print(f"Already present: {extracted} (use --force to redo)")
        return extracted

    if not archive.exists() or force:
        _download(KVASIR_SEG_URL, archive)
    else:
        print(f"Reusing existing archive: {archive}")

    _extract(archive, dest)

    images = extracted / "images"
    masks = extracted / "masks"
    if not images.is_dir() or not masks.is_dir():
        raise RuntimeError(
            f"Extraction did not produce expected layout under {extracted}. "
            "Inspect the archive contents."
        )
    n_images = sum(1 for _ in images.iterdir())
    n_masks = sum(1 for _ in masks.iterdir())
    print(f"Done. images={n_images}, masks={n_masks} at {extracted}")
    return extracted


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("data/kvasir-seg"),
        help="Destination directory (default: data/kvasir-seg)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and re-extract even if the dataset is present.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    download_kvasir_seg(args.dest, force=args.force)
