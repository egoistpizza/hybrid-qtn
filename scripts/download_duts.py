"""Download and extract the DUTS salient object detection dataset.

DUTS (Wang et al., 2017) ships as two archives on the official site

    https://saliencydetection.net/duts/

    DUTS-TR.zip   10,553 training image/mask pairs   (~271 MB)
    DUTS-TE.zip    5,019 test image/mask pairs       (~140 MB)

Sources:

    --source url     (default) the official download links
    --source local   archives you already downloaded, via --tr-archive / --te-archive

Target layout (matches ``configs/duts.yaml``)::

    data/duts/DUTS-TR/DUTS-TR-Image/*.jpg
    data/duts/DUTS-TR/DUTS-TR-Mask/*.png
    data/duts/DUTS-TE/DUTS-TE-Image/*.jpg
    data/duts/DUTS-TE/DUTS-TE-Mask/*.png

Usage:
    python scripts/download_duts.py
    python scripts/download_duts.py --source local --tr-archive ~/Downloads/DUTS-TR.zip --te-archive ~/Downloads/DUTS-TE.zip
    python scripts/download_duts.py --force
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

BASE_URL = "https://saliencydetection.net/duts/download"
IMAGE_EXT = ".jpg"
MASK_EXT = ".png"
EXPECTED_COUNTS = {"TR": 10553, "TE": 5019}


def _split_dir(dest: Path, split: str) -> Path:
    return dest / f"DUTS-{split}"


def _ssl_context() -> ssl.SSLContext:
    """Build an SSL context using certifi's CA bundle when available."""
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _download_with_curl(url: str, dest: Path) -> bool:
    curl = shutil.which("curl")
    if curl is None:
        return False
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"Downloading via curl {url}\n           -> {dest}")
    try:
        subprocess.run([curl, "-fL", "--retry", "3", "-o", str(tmp), url], check=True)
    except subprocess.CalledProcessError as exc:
        tmp.unlink(missing_ok=True)
        print(f"  curl failed (exit {exc.returncode}); will try urllib fallback")
        return False
    tmp.replace(dest)
    return True


def _download(url: str, dest: Path, chunk: int = 1 << 20) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)

    if _download_with_curl(url, dest):
        return

    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"Downloading via urllib {url}\n           -> {dest}")
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(url, context=_ssl_context()) as resp:
            total = int(resp.headers.get("Content-Length") or 0)
            read = 0
            with tmp.open("wb") as f:
                while block := resp.read(chunk):
                    f.write(block)
                    digest.update(block)
                    read += len(block)
                    if total:
                        sys.stdout.write(
                            f"\r  {read / 1e6:8.2f} / {total / 1e6:.2f} MB "
                            f"({100.0 * read / total:5.1f}%)"
                        )
                    else:
                        sys.stdout.write(f"\r  {read / 1e6:8.2f} MB")
                    sys.stdout.flush()
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    sys.stdout.write("\n")
    tmp.replace(dest)
    print(f"  sha256: {digest.hexdigest()}")


def _extract(archive: Path, dest: Path, split: str) -> None:
    """Extract into a staging folder, then move DUTS-<split> into place.

    An interrupted extraction therefore never leaves a half-filled
    DUTS-<split> folder that a later run would mistake for a finished one.
    """
    target = _split_dir(dest, split)
    staging = dest / f".DUTS-{split}.extracting"
    shutil.rmtree(staging, ignore_errors=True)

    print(f"Extracting {archive.name} -> {dest}")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(staging)

    extracted = staging / target.name
    if not extracted.is_dir():
        found = sorted(p.name for p in staging.iterdir())
        shutil.rmtree(staging, ignore_errors=True)
        raise RuntimeError(
            f"{archive.name} has no top-level {target.name}/ folder (found: {found}). "
            "Is this the official DUTS archive?"
        )

    if target.exists():
        shutil.rmtree(target)
    extracted.replace(target)
    shutil.rmtree(staging, ignore_errors=True)


def _verify(dest: Path, split: str) -> None:
    root = _split_dir(dest, split)
    images_dir = root / f"DUTS-{split}-Image"
    masks_dir = root / f"DUTS-{split}-Mask"

    for directory in (images_dir, masks_dir):
        if not directory.is_dir():
            raise RuntimeError(f"Expected directory is missing: {directory}")

    images = {p.stem for p in images_dir.iterdir() if p.suffix.lower() == IMAGE_EXT}
    masks = {p.stem for p in masks_dir.iterdir() if p.suffix.lower() == MASK_EXT}
    paired = images & masks

    print(f"DUTS-{split}: images={len(images)} masks={len(masks)} paired={len(paired)}")

    if images != masks:
        raise RuntimeError(
            f"DUTS-{split}: {len(images - masks)} image(s) without mask, "
            f"{len(masks - images)} mask(s) without image. Re-run with --force."
        )
    if len(paired) != EXPECTED_COUNTS[split]:
        raise RuntimeError(
            f"DUTS-{split}: expected {EXPECTED_COUNTS[split]} pairs, found {len(paired)}. "
            "Re-run with --force."
        )


def download_duts(
    dest: Path,
    source: str = "url",
    archives: dict[str, Path | None] | None = None,
    force: bool = False,
) -> Path:
    dest = dest.expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)
    archives = archives or {}

    for split in EXPECTED_COUNTS:
        if _split_dir(dest, split).is_dir() and not force:
            print(f"Already present: {_split_dir(dest, split)} (use --force to redo)")
            _verify(dest, split)
            continue

        if source == "url":
            archive = dest / f"DUTS-{split}.zip"
            if not archive.exists() or force:
                _download(f"{BASE_URL}/DUTS-{split}.zip", archive)
            else:
                print(f"Reusing existing archive: {archive}")
        elif source == "local":
            archive = archives.get(split)
            if archive is None:
                raise ValueError(f"--source local requires --{split.lower()}-archive")
            archive = archive.expanduser().resolve()
            if not archive.is_file():
                raise FileNotFoundError(f"No such archive: {archive}")
        else:  # pragma: no cover - argparse constrains this
            raise ValueError(f"Unknown source: {source!r}")

        _extract(archive, dest, split)
        _verify(dest, split)

    print(f"Done. Dataset at {dest}")
    return dest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("data/duts"),
        help="Destination directory (default: data/duts)",
    )
    parser.add_argument(
        "--source",
        choices=["url", "local"],
        default="url",
        help="Where to get the archives from (default: url)",
    )
    parser.add_argument(
        "--tr-archive",
        type=Path,
        default=None,
        help="Path to an already-downloaded DUTS-TR.zip. Used with --source local.",
    )
    parser.add_argument(
        "--te-archive",
        type=Path,
        default=None,
        help="Path to an already-downloaded DUTS-TE.zip. Used with --source local.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and re-extract even if the dataset is present.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    download_duts(
        args.dest,
        source=args.source,
        archives={"TR": args.tr_archive, "TE": args.te_archive},
        force=args.force,
    )
