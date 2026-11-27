"""Download, extract and normalize the CVC-ClinicDB dataset.

CVC-ClinicDB holds 612 frames from 29 colonoscopy sequences (Bernal et al., 2015). The
official page only offers a Google Drive link to a .rar archive, which cannot be
scripted, so three sources are supported:

    --source kaggle   (default) the Kaggle mirror balraj98/cvcclinicdb, via the Kaggle API
    --source url      any direct HTTP(S) archive link given with --url (``--insecure`` skips
                      certificate verification for that download)
    --source local    an archive downloaded manually, given with --archive

The archive is extracted, its folders are moved to the layout expected by
configs/cvc_clinicdb.yaml, and the result is verified:

    data/cvc-clinicdb/CVC-ClinicDB/Original/
    data/cvc-clinicdb/CVC-ClinicDB/Ground Truth/

Mirrors differ in file format (.tif in the official release, .png in several mirrors);
the verification requires all 612 pairs and prints the extensions found so the YAML can
be set accordingly.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import urllib.parse
from collections import Counter
from pathlib import Path

from scripts.fetch import extract_zip, fetch, fetch_kaggle_dataset, pairing_problems, require_no_problems

KAGGLE_DATASET = "balraj98/cvcclinicdb"
EXTRACTED_MARKER = "CVC-ClinicDB"
IMAGES_SUBDIR = "Original"
MASKS_SUBDIR = "Ground Truth"
EXPECTED_COUNT = 612

IMAGE_DIR_ALIASES = {"original", "originals", "images", "image", "img", "frames"}
MASK_DIR_ALIASES = {
    "ground truth",
    "ground_truth",
    "groundtruth",
    "gt",
    "masks",
    "mask",
    "annotations",
}
IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


def _extract_rar(archive: Path, dest: Path) -> None:
    """Extract a .rar archive with the ``rarfile`` module, or with an external unrar / 7z."""
    try:
        import rarfile

        with rarfile.RarFile(archive) as rf:
            rf.extractall(dest)
        return
    except ImportError:
        pass
    except Exception as exc:
        print(f"  rarfile failed ({exc}); trying an external extractor")

    for exe, args in (
        ("7z", ["x", "-y", str(archive), f"-o{dest}"]),
        ("7za", ["x", "-y", str(archive), f"-o{dest}"]),
        ("unrar", ["x", "-y", str(archive), str(dest)]),
    ):
        binary = shutil.which(exe)
        if binary is None:
            continue
        subprocess.run([binary, *args], check=True)
        return

    raise RuntimeError(
        f"Cannot extract {archive.name}: no RAR extractor available.\n"
        "  Option 1: pip install rarfile  (still needs unrar/bsdtar on PATH)\n"
        "  Option 2: install 7-Zip and put 7z.exe on PATH\n"
        "  Option 3: extract the archive by hand into "
        f"{dest}, then re-run this script."
    )


def _extract(archive: Path, dest: Path) -> None:
    print(f"Extracting {archive.name} -> {dest}")
    suffix = archive.suffix.lower()
    if suffix == ".zip":
        extract_zip(archive, dest)
    elif suffix == ".rar":
        _extract_rar(archive, dest)
    else:
        raise RuntimeError(
            f"Unsupported archive type {suffix!r} for {archive.name}. "
            "Expected .zip or .rar."
        )


def _count_images(directory: Path) -> int:
    return sum(
        1
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def _find_layout(root: Path) -> tuple[Path, Path]:
    """Locate the image and mask directories in an extracted tree.

    Searches for a directory with one image-named and one mask-named subdirectory (see the
    alias sets) and keeps the pair containing the most image files.
    """
    best: tuple[int, Path, Path] | None = None

    for parent in [root, *(p for p in root.rglob("*") if p.is_dir())]:
        images_dir = masks_dir = None
        for child in parent.iterdir():
            if not child.is_dir():
                continue
            name = child.name.lower()
            if images_dir is None and name in IMAGE_DIR_ALIASES:
                images_dir = child
            elif masks_dir is None and name in MASK_DIR_ALIASES:
                masks_dir = child

        if images_dir is None or masks_dir is None:
            continue

        score = _count_images(images_dir) + _count_images(masks_dir)
        if score and (best is None or score > best[0]):
            best = (score, images_dir, masks_dir)

    if best is None:
        raise RuntimeError(
            f"Could not find an image/mask directory pair under {root}.\n"
            f"  Looked for a directory containing two subdirectories named one "
            f"of {sorted(IMAGE_DIR_ALIASES)} and one of "
            f"{sorted(MASK_DIR_ALIASES)}.\n"
            "  Inspect the extracted tree and move the folders into "
            f"{root / EXTRACTED_MARKER} manually."
        )

    return best[1], best[2]


def _normalize_layout(dest: Path) -> Path:
    """Move the discovered image and mask directories into the canonical layout."""
    extracted = dest / EXTRACTED_MARKER
    target_images = extracted / IMAGES_SUBDIR
    target_masks = extracted / MASKS_SUBDIR

    if target_images.is_dir() and target_masks.is_dir():
        return extracted

    images_dir, masks_dir = _find_layout(dest)
    print(f"Found images at {images_dir}\n       masks  at {masks_dir}")

    extracted.mkdir(parents=True, exist_ok=True)
    for src, target in ((images_dir, target_images), (masks_dir, target_masks)):
        if src.resolve() == target.resolve():
            continue
        if target.exists():
            shutil.rmtree(target)
        print(f"  moving {src} -> {target}")
        shutil.move(str(src), str(target))

    return extracted


def _dominant_extension(directory: Path) -> str:
    suffixes = (path.suffix.lower() for path in directory.iterdir() if path.is_file())
    counts = Counter(suffix for suffix in suffixes if suffix in IMAGE_EXTS)
    if not counts:
        raise RuntimeError(f"No image files in {directory}")
    return counts.most_common(1)[0][0]


def _verify(extracted: Path) -> None:
    """Require all 612 image/mask pairs and print the extensions to set in the YAML.

    Mirrors differ in file format (.tif or .png), so the dominant extension of each
    directory is used for pairing.

    Raises:
        RuntimeError: If a directory is missing, files are unpaired or the count differs.
    """
    images_dir = extracted / IMAGES_SUBDIR
    masks_dir = extracted / MASKS_SUBDIR
    for directory in (images_dir, masks_dir):
        if not directory.is_dir():
            raise RuntimeError(f"Expected directory is missing: {directory}")

    image_ext, mask_ext = _dominant_extension(images_dir), _dominant_extension(masks_dir)
    problems = pairing_problems("CVC-ClinicDB", images_dir, masks_dir, image_ext, mask_ext, EXPECTED_COUNT)
    require_no_problems("CVC-ClinicDB", problems)

    print(f"CVC-ClinicDB: {EXPECTED_COUNT} image/mask pairs. Set these in configs/cvc_clinicdb.yaml:")
    print(f'  images_dir: "{images_dir.as_posix()}"')
    print(f'  masks_dir:  "{masks_dir.as_posix()}"')
    print(f"  image_ext:  {image_ext}")
    print(f"  mask_ext:   {mask_ext}")


def download_cvc_clinicdb(
    dest: Path,
    source: str = "kaggle",
    url: str | None = None,
    archive: Path | None = None,
    insecure: bool = False,
    force: bool = False,
) -> Path:
    dest = dest.expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)
    extracted = dest / EXTRACTED_MARKER

    if extracted.is_dir() and not force:
        print(f"Already present: {extracted} (use --force to redo)")
        _verify(extracted)
        return extracted

    if source == "kaggle":
        archive_path = fetch_kaggle_dataset(KAGGLE_DATASET, dest)
    elif source == "url":
        if not url:
            raise ValueError("--source url requires --url")
        name = Path(urllib.parse.urlparse(url).path).name or "cvc-clinicdb.zip"
        archive_path = dest / name
        if force:
            archive_path.unlink(missing_ok=True)
        fetch(url, archive_path, insecure=insecure)
    elif source == "local":
        if not archive:
            raise ValueError("--source local requires --archive")
        archive_path = archive.expanduser().resolve()
        if not archive_path.is_file():
            raise FileNotFoundError(f"No such archive: {archive_path}")
    else:
        raise ValueError(f"Unknown source: {source!r}")

    _extract(archive_path, dest)
    extracted = _normalize_layout(dest)
    _verify(extracted)
    print(f"Done. Dataset at {extracted}")
    return extracted


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("data/cvc-clinicdb"),
        help="Destination directory (default: data/cvc-clinicdb)",
    )
    parser.add_argument(
        "--source",
        choices=["kaggle", "url", "local"],
        default="kaggle",
        help="Where to get the archive from (default: kaggle)",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="Direct HTTP(S) archive URL. Required for --source url.",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=None,
        help="Path to an already-downloaded .zip/.rar. Required for --source local.",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Skip TLS certificate verification for --source url.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and re-extract even if the dataset is present.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    download_cvc_clinicdb(
        args.dest,
        source=args.source,
        url=args.url,
        archive=args.archive,
        insecure=args.insecure,
        force=args.force,
    )
