"""Download and extract the CVC-ClinicDB dataset.

CVC-ClinicDB is 612 frames from 29 colonoscopy sequences (Bernal et al., 2015).
Unlike Kvasir-SEG there is no stable direct-download URL: the official page

    https://polyp.grand-challenge.org/CVCClinicDB/

hands out a JavaScript-rendered Google Drive link to a ``.rar``, which is not
scriptable and which ``zipfile`` cannot open. This script therefore supports
three sources:

    --source kaggle   (default) via the `kaggle` CLI, dataset balraj98/cvcclinicdb
    --source url      any direct HTTP(S) link you supply with --url
    --source local    an archive you already downloaded, passed via --archive

Whatever the source, the archive is extracted, its internal layout is
normalised, and the result is verified. Target layout (matches
``configs/cvc_clinicdb.yaml``)::

    data/cvc-clinicdb/<archive>
    data/cvc-clinicdb/CVC-ClinicDB/Original/*.tif
    data/cvc-clinicdb/CVC-ClinicDB/Ground Truth/*.tif

Mirrors differ in file format (.tif on the official release, .png on several
Kaggle mirrors). The verification step prints the extensions it actually found
so you can set ``image_ext`` / ``mask_ext`` in the YAML accordingly.

Usage:
    python scripts/download_cvc_clinicdb.py
    python scripts/download_cvc_clinicdb.py --source url --url https://.../CVC-ClinicDB.zip
    python scripts/download_cvc_clinicdb.py --source local --archive ~/Downloads/CVC-ClinicDB.rar
    python scripts/download_cvc_clinicdb.py --force
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import ssl
import subprocess
import sys
import urllib.parse
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

KAGGLE_DATASET = "balraj98/cvcclinicdb"
EXTRACTED_MARKER = "CVC-ClinicDB"
IMAGES_SUBDIR = "Original"
MASKS_SUBDIR = "Ground Truth"
EXPECTED_COUNT = 612

# Directory names used by the various mirrors, lowercased for matching.
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


def _ssl_context() -> ssl.SSLContext:
    """Build an SSL context using certifi's CA bundle when available."""
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _download_with_curl(url: str, dest: Path) -> bool:
    """Try curl first — it does AIA chasing, which some servers rely on."""
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


def _download_from_kaggle(dest: Path) -> Path:
    """Fetch the dataset zip with the `kaggle` CLI. Returns the archive path."""
    kaggle = shutil.which("kaggle")
    if kaggle is None:
        raise RuntimeError(
            "The `kaggle` CLI was not found on PATH.\n"
            "  Install it:      pip install kaggle\n"
            "  Add a token:     https://www.kaggle.com/settings -> API -> "
            "Create New Token, save kaggle.json to ~/.kaggle/kaggle.json\n"
            "Alternatively download the archive by hand and re-run with:\n"
            "  python scripts/download_cvc_clinicdb.py --source local "
            "--archive <path-to-archive>"
        )

    dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading via kaggle {KAGGLE_DATASET}\n           -> {dest}")
    try:
        subprocess.run(
            [kaggle, "datasets", "download", "-d", KAGGLE_DATASET, "-p", str(dest)],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"`kaggle datasets download` failed (exit {exc.returncode}). "
            "Most often this means the API token is missing or the dataset "
            "licence has not been accepted — open "
            f"https://www.kaggle.com/datasets/{KAGGLE_DATASET} once in a "
            "browser, then retry."
        ) from exc

    archives = sorted(dest.glob("*.zip"), key=lambda p: p.stat().st_mtime)
    if not archives:
        raise RuntimeError(f"kaggle reported success but no .zip appeared in {dest}")
    return archives[-1]


def _extract_rar(archive: Path, dest: Path) -> None:
    """Extract a .rar via the `rarfile` module, else an external unrar/7z."""
    try:
        import rarfile

        with rarfile.RarFile(archive) as rf:
            rf.extractall(dest)
        return
    except ImportError:
        pass
    except Exception as exc:  # rarfile present but no backend binary
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
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
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
    """Locate the image/mask directory pair inside an extracted tree.

    Mirrors nest the pair at different depths and under different names, so
    walk the tree and pick the sibling pair holding the most image files.
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
    """Move the discovered image/mask pair into the canonical layout."""
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


def _verify(extracted: Path) -> None:
    """Check pair counts and report the extensions found, for the YAML."""
    images_dir = extracted / IMAGES_SUBDIR
    masks_dir = extracted / MASKS_SUBDIR

    for directory in (images_dir, masks_dir):
        if not directory.is_dir():
            raise RuntimeError(f"Expected directory is missing: {directory}")

    def _index(directory: Path) -> dict[str, Path]:
        return {
            p.stem: p
            for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        }

    images = _index(images_dir)
    masks = _index(masks_dir)
    paired = set(images) & set(masks)

    def _dominant_ext(index: dict[str, Path]) -> str:
        counts = Counter(p.suffix.lower() for p in index.values())
        return counts.most_common(1)[0][0] if counts else "?"

    image_ext = _dominant_ext(images)
    mask_ext = _dominant_ext(masks)

    print("-" * 60)
    print(f"images:  {len(images):4d}  ({image_ext})  {images_dir}")
    print(f"masks:   {len(masks):4d}  ({mask_ext})  {masks_dir}")
    print(f"paired:  {len(paired):4d}")

    unpaired_images = sorted(set(images) - set(masks))
    unpaired_masks = sorted(set(masks) - set(images))
    if unpaired_images:
        print(f"  WARNING {len(unpaired_images)} image(s) without mask "
              f"(first: {unpaired_images[:5]})")
    if unpaired_masks:
        print(f"  WARNING {len(unpaired_masks)} mask(s) without image "
              f"(first: {unpaired_masks[:5]})")

    if len(paired) != EXPECTED_COUNT:
        print(
            f"  WARNING expected {EXPECTED_COUNT} pairs, found {len(paired)}. "
            "This mirror may be a subset or a pre-split variant — check before "
            "reporting numbers."
        )

    print("-" * 60)
    print("Set these in configs/cvc_clinicdb.yaml:")
    print(f'  images_dir: "{images_dir.as_posix()}"')
    print(f'  masks_dir:  "{masks_dir.as_posix()}"')
    print(f"  image_ext:  {image_ext}")
    print(f"  mask_ext:   {mask_ext}")


def download_cvc_clinicdb(
    dest: Path,
    source: str = "kaggle",
    url: str | None = None,
    archive: Path | None = None,
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
        archive_path = _download_from_kaggle(dest)
    elif source == "url":
        if not url:
            raise ValueError("--source url requires --url")
        name = Path(urllib.parse.urlparse(url).path).name or "cvc-clinicdb.zip"
        archive_path = dest / name
        if not archive_path.exists() or force:
            _download(url, archive_path)
        else:
            print(f"Reusing existing archive: {archive_path}")
    elif source == "local":
        if not archive:
            raise ValueError("--source local requires --archive")
        archive_path = archive.expanduser().resolve()
        if not archive_path.is_file():
            raise FileNotFoundError(f"No such archive: {archive_path}")
    else:  # pragma: no cover - argparse constrains this
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
        force=args.force,
    )
