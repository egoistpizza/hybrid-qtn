"""Downloads, archive extraction and layout checks shared by the dataset download scripts.

Every failure raises; nothing falls back silently. Certificate verification is on unless
a caller passes ``insecure=True``, which affects that request only.
"""

from __future__ import annotations

import hashlib
import shutil
import zipfile
from pathlib import Path, PurePosixPath
from typing import Callable, Mapping, Optional

import requests
from tqdm import tqdm

CHUNK_BYTES = 1 << 20
REQUEST_TIMEOUT_SECONDS = 60

MemberSelector = Callable[[str], Optional[PurePosixPath]]


def fetch(
    url: str,
    destination: Path,
    expected_size: int | None = None,
    sha256: str | None = None,
    insecure: bool = False,
    params: Mapping[str, str] | None = None,
) -> Path:
    """Stream ``url`` to ``destination`` through a ``.part`` file that is renamed only once complete.

    An existing file that matches ``expected_size`` and ``sha256`` is reused. With
    ``insecure``, the server certificate is not verified; a pinned ``sha256`` still
    guarantees the content.

    Raises:
        requests.RequestException: On connection, TLS or HTTP errors.
        RuntimeError: If the size or digest of the download differs from the expected one.
    """
    if destination.is_file() and _mismatch(destination, expected_size, sha256) is None:
        print(f"Using existing {destination}")
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    try:
        with requests.get(url, params=params, stream=True, timeout=REQUEST_TIMEOUT_SECONDS, verify=not insecure) as response:
            response.raise_for_status()
            total = expected_size or int(response.headers.get("content-length", 0)) or None
            with partial.open("wb") as output, tqdm(total=total, unit="B", unit_scale=True, desc=destination.name) as progress:
                for chunk in response.iter_content(chunk_size=CHUNK_BYTES):
                    output.write(chunk)
                    progress.update(len(chunk))
        mismatch = _mismatch(partial, expected_size, sha256)
        if mismatch is not None:
            raise RuntimeError(f"{destination.name}: {mismatch}")
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    partial.replace(destination)
    return destination


def _mismatch(path: Path, expected_size: int | None, sha256: str | None) -> str | None:
    size = path.stat().st_size
    if expected_size is not None and size != expected_size:
        return f"expected {expected_size} bytes, found {size}"
    if sha256 is not None:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(CHUNK_BYTES), b""):
                digest.update(block)
        if digest.hexdigest() != sha256:
            return f"expected SHA-256 {sha256}, found {digest.hexdigest()}"
    return None


def fetch_kaggle_dataset(slug: str, destination: Path) -> Path:
    """Download the archive of the Kaggle dataset ``owner/name`` into ``destination``.

    Credentials come from ``KAGGLE_USERNAME`` / ``KAGGLE_KEY`` or ``~/.kaggle/kaggle.json``.
    """
    # The kaggle package authenticates on import, so it is imported only when needed.
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    destination.mkdir(parents=True, exist_ok=True)
    api.dataset_download_files(slug, path=str(destination), unzip=False, quiet=False)
    archive = destination / f"{slug.split('/')[-1]}.zip"
    if not archive.is_file():
        raise RuntimeError(f"Kaggle reported success but {archive} does not exist")
    return archive


def extract_zip(archive: Path, destination: Path, select: MemberSelector | None = None) -> int:
    """Extract ``archive`` into ``destination`` and return the number of files written.

    Args:
        select: Maps a member name to its relative target path, or to ``None`` to skip it.
            Without it, every file keeps its archive path.
    """
    written = 0
    with zipfile.ZipFile(archive) as zip_file:
        members = [(member, select(member) if select else PurePosixPath(member)) for member in zip_file.namelist()]
        selected = [(member, target) for member, target in members if target is not None and not member.endswith("/")]
        for member, target in tqdm(selected, desc=f"Extracting {archive.name}", unit="file"):
            path = destination / target
            path.parent.mkdir(parents=True, exist_ok=True)
            with zip_file.open(member) as source, path.open("wb") as output:
                shutil.copyfileobj(source, output)
            written += 1
    return written


def pairing_problems(
    name: str, images_dir: Path, masks_dir: Path, image_suffix: str, mask_suffix: str, expected: int
) -> list[str]:
    """Unpaired files and count mismatches of one image / mask directory pair."""
    images = {path.stem for path in images_dir.glob(f"*{image_suffix}")}
    masks = {path.stem for path in masks_dir.glob(f"*{mask_suffix}")}
    problems = []
    if images != masks:
        problems.append(f"{name}: {len(images ^ masks)} unpaired file(s)")
    if len(images) != expected:
        problems.append(f"{name}: expected {expected} images, found {len(images)}")
    return problems


def require_no_problems(dataset: str, problems: list[str]) -> None:
    if problems:
        raise RuntimeError(f"{dataset} layout verification failed:\n  " + "\n  ".join(problems))
