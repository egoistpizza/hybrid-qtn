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

# TODO: Maybe get the vars like KVASIR_SEG_URL from a YAML file in /configs
# TODO: Maybe move the whole download process in a file namely download.py (if ever needed for another dataset etc.)

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
import logging
from tqdm import tqdm

# Add parent folder so we can import from there
def _(): #   {{{
    import os, sys
    
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Get the parent
    parent_dir = os.path.dirname(current_dir)
    
    # Add it to sys.path
    sys.path.append(parent_dir)
_(); del _ # }}}

from utils.init_first import init_logger_basicconfig

init_logger_basicconfig()
logger = logging.getLogger(__name__)


def _ssl_context() -> ssl.SSLContext: # {{{
    """Build an SSL context using certifi's CA bundle when available."""
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()
# }}}


# def _download_with_curl(url: str, dest: Path) -> bool:
def _download_with_curl(url: str, dest: Path) -> None: # {{{
    """Try curl first — it does AIA chasing, which some servers rely on
    (notably datasets.simula.no, which serves an incomplete TLS chain)."""
    
    logger.debug(f"\x1b[38;5;240mdownload_kvasir_seg.py::_download_with_curl(url={url}, dest={dest})\x1b[0m")
    
    curl = shutil.which("curl")
    if curl is None:
        # return False
        raise RuntimeError("shutil.which(\"curl\") == None")
    
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"Downloading via curl {url}\n           -> {dest}")
    
    try:
        subprocess.run(
            [curl, "-fL", "--retry", "3", "-o", str(tmp), url],
            check=True,
        )
        tmp.replace(dest)
        
    except subprocess.CalledProcessError as exc:
        tmp.unlink(missing_ok=True)
        # print(f"  curl failed (exit {exc.returncode}); will try urllib fallback")
        # return False
        raise RuntimeError(f"Download failed (curl) (exit code: {exc.returncode}): {exc}")
    # return True
# }}}

KVASIR_SEG_URL = "https://datasets.simula.no/downloads/kvasir-seg.zip"
ARCHIVE_NAME = "kvasir-seg.zip"
EXTRACTED_MARKER = "Kvasir-SEG"


def _download_with_urllib(url: str, dest: Path, insecure_omit_SSL_context: bool, chunk: int = 1 << 20) -> None: # {{{
    logger.debug(f"\x1b[38;5;240mdownload_kvasir_seg.py::_download_with_urllib(url={url}, dest={dest}, insecure_omit_SSL_context={insecure_omit_SSL_context}, chunk={chunk})\x1b[0m")
    
    if insecure_omit_SSL_context:
        # Disable HTTPS context verifying to avoid "urllib.error.URLError: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED]
        #  certificate verify failed: unable to get local issuer certificate (_ssl.c:1032)>"
        ssl._create_default_https_context = ssl._create_unverified_context
        
        # Disable HTTPS context verifying to avoid "urllib.error.URLError: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED]
        #  certificate verify failed: unable to get local issuer certificate (_ssl.c:1032)>"
        # Removed context=_ssl_context as well (to avoid the same error) in urllib.request.urlopen(url, context=_ssl_context())
        def custom_urllib_request_urlopen(url): # {{{
            return urllib.request.urlopen(url)
        # }}}
        
    else:
        def custom_urllib_request_urlopen(url): # {{{
            return urllib.request.urlopen(url, context = _ssl_context())
        # }}}
    
    
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"Downloading via urllib {url}\n           -> {dest}")
    
    from urllib.error import HTTPError, URLError
    
    try:
        with custom_urllib_request_urlopen(url) as resp:
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
        
        tmp.replace(dest)
        sys.stdout.write("\n")
    
    except (HTTPError, URLError) as exc:
        tmp.unlink(missing_ok=True)
        # return False
        raise RuntimeError(f"Download failed (requests): {exc}")
    
    print(f"  sha256: {digest.hexdigest()}")
# }}}

def _download_with_requests(url: str, dest: Path, insecure_no_verify: bool, chunk: int = 1 << 20) -> None: # {{{
    logger.debug(f"\x1b[38;5;240mdownload_kvasir_seg.py::_download_with_requests(url={url}, dest={dest}, insecure_no_verify={insecure_no_verify}, chunk={chunk})\x1b[0m")
    import requests
    
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"Downloading via requests {url}\n           -> {dest}")
    
    from requests.exceptions import ConnectionError, SSLError, HTTPError
    
    try:
        response = requests.get(url, stream=True, verify = (not insecure_no_verify))
        response.raise_for_status()
        
        # Retrieve the total file size from the headers
        total_size = int(response.headers.get('content-length', 0))
        
    
        with tmp.open("wb") as file, tqdm(
            desc=dest.name,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            
            for data_chunk in response.iter_content(chunk_size=chunk):
                if data_chunk:
                    file.write(data_chunk)
                    pbar.update(len(data_chunk))
                
        tmp.replace(dest)
                    
    except (ConnectionError, SSLError, HTTPError) as exc:
        tmp.unlink(missing_ok=True)
        # return False
        raise RuntimeError(f"Download failed (requests): {exc}")
    
# }}}







def _download(url: str, dest: Path) -> None: # {{{
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.debug(f"\x1b[38;5;240mdownload_kvasir_seg.py::_download(url={url}, dest={dest})\x1b[0m")
    
    # Try CURL first...
    print(f"Will try CURL...")
    try:
        _download_with_curl(url, dest)
        return
    except RuntimeError as e:
        print(f"Error:")
        print(f"{str(e)}")
        print()
    
    # ...Then requests (insecure_no_verify = False)...
    print(f"Will try requests (insecure_no_verify = False)...")
    try:
        _download_with_requests(url, dest, insecure_no_verify = False)
        return
    except RuntimeError as e:
        print(f"Error:")
        print(f"{str(e)}")
        print()
    
    # ...Then try URLLib (insecure_omit_SSL_context = False)...
    print(f"Will try urllib (insecure_omit_SSL_context = False)...")
    try:
        _download_with_urllib(url, dest, insecure_omit_SSL_context = False)
        return
    except RuntimeError as e:
        print(f"Error:")
        print(f"{str(e)}")
        print()
    
    # ...Then requests (insecure_no_verify = True)...
    print(f"Will try requests (insecure_no_verify = True)...")
    try:
        _download_with_requests(url, dest, insecure_no_verify = True)
        return
    except RuntimeError as e:
        print(f"Error:")
        print(f"{str(e)}")
        print()
    
    # ...Then try URLLib (insecure_omit_SSL_context = True)
    print(f"Will try urllib (insecure_omit_SSL_context = True)...")
    try:
        _download_with_urllib(url, dest, insecure_omit_SSL_context = True)
        return
    except Exception as e:
        print(f"Error:")
        print(f"{str(e)}")
        print()
# }}}


def _extract(archive: Path, dest: Path) -> None: # {{{
    print(f"Extracting {archive.name} -> {dest}")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest)
# }}}


def download_kvasir_seg(dest: Path, force: bool = False) -> Path: # {{{
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
# }}}


def _parse_args() -> argparse.Namespace: # {{{
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
# }}}


if __name__ == "__main__": # {{{
    args = _parse_args()
    download_kvasir_seg(args.dest, force=args.force)
# }}}
