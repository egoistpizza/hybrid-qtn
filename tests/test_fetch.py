import hashlib
import ssl
import threading
import zipfile
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import PurePosixPath

import pytest
import requests

from scripts.download_mass_roads import destination_for as mass_roads_destination
from scripts.fetch import extract_zip, fetch, pairing_problems, require_no_problems

PAYLOAD = b"archive bytes" * 1000


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *_: object) -> None:
        pass


@pytest.fixture
def server(tmp_path):
    root = tmp_path / "served"
    root.mkdir()
    (root / "archive.zip").write_bytes(PAYLOAD)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), partial(QuietHandler, directory=str(root)))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def test_download_is_written_once_complete(server, tmp_path):
    sha256 = hashlib.sha256(PAYLOAD).hexdigest()
    path = fetch(f"{server}/archive.zip", tmp_path / "out" / "archive.zip", len(PAYLOAD), sha256)
    assert path.read_bytes() == PAYLOAD
    assert not path.with_suffix(".zip.part").exists()


def test_size_mismatch_raises_and_leaves_no_file(server, tmp_path):
    destination = tmp_path / "archive.zip"
    with pytest.raises(RuntimeError, match="expected 5 bytes"):
        fetch(f"{server}/archive.zip", destination, expected_size=5)
    assert list(tmp_path.iterdir()) == [tmp_path / "served"]


def test_digest_mismatch_raises(server, tmp_path):
    with pytest.raises(RuntimeError, match="expected SHA-256"):
        fetch(f"{server}/archive.zip", tmp_path / "archive.zip", sha256="0" * 64)


def test_http_errors_raise(server, tmp_path):
    with pytest.raises(requests.HTTPError):
        fetch(f"{server}/missing.zip", tmp_path / "missing.zip")


def test_unreachable_hosts_raise(tmp_path):
    with pytest.raises(requests.ConnectionError):
        fetch("http://127.0.0.1:9/archive.zip", tmp_path / "archive.zip")


def test_a_matching_existing_file_is_reused_without_a_request(tmp_path):
    destination = tmp_path / "archive.zip"
    destination.write_bytes(PAYLOAD)
    assert fetch("http://127.0.0.1:9/archive.zip", destination, len(PAYLOAD)) == destination


def test_insecure_requests_leave_the_global_tls_context_untouched(server, tmp_path):
    default_context = ssl._create_default_https_context
    fetch(f"{server}/archive.zip", tmp_path / "archive.zip", insecure=True)
    assert ssl._create_default_https_context is default_context


def test_extraction_follows_the_member_selector(tmp_path):
    archive = tmp_path / "roads.zip"
    with zipfile.ZipFile(archive, "w") as zip_file:
        for member in ("tiff/train/1.tiff", "tiff/train_labels/1.tif", "tiff/metadata.csv", "tiff/test/"):
            zip_file.writestr(member, b"" if member.endswith("/") else b"x")
    assert extract_zip(archive, tmp_path / "roads", mass_roads_destination) == 2
    assert sorted(path.relative_to(tmp_path / "roads").as_posix() for path in (tmp_path / "roads").rglob("*.*")) == [
        "images/1.tiff",
        "masks/1.tif",
    ]


def test_mass_roads_members_map_by_extension():
    assert mass_roads_destination("tiff/val/2.tiff") == PurePosixPath("images/2.tiff")
    assert mass_roads_destination("tiff/val_labels/2.tif") == PurePosixPath("masks/2.tif")
    assert mass_roads_destination("png/val/2.png") is None


def test_pairing_problems_are_reported_together(tmp_path):
    images, masks = tmp_path / "images", tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    (images / "a.jpg").touch()
    (masks / "b.jpg").touch()
    problems = pairing_problems("set", images, masks, ".jpg", ".jpg", expected=2)
    assert problems == ["set: 2 unpaired file(s)", "set: expected 2 images, found 1"]
    with pytest.raises(RuntimeError, match="set layout verification failed"):
        require_no_problems("set", problems)
