import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from scripts.backup_sync import (
    IGNORE_FILE,
    MARKER_DIR,
    NoSync,
    SyncError,
    SyncthingAPI,
    SyncthingSync,
    create_backup_sync,
    device_summary,
    normalize_device_id,
)


def device_id(letter: str) -> str:
    return "-".join([letter * 7] * 8)


POD, PC, NAS = device_id("P"), device_id("Q"), device_id("R")
UP_TO_DATE = {"completion": 100, "needBytes": 0, "needItems": 0, "remoteState": "valid"}


class FakeSyncthing:
    """In-memory stand-in for the parts of Syncthing's REST API the sync uses."""

    def __init__(self, paired=(), connected=()):
        self.devices = {device: {"deviceID": device, "name": ""} for device in (POD, *paired)}
        self.folders: dict[str, dict] = {}
        self.connected = set(connected)
        self.completion: dict[str, dict] = {}
        self.scans: list[dict] = []
        self.down = False

    def get(self, path, **params):
        if self.down:
            raise SyncError("connection refused")
        responses = {
            "/rest/system/status": lambda: {"myID": POD},
            "/rest/config/devices": lambda: [dict(device) for device in self.devices.values()],
            "/rest/config/defaults/device": lambda: {"deviceID": "", "name": "", "addresses": ["dynamic"]},
            "/rest/config/folders": lambda: [json.loads(json.dumps(folder)) for folder in self.folders.values()],
            "/rest/config/defaults/folder": lambda: {"id": "", "path": "", "type": "sendreceive", "devices": [{"deviceID": POD}]},
            "/rest/system/connections": lambda: {"connections": {device: {"connected": device in self.connected} for device in self.devices}},
            "/rest/db/completion": lambda: self.completion.get(params.get("device"), UP_TO_DATE),
            "/rest/db/status": lambda: {"state": "idle", "error": ""},
        }
        return responses[path]()

    def put(self, path, body):
        collection, _, key = path.rpartition("/")
        {"/rest/config/devices": self.devices, "/rest/config/folders": self.folders}[collection][key] = body

    def post(self, path, **params):
        if self.down:
            raise SyncError("connection refused")
        assert path == "/rest/db/scan"
        self.scans.append(params)


def shared_devices(fake, folder_id):
    return [entry["deviceID"] for entry in fake.folders[folder_id]["devices"]]


def test_device_ids_are_normalized_and_validated():
    assert normalize_device_id(PC.replace("-", "").lower()) == PC
    with pytest.raises(SyncError):
        normalize_device_id("not-a-device")


def test_prepare_shares_a_send_only_folder_with_a_new_device(tmp_path):
    fake = FakeSyncthing()
    sync = SyncthingSync(fake, "showcase_runs", [PC.lower()])
    sync.prepare(tmp_path / "showcase_runs")

    folder = fake.folders["hybrid-qtn-showcase_runs"]
    assert folder["type"] == "sendonly" and folder["path"] == str(tmp_path / "showcase_runs")
    assert shared_devices(fake, "hybrid-qtn-showcase_runs") == [POD, PC]
    assert PC in fake.devices
    assert (tmp_path / "showcase_runs" / MARKER_DIR).is_dir()
    assert "*.partial" in (tmp_path / "showcase_runs" / IGNORE_FILE).read_text()
    assert fake.scans == [{"folder": "hybrid-qtn-showcase_runs"}]


def test_prepare_defaults_to_every_paired_device(tmp_path):
    fake = FakeSyncthing(paired=[PC, NAS])
    SyncthingSync(fake, "t").prepare(tmp_path / "t")
    assert shared_devices(fake, "hybrid-qtn-t") == [POD, PC, NAS]


def test_prepare_without_a_device_refuses(tmp_path):
    with pytest.raises(SyncError, match="no device"):
        SyncthingSync(FakeSyncthing(), "t").prepare(tmp_path / "t")


def test_prepare_again_keeps_the_devices_already_sharing(tmp_path):
    fake = FakeSyncthing()
    SyncthingSync(fake, "t", [PC]).prepare(tmp_path / "t")
    SyncthingSync(fake, "t", [NAS]).prepare(tmp_path / "t")
    assert shared_devices(fake, "hybrid-qtn-t") == [POD, PC, NAS]


def test_prepare_refuses_a_folder_id_that_points_elsewhere(tmp_path):
    fake = FakeSyncthing()
    SyncthingSync(fake, "t", [PC]).prepare(tmp_path / "first")
    with pytest.raises(SyncError, match="already points"):
        SyncthingSync(fake, "t", [PC]).prepare(tmp_path / "second")


def test_notify_rescans_the_changed_path_and_never_raises(tmp_path):
    fake = FakeSyncthing()
    sync = SyncthingSync(fake, "t", [PC])
    sync.prepare(tmp_path / "t")
    sync.notify(tmp_path / "t" / "checkpoints" / "run")
    sync.notify(tmp_path / "t")
    assert fake.scans[1:] == [{"folder": "hybrid-qtn-t", "sub": "checkpoints/run"}, {"folder": "hybrid-qtn-t"}]

    fake.down = True
    sync.notify(tmp_path / "t")
    assert "rescan request failed" in sync.last_error


def test_status_reports_every_device(tmp_path):
    fake = FakeSyncthing(paired=[PC, NAS], connected=[PC])
    sync = SyncthingSync(fake, "t")
    assert "not set up yet" in sync.status()

    sync.prepare(tmp_path / "t")
    fake.completion[PC] = {"completion": 62.5, "needBytes": 3 * 1024**3, "needItems": 4, "remoteState": "valid"}
    fake.completion[NAS] = {"completion": 0, "needBytes": 0, "needItems": 0, "remoteState": "notSharing"}
    status = sync.status()
    assert "QQQQQQQ connected, 62.5% (3.0 GB to send)" in status
    assert "RRRRRRR offline, has not accepted the folder yet" in status
    assert not sync.synced()

    fake.completion = {}
    assert sync.synced()
    fake.down = True
    assert sync.status().startswith("syncthing: unavailable")


def test_device_summary_when_up_to_date():
    assert device_summary("laptop", True, UP_TO_DATE) == "laptop connected, up to date"


def test_no_sync_and_factory():
    assert NoSync().status() == "off"
    assert isinstance(create_backup_sync("none", "t"), NoSync)
    assert isinstance(create_backup_sync("syncthing", "t", [PC]), SyncthingSync)
    with pytest.raises(SyncError):
        create_backup_sync("syncthing", "t", ["bogus"])


@pytest.fixture
def syncthing_server():
    """A local HTTP server answering like Syncthing; records every request."""
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def _answer(self):
            length = int(self.headers.get("Content-Length") or 0)
            requests.append((self.command, self.path, self.headers.get("X-API-Key"), self.rfile.read(length)))
            if self.headers.get("X-API-Key") != "secret":
                self.send_response(403)
                self.end_headers()
                self.wfile.write(b"CSRF Error")
                return
            body = b'{"myID": "POD"}' if self.command == "GET" else b""
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        do_GET = do_PUT = do_POST = _answer

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}", requests
    server.shutdown()


def test_api_sends_the_key_parameters_and_json_body(syncthing_server):
    url, requests = syncthing_server
    api = SyncthingAPI(url, "secret")
    assert api.get("/rest/system/status") == {"myID": "POD"}
    assert api.put("/rest/config/folders/f", {"id": "f"}) is None
    assert api.post("/rest/db/scan", folder="f", sub="a b") is None
    assert [(method, path) for method, path, _, _ in requests] == [
        ("GET", "/rest/system/status"), ("PUT", "/rest/config/folders/f"), ("POST", "/rest/db/scan?folder=f&sub=a+b"),
    ]
    assert json.loads(requests[1][3]) == {"id": "f"}


def test_api_errors_become_sync_errors(syncthing_server):
    url, _ = syncthing_server
    with pytest.raises(SyncError, match="HTTP 403"):
        SyncthingAPI(url, "wrong").get("/rest/system/status")
    with pytest.raises(SyncError, match="no API key"):
        SyncthingAPI(url, "").get("/rest/system/status")
    with pytest.raises(SyncError, match="not reachable"):
        SyncthingAPI("http://127.0.0.1:9", "secret").get("/rest/system/status")
