"""Copy the queue's backup folder to another machine while the queue runs.

The queue talks to a :class:`BackupSync`: :meth:`~BackupSync.prepare` once before the first
run, :meth:`~BackupSync.notify` after every run is copied to ``<backup_root>/<tag>/``, and
:meth:`~BackupSync.status` whenever it rewrites the status file. ``--sync none`` (the default)
does nothing. ``--sync syncthing`` shares ``<backup_root>/<tag>/`` as a send-only folder of the
Syncthing instance on this machine, so every finished run reaches the paired devices (e.g. your
PC) shortly after its backup and outlives the instance.

Syncthing's REST API is reached at ``$SYNCTHING_URL`` (default ``http://127.0.0.1:18384``, the
internal address on the Vast.ai images) with the key ``$SYNCTHING_API_KEY`` (default
``$OPEN_BUTTON_TOKEN``, the key the Vast.ai images start Syncthing with).

Usage (from the repository root), to pair before a launch or to check that everything arrived
before destroying the instance:
    python -m scripts.backup_sync --tag showcase_runs --setup --syncthing_device <PC device ID>
    python -m scripts.backup_sync --tag showcase_runs            # print the sync status
    python -m scripts.backup_sync --tag showcase_runs --wait     # wait until every device has everything
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Optional, Protocol, Sequence

SYNC_CHOICES = ("none", "syncthing")
DEFAULT_SYNCTHING_URL = "http://127.0.0.1:18384"
FOLDER_ID_PREFIX = "hybrid-qtn-"
REQUEST_TIMEOUT_S = 10.0
SCAN_TIMEOUT_S = 120.0  # a scan request returns when the scan is done
WAIT_POLL_S = 30
MARKER_DIR = ".stfolder"
IGNORE_FILE = ".stignore"
# Temporary names of the queue's atomic copies: a half-copied run never leaves the machine.
IGNORE_PATTERNS = ("*.partial", "*.tmp")
DEVICE_ID_CHARACTERS = re.compile(r"[A-Z2-7]{56}")


class SyncError(RuntimeError):
    """The sync target cannot be reached or set up."""


class BackupSync(Protocol):
    def prepare(self, folder: Path) -> None:
        """Start syncing ``folder``; called once before the first run.

        Raises:
            SyncError: If syncing cannot be set up.
        """

    def notify(self, path: Path) -> None:
        """``path``, inside the folder, has changed; never raises."""

    def status(self) -> str:
        """One line for the queue's status file; never raises."""


class NoSync:
    """``--sync none``: finished runs stay on this machine."""

    def prepare(self, folder: Path) -> None:
        pass

    def notify(self, path: Path) -> None:
        pass

    def status(self) -> str:
        return "off"


# --------------------------------------------------------------------------- Syncthing


class SyncthingAPI:
    """Minimal client of Syncthing's REST API."""

    def __init__(self, url: str, api_key: str) -> None:
        self.url = url.rstrip("/")
        self.api_key = api_key
        # The API is local: never route it through an HTTP proxy from the environment.
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    @classmethod
    def from_environment(cls) -> "SyncthingAPI":
        api_key = os.environ.get("SYNCTHING_API_KEY") or os.environ.get("OPEN_BUTTON_TOKEN", "")
        return cls(os.environ.get("SYNCTHING_URL", DEFAULT_SYNCTHING_URL), api_key)

    def request(self, method: str, path: str, body: Any = None, timeout: float = REQUEST_TIMEOUT_S, **params: str) -> Any:
        """The decoded JSON response, or ``None`` for an empty one.

        Raises:
            SyncError: If there is no API key, Syncthing is unreachable or the request fails.
        """
        if not self.api_key:
            raise SyncError("no API key: set SYNCTHING_API_KEY (Vast.ai images set OPEN_BUTTON_TOKEN)")
        query = f"?{urllib.parse.urlencode(params)}" if params else ""
        request = urllib.request.Request(
            f"{self.url}{path}{query}",
            data=None if body is None else json.dumps(body).encode(),
            method=method,
            headers={"X-API-Key": self.api_key, "Content-Type": "application/json"},
        )
        try:
            with self.opener.open(request, timeout=timeout) as response:
                payload = response.read()
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace").strip()
            raise SyncError(f"{method} {path} failed: HTTP {error.code} {detail}") from error
        except (urllib.error.URLError, OSError) as error:
            raise SyncError(f"Syncthing API not reachable at {self.url} ({error})") from error
        return json.loads(payload) if payload.strip() else None

    def get(self, path: str, **params: str) -> Any:
        return self.request("GET", path, **params)

    def put(self, path: str, body: Any) -> Any:
        return self.request("PUT", path, body)

    def post(self, path: str, **params: str) -> Any:
        return self.request("POST", path, timeout=SCAN_TIMEOUT_S, **params)


def normalize_device_id(device_id: str) -> str:
    """Syncthing's canonical form ``XXXXXXX-XXXXXXX-...`` (8 groups of 7) of a device ID.

    Raises:
        SyncError: If ``device_id`` is not shaped like a device ID.
    """
    characters = re.sub(r"[\s-]", "", device_id).upper()
    if not DEVICE_ID_CHARACTERS.fullmatch(characters):
        raise SyncError(f"not a Syncthing device ID: {device_id!r}")
    return "-".join(characters[index:index + 7] for index in range(0, 56, 7))


def folder_id(tag: str) -> str:
    """Syncthing folder ID of a queue tag, e.g. ``hybrid-qtn-showcase_runs``."""
    return FOLDER_ID_PREFIX + re.sub(r"[^A-Za-z0-9_.-]", "-", tag)


def public_listen_address() -> Optional[str]:
    """This machine's direct Syncthing address on Vast.ai (public IP, port mapped for 72299), if known."""
    host, port = os.environ.get("PUBLIC_IPADDR"), os.environ.get("VAST_TCP_PORT_72299")
    return f"tcp://{host}:{port}" if host and port else None


def format_bytes(count: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if count < 1024 or unit == "GB":
            return f"{count:.0f} {unit}" if unit == "B" else f"{count:.1f} {unit}"
        count /= 1024
    raise AssertionError("unreachable")


def write_ignore_file(folder: Path) -> None:
    (folder / IGNORE_FILE).write_text(
        "// Written by scripts/backup_sync.py: temporary copies never leave this machine.\n"
        + "".join(f"{pattern}\n" for pattern in IGNORE_PATTERNS)
    )


def device_summary(name: str, connected: bool, completion: dict) -> str:
    link = "connected" if connected else "offline"
    remote_state = completion.get("remoteState", "valid")
    if remote_state == "notSharing":
        return f"{name} {link}, has not accepted the folder yet"
    if remote_state == "unknown":
        return f"{name} {link}, not seen since Syncthing started"
    need_bytes = completion.get("needBytes", 0)
    if need_bytes == 0 and completion.get("needItems", 0) == 0:
        return f"{name} {link}, up to date"
    return f"{name} {link}, {completion.get('completion', 0.0):.1f}% ({format_bytes(need_bytes)} to send)"


class SyncthingSync:
    """Shares the backup folder of one queue tag as a send-only Syncthing folder.

    Send-only: this machine never takes changes back from the other devices. Each tag is a
    folder of its own, ID ``hybrid-qtn-<tag>``. Devices are the requested ones, which are added
    to Syncthing if new, or else every device already paired with this machine.
    """

    def __init__(self, api: SyncthingAPI, tag: str, devices: Sequence[str] = ()) -> None:
        self.api = api
        self.tag = tag
        self.folder_id = folder_id(tag)
        self.devices = list(dict.fromkeys(normalize_device_id(device) for device in devices))
        self.folder: Optional[Path] = None
        self.last_error = ""

    def my_id(self) -> str:
        return self.api.get("/rest/system/status")["myID"]

    def prepare(self, folder: Path) -> None:
        folder = folder.resolve()
        my_id = self.my_id()
        folder.mkdir(parents=True, exist_ok=True)
        (folder / MARKER_DIR).mkdir(exist_ok=True)
        write_ignore_file(folder)
        self._share(folder, my_id, self._target_devices(my_id))
        self.folder = folder
        self.api.post("/rest/db/scan", folder=self.folder_id)

    def _target_devices(self, my_id: str) -> list[str]:
        known = {device["deviceID"] for device in self.api.get("/rest/config/devices")}
        for device_id in self.devices:
            if device_id not in known:
                device = self.api.get("/rest/config/defaults/device")
                device.update(deviceID=device_id, name="")  # Syncthing takes the name the device announces
                self.api.put(f"/rest/config/devices/{device_id}", device)
        targets = self.devices or sorted(known - {my_id})
        if not targets:
            raise SyncError(
                "no device to sync with: pass --syncthing_device <device ID of your PC's Syncthing>, "
                "or pair a device in this machine's Syncthing GUI first"
            )
        return targets

    def _share(self, folder: Path, my_id: str, targets: list[str]) -> None:
        existing = next((entry for entry in self.api.get("/rest/config/folders") if entry["id"] == self.folder_id), None)
        if existing and Path(existing["path"]).expanduser() != folder:
            raise SyncError(f"Syncthing folder {self.folder_id} already points to {existing['path']}, not {folder}")
        config = existing or self.api.get("/rest/config/defaults/folder")
        shared = {entry["deviceID"]: entry for entry in config.get("devices", [])}
        for device_id in (my_id, *targets):
            shared.setdefault(device_id, {"deviceID": device_id})
        config.update(
            id=self.folder_id, label=f"hybrid-qtn {self.tag}", path=str(folder), type="sendonly", paused=False,
            devices=list(shared.values()),
        )
        self.api.put(f"/rest/config/folders/{self.folder_id}", config)

    def notify(self, path: Path) -> None:
        if self.folder is None:
            return
        try:
            relative = Path(path).resolve().relative_to(self.folder).as_posix()
            params = {"folder": self.folder_id, **({"sub": relative} if relative != "." else {})}
            self.api.post("/rest/db/scan", **params)
            self.last_error = ""
        except (SyncError, ValueError) as error:
            self.last_error = f"rescan request failed: {error}"

    def _device_states(self) -> Optional[list[tuple[str, bool, dict]]]:
        """``(name, connected, completion)`` of every other device sharing the folder; ``None`` if not set up."""
        folder = next((entry for entry in self.api.get("/rest/config/folders") if entry["id"] == self.folder_id), None)
        if folder is None:
            return None
        my_id = self.my_id()
        names = {device["deviceID"]: device.get("name") or device["deviceID"][:7] for device in self.api.get("/rest/config/devices")}
        connections = self.api.get("/rest/system/connections").get("connections", {})
        return [
            (
                names.get(entry["deviceID"], entry["deviceID"][:7]),
                bool(connections.get(entry["deviceID"], {}).get("connected")),
                self.api.get("/rest/db/completion", folder=self.folder_id, device=entry["deviceID"]),
            )
            for entry in folder["devices"]
            if entry["deviceID"] != my_id
        ]

    def status(self) -> str:
        try:
            states = self._device_states()
            if states is None:
                return f"syncthing: folder {self.folder_id} is not set up yet (the queue sets it up at launch)"
            parts = [device_summary(*state) for state in states] or ["shared with no device"]
            folder_error = self.api.get("/rest/db/status", folder=self.folder_id).get("error")
        except (SyncError, KeyError, TypeError, AttributeError) as error:
            return f"syncthing: unavailable ({error})"
        problems = [f"folder error: {folder_error}"] if folder_error else []
        problems += [self.last_error] if self.last_error else []
        return f"syncthing {self.folder_id} -> " + "; ".join(parts + problems)

    def synced(self) -> bool:
        """Whether every device sharing the folder has all of it.

        Raises:
            SyncError: If Syncthing is unreachable or the folder is not set up.
        """
        states = self._device_states()
        if states is None:
            raise SyncError(f"folder {self.folder_id} is not set up")
        return bool(states) and all(
            completion.get("remoteState", "valid") == "valid"
            and completion.get("needBytes", 0) == 0
            and completion.get("needItems", 0) == 0
            for _, _, completion in states
        )

    def pairing_hint(self) -> str:
        address = public_listen_address()
        return f"This machine's device ID: {self.my_id()}" + (f"  (direct address {address})" if address else "")


def create_backup_sync(kind: str, tag: str, devices: Sequence[str] = ()) -> BackupSync:
    """The :class:`BackupSync` selected by ``--sync``.

    Raises:
        SyncError: If a device ID is malformed.
    """
    if kind == "none":
        return NoSync()
    if kind == "syncthing":
        return SyncthingSync(SyncthingAPI.from_environment(), tag, devices)
    raise ValueError(f"Unknown sync {kind!r}; choose from {SYNC_CHOICES}")


# --------------------------------------------------------------------------- command line


def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Set up or check the Syncthing copy of a queue tag's backup folder")
    parser.add_argument("--tag", required=True, help="Queue tag, e.g. showcase_runs (or showcase_runs_e1)")
    parser.add_argument("--backup_root", type=Path, default=None, help="Default: the queue's backup root")
    parser.add_argument("--syncthing_device", nargs="+", default=[], help="Device ID(s) to share with (--setup)")
    parser.add_argument("--setup", action="store_true", help="Create or update the folder and share it")
    parser.add_argument("--wait", action="store_true", help="Wait until every device has the whole folder")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    from scripts.run_queue import DEFAULT_BACKUP_ROOT

    arguments = parse_arguments(argv)
    try:
        sync = SyncthingSync(SyncthingAPI.from_environment(), arguments.tag, arguments.syncthing_device)
        if arguments.setup:
            folder = (arguments.backup_root or DEFAULT_BACKUP_ROOT) / arguments.tag
            sync.prepare(folder)
            print(f"Sharing {folder} as Syncthing folder {sync.folder_id}")
        print(sync.pairing_hint())
        print(sync.status())
        while arguments.wait and not sync.synced():
            time.sleep(WAIT_POLL_S)
            print(f"{time.strftime('%H:%M:%S')}  {sync.status()}")
        if arguments.wait:
            print("Every device has the whole folder.")
    except SyncError as error:
        sys.exit(f"Syncthing: {error}")


if __name__ == "__main__":
    main()
