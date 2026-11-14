"""Run one Kaggle session of a training queue: set up, continue the previous session, train.

A Kaggle GPU session ends after at most 12 hours. Each session (one "Save & Run All"
version of the notebook in ``notebooks/kaggle_session.ipynb``) therefore:

1. reports the environment (Python, torch, CUDA, GPUs, CPU count);
2. installs ``requirements-kaggle.txt`` on top of the Kaggle image;
3. downloads the dataset into ``/tmp``, outside the saved notebook output;
4. copies ``checkpoints/``, ``outputs/`` and ``backup/`` from the previous version's output,
   when it is attached as an input, so unfinished runs resume from their ``last.pt``;
5. samples GPU utilization and CPU load into ``outputs/kaggle/usage.csv``;
6. runs ``scripts.run_queue`` with the time left in the session, so the current run pauses
   at an epoch boundary instead of being killed, then gathers the results. With several
   GPUs, one queue lane runs per GPU (``--lanes``, default: one per GPU), side by side.

Everything under ``/kaggle/working/hybrid-qtn`` is kept as the version's output.

Usage, from the repository root on Kaggle (options other than the ones below go to the queue)::

    python -m scripts.kaggle_session --dataset duts --tag paper_v1 --tiers 1
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Sequence

from scripts.run_queue import DEFAULT_DATASET, DEFAULT_EPOCHS, queue_tag, visible_gpus

REPO_ROOT = Path(__file__).resolve().parents[1]
KAGGLE_INPUT = Path("/kaggle/input")
SCRATCH_DATA = Path("/tmp/hybrid-qtn-data")
REQUIREMENTS = "requirements-kaggle.txt"
# Directories carried from one session to the next.
PERSISTED = ("checkpoints", "outputs", "backup")
BACKUP_DIR = "backup"
USAGE_FILE = Path("outputs") / "kaggle" / "usage.csv"
USAGE_INTERVAL_SECONDS = 30
# Kaggle's session limit is 12 h; the default leaves an hour for start-up, the final
# SWA and test evaluation of a run, and saving the output.
DEFAULT_MAX_HOURS = 11.0
DOWNLOADERS = {
    "cod": ["scripts.download_cod"],
    "cvc_clinicdb": ["scripts.download_cvc_clinicdb"],
    "duts": ["scripts.download_duts"],
    "kvasir_seg": ["scripts.download_kvasir_seg", "--insecure"],
    "mass_roads": ["scripts.download_mass_roads"],
}


def environment_report() -> str:
    """Python, torch, CUDA, GPU and CPU details of this session."""
    import torch

    lines = [
        f"Python   {platform.python_version()}",
        f"torch    {torch.__version__} (CUDA {torch.version.cuda})",
        f"CPUs     {os.cpu_count()}",
    ]
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        lines.append(f"GPU {index}    {properties.name}, {properties.total_memory / 1024**3:.1f} GB")
    if not torch.cuda.is_available():
        lines.append("GPU      none (enable a GPU accelerator in the notebook settings)")
    return "\n".join(lines)


def run(command: Sequence[str], **options) -> None:
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True, **options)


def run_side_by_side(commands: Sequence[Sequence[str]], **options) -> None:
    """Run ``commands`` at the same time and wait for all of them.

    Raises:
        subprocess.CalledProcessError: For the first command that failed, once all have ended.
    """
    processes = []
    for command in commands:
        print("$ " + " ".join(command), flush=True)
        processes.append(subprocess.Popen(command, cwd=REPO_ROOT, **options))
    return_codes = [process.wait() for process in processes]
    for command, return_code in zip(commands, return_codes):
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command)


def link_data_directory(repo: Path, scratch: Path) -> Path:
    """Point ``repo/data`` at ``scratch``, so datasets stay out of the saved output.

    An existing ``data`` directory (a local checkout) is left alone.
    """
    data = repo / "data"
    if data.is_symlink() or not data.exists():
        scratch.mkdir(parents=True, exist_ok=True)
        if data.is_symlink():
            data.unlink()
        data.symlink_to(scratch, target_is_directory=True)
    return data


def previous_sessions(input_root: Path, repo_name: str) -> list[Path]:
    """Copies of the repository among the attached inputs that hold training state."""
    if not input_root.is_dir():
        return []
    return sorted(
        path for path in input_root.glob(f"**/{repo_name}")
        if path.is_dir() and any((path / name).is_dir() for name in PERSISTED)
    )


def restore_previous_session(repo: Path, sessions: Sequence[Path]) -> Optional[Path]:
    """Copy the persisted directories of the previous session into ``repo``.

    Raises:
        RuntimeError: If more than one previous session is attached, since it is unclear which to continue.
    """
    if not sessions:
        return None
    if len(sessions) > 1:
        listed = "\n  ".join(str(path) for path in sessions)
        raise RuntimeError(f"Attach only the latest version's output; found several:\n  {listed}")
    (session,) = sessions
    for name in PERSISTED:
        if (session / name).is_dir():
            shutil.copytree(session / name, repo / name, dirs_exist_ok=True, symlinks=True)
    return session


class UsageMonitor:
    """Appends GPU utilization, GPU memory and the 1-minute CPU load average to a CSV
    every ``interval`` seconds, from a background thread."""

    def __init__(self, path: Path, interval: float = USAGE_INTERVAL_SECONDS, query: Optional[Callable[[], list[list[str]]]] = None) -> None:
        self.path = path
        self.interval = interval
        self.query = query or query_gpus
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def __enter__(self) -> "UsageMonitor":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self.path.open("w", newline="") as output:
                csv.writer(output).writerow(["time", "gpu", "utilization_pct", "memory_mib", "cpu_load_1min", "cpus"])
        self._thread.start()
        return self

    def __exit__(self, *_) -> None:
        self._stop.set()
        self._thread.join()

    def sample(self) -> None:
        now = datetime.now().isoformat(timespec="seconds")
        load = os.getloadavg()[0]
        rows = self.query() or [["", "", ""]]
        with self.path.open("a", newline="") as output:
            writer = csv.writer(output)
            for gpu, utilization, memory in rows:
                writer.writerow([now, gpu, utilization, memory, f"{load:.2f}", os.cpu_count()])

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self.sample()


def query_gpus() -> list[list[str]]:
    """``[index, utilization %, memory MiB]`` per GPU, or ``[]`` without ``nvidia-smi``."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    return [[field.strip() for field in line.split(",")] for line in result.stdout.strip().splitlines()]


def queue_command(
    arguments: argparse.Namespace,
    queue_options: Sequence[str],
    hours_left: float,
    wandb_logged_in: bool,
    wandb_entity: Optional[str] = None,
    lane: int = 0,
    lanes: int = 1,
) -> list[str]:
    """The ``scripts.run_queue`` command of this session, for one lane of ``lanes``.

    ``wandb_entity`` (``WANDB_ENTITY``, which ``wandb.init`` also reads) makes the queue look
    up finished runs in the same W&B project the training logs to.
    """
    command = [
        sys.executable, "-m", "scripts.run_queue",
        "--dataset", arguments.dataset, "--tag", arguments.tag, "--epochs", str(arguments.epochs),
        "--max_hours", f"{hours_left:.3f}", "--backup_root", str(REPO_ROOT / BACKUP_DIR),
        *queue_options,
    ]
    if lanes > 1:
        command += ["--lanes", str(lanes), "--lane", str(lane)]
    if not wandb_logged_in:
        command.append("--skip_wandb_check")
    elif wandb_entity:
        command += ["--wandb_entity", wandb_entity]
    return command


def parse_arguments(argv: Optional[Sequence[str]] = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Run one Kaggle session of a training queue")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--max_hours", type=float, default=DEFAULT_MAX_HOURS, help="Budget of the whole session, set-up included")
    parser.add_argument("--lanes", type=int, default=None, help="Queue lanes run side by side, one per GPU; default: one per visible GPU")
    parser.add_argument("--skip_install", action="store_true")
    parser.add_argument("--skip_download", action="store_true")
    return parser.parse_known_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    session_started = time.monotonic()
    arguments, queue_options = parse_arguments(argv)
    if arguments.dataset not in DOWNLOADERS:
        sys.exit(f"No downloader for {arguments.dataset!r}. Known: {', '.join(sorted(DOWNLOADERS))}")

    print(environment_report(), flush=True)
    if not arguments.skip_install:
        run([sys.executable, "-m", "pip", "install", "--quiet", "-r", REQUIREMENTS])
    link_data_directory(REPO_ROOT, SCRATCH_DATA)
    if not arguments.skip_download:
        run([sys.executable, "-m", *DOWNLOADERS[arguments.dataset]])

    restored = restore_previous_session(REPO_ROOT, previous_sessions(KAGGLE_INPUT, REPO_ROOT.name))
    print(f"Continuing the session in {restored}" if restored else "No previous session attached; starting fresh.", flush=True)

    wandb_logged_in = bool(os.environ.get("WANDB_API_KEY"))
    environment = dict(os.environ)
    if not wandb_logged_in:
        print("WANDB_API_KEY is not set: W&B logs offline and the queue skips its W&B check.", flush=True)
        environment["WANDB_MODE"] = "offline"

    lanes = arguments.lanes or max(1, len(visible_gpus()))
    hours_left = arguments.max_hours - (time.monotonic() - session_started) / 3600
    commands = [
        queue_command(arguments, queue_options, hours_left, wandb_logged_in, os.environ.get("WANDB_ENTITY"), lane, lanes)
        for lane in range(lanes)
    ]
    with UsageMonitor(REPO_ROOT / USAGE_FILE):
        run_side_by_side(commands, env=environment)

    tag = queue_tag(arguments.tag, arguments.epochs)
    run([sys.executable, "-m", "scripts.collect_results", "--tag", tag, "--dataset", arguments.dataset,
         "--backup_root", str(REPO_ROOT / BACKUP_DIR), "--skip_wandb"])


if __name__ == "__main__":
    main()
