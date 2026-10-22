"""Train the Kvasir-SEG ablation grid as a resumable, prioritized queue.

The grid is vanilla and hybrid U-Nets x bond dimension {32, 64} x axial operator {relative,
absolute, circular, mean boundary} x positional encoding {off, on} x loss {Focal Tversky,
BCE + Dice}: 34 configurations, each trained once per seed. Runs are grouped into priority
tiers, so a cut-short queue still covers the most important comparisons, and within a tier
every configuration gets its first seed before any gets its second.

A run counts as done when its run directory, with test metrics in ``summary.json``, is in
the backup folder, or when W&B has a finished run of that name with test metrics. Runs that
are done only in W&B have no checkpoints; a final tier trains them again under the
``<tag>_rerun`` tag to recover the checkpoints.

Each finished run directory is copied to ``<backup_root>/<tag>/checkpoints/<run>/`` (permanent
storage that outlives the pod), and the status file and logs to ``<backup_root>/<tag>/queue/``.

Usage (from the repository root, always through ``-m``):
    python -m scripts.run_queue --tag showcase_runs --dry_run
    python -m scripts.run_queue --tag showcase_runs --tiers 1 --seeds 42 --epochs 1
    python -m scripts.run_queue --tag showcase_runs --max_hours 20

Progress: ``outputs/queue/<tag>/status.txt``; one log per run in ``outputs/queue/<tag>/logs/``.
``touch outputs/queue/<tag>/STOP`` stops the queue after the current run; Ctrl-C stops the
current run (its weights are saved) and a second Ctrl-C kills it. Launching the same command
again skips finished runs.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Iterable, Optional, Sequence

from models.variant import MPSVariant
from utils.run_logger import RUN_SUMMARY_FILE
from utils.run_spec import CHECKPOINT_ROOT, RunSpec

REPO_ROOT = Path(__file__).resolve().parents[1]
QUEUE_ROOT = Path("outputs") / "queue"
DEFAULT_BACKUP_ROOT = Path("/workspace/hybrid-qtn")
DATASET = "kvasir_seg"
DEFAULT_SEEDS = (42, 43, 44)
DEFAULT_EPOCHS = 100
FOCAL_TVERSKY = "focal_tversky"
BCE_DICE = "bce_dice"
BOND_DIMS = (32, 64)
# (axial operator, boundary) pairs of the grid; the mean boundary needs the relative operator.
OPERATOR_CHOICES = (("relative", "zero"), ("absolute", "zero"), ("circular", "zero"), ("relative", "mean"))
OPERATOR_LABELS = {"relative": "rel", "absolute": "abs", "circular": "circ"}
RERUN_SUFFIX = "_rerun"
RERUN_TIER = 9
DONE_METRIC = "test_best_live_dice"
QUEUE_RECORD_FILE = "queue_record.json"
STATUS_REFRESH_SECONDS = 30
LOG_TAIL_BYTES = 64 * 1024
CHILD_ENVIRONMENT = {"PYTHONUNBUFFERED": "1", "TQDM_MININTERVAL": "5"}


# --------------------------------------------------------------------------- grid and tiers


@dataclass(frozen=True)
class Config:
    """One grid configuration; trained once per seed."""

    model: str
    bond_dim: int = 32
    variant: MPSVariant = field(default_factory=MPSVariant)
    loss: str = FOCAL_TVERSKY

    def spec(self, seed: int, tag: str) -> RunSpec:
        return RunSpec(
            model_type=self.model, datasets=(DATASET,), loss=self.loss, seed=seed,
            bond_dim=self.bond_dim, variant=self.variant, tag=tag,
        )

    def with_loss(self, loss: str) -> "Config":
        return Config(self.model, self.bond_dim, self.variant, loss)

    @property
    def label(self) -> str:
        """Short description, e.g. ``hybrid b32 rel no-PE`` or ``vanilla (bce_dice)``."""
        loss = "" if self.loss == FOCAL_TVERSKY else f" ({self.loss})"
        if self.model == "vanilla":
            return f"vanilla{loss}"
        operator = "mean" if self.variant.boundary == "mean" else OPERATOR_LABELS[self.variant.axial_operator]
        encoding = "PE" if self.variant.positional_encoding else "no-PE"
        return f"hybrid b{self.bond_dim} {operator} {encoding}{loss}"


VANILLA = Config("vanilla")


def hybrid(bond_dim: int, operator: str = "relative", positional_encoding: bool = False, boundary: str = "zero") -> Config:
    return Config("hybrid", bond_dim, MPSVariant(operator, positional_encoding, boundary))


def full_grid() -> list[Config]:
    """All 34 configurations: vanilla and every hybrid variant, with both losses."""
    architectures = [VANILLA] + [
        hybrid(bond_dim, operator, encoding, boundary)
        for bond_dim in BOND_DIMS
        for operator, boundary in OPERATOR_CHOICES
        for encoding in (False, True)
    ]
    return [architecture.with_loss(loss) for loss in (FOCAL_TVERSKY, BCE_DICE) for architecture in architectures]


@dataclass(frozen=True)
class Tier:
    number: int
    description: str
    configs: tuple[Config, ...]


def build_tiers() -> list[Tier]:
    """Priority tiers 1-8. Tier 8 is the rest of the grid, so no configuration can be missing."""
    tiers = [
        Tier(1, "vanilla, hybrid b32 rel no-PE", (VANILLA, hybrid(32))),
        Tier(2, "hybrid b64 rel no-PE", (hybrid(64),)),
        Tier(3, "hybrid b32: rel+PE, abs+PE, abs no-PE", (hybrid(32, positional_encoding=True), hybrid(32, "absolute", True), hybrid(32, "absolute"))),
        Tier(4, "hybrid b64: rel+PE, abs+PE, abs no-PE", (hybrid(64, positional_encoding=True), hybrid(64, "absolute", True), hybrid(64, "absolute"))),
        Tier(5, "hybrid b32: circular, mean (no PE)", (hybrid(32, "circular"), hybrid(32, boundary="mean"))),
        Tier(6, "bce_dice: vanilla, hybrid b32 rel no-PE", (VANILLA.with_loss(BCE_DICE), hybrid(32).with_loss(BCE_DICE))),
        Tier(
            7,
            "circular and mean: b32 with PE, b64 with and without PE",
            (
                hybrid(32, "circular", True), hybrid(32, boundary="mean", positional_encoding=True),
                hybrid(64, "circular"), hybrid(64, "circular", True),
                hybrid(64, boundary="mean"), hybrid(64, boundary="mean", positional_encoding=True),
            ),
        ),
    ]
    queued = {config for tier in tiers for config in tier.configs}
    rest = tuple(config for config in full_grid() if config not in queued)
    tiers.append(Tier(8, f"bce_dice for the other {len(rest)} hybrid configs", rest))
    return tiers


# --------------------------------------------------------------------------- queue items


@dataclass(frozen=True)
class QueueItem:
    tier: int
    config: Config
    seed: int
    tag: str

    @property
    def spec(self) -> RunSpec:
        return self.config.spec(self.seed, self.tag)

    @property
    def run_name(self) -> str:
        return self.spec.run_name

    @property
    def is_rerun(self) -> bool:
        return self.tier == RERUN_TIER

    def train_arguments(self, epochs: int) -> list[str]:
        """Options for ``train.py``; they resolve to :attr:`spec` through :meth:`RunSpec.from_arguments`."""
        config = self.config
        arguments = [
            "--model", config.model, "--dataset", DATASET, "--loss", config.loss, "--seed", str(self.seed),
            "--tag", self.tag, "--epochs", str(epochs),
        ]
        if config.model != "vanilla":
            variant = config.variant
            arguments += [
                "--bond_dim", str(config.bond_dim),
                "--axial_operator", variant.axial_operator,
                "--axial_boundary", variant.boundary,
                "--positional_encoding" if variant.positional_encoding else "--no-positional_encoding",
            ]
        return arguments


def queue_tag(tag: str, epochs: int) -> str:
    """The tag of every run in this queue; shortened runs get ``_e<epochs>`` so they never pass for full ones."""
    return tag if epochs == DEFAULT_EPOCHS else f"{tag}_e{epochs}"


def build_queue(tag: str, seeds: Sequence[int] = DEFAULT_SEEDS, tiers: Optional[Iterable[int]] = None) -> list[QueueItem]:
    """Items of tiers 1-8 in priority order: tier by tier, and within a tier seed by seed.

    Raises:
        ValueError: If two items share a run name.
    """
    selected = set(tiers) if tiers is not None else None
    items = [
        QueueItem(tier.number, config, seed, tag)
        for tier in build_tiers()
        if selected is None or tier.number in selected
        for seed in seeds
        for config in tier.configs
    ]
    names = [item.run_name for item in items]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate run names in the queue: {duplicates}")
    return items


def rerun_item(item: QueueItem) -> QueueItem:
    return QueueItem(RERUN_TIER, item.config, item.seed, item.tag + RERUN_SUFFIX)


# --------------------------------------------------------------------------- paths and state


@dataclass(frozen=True)
class QueuePaths:
    tag: str
    backup_root: Optional[Path]

    @property
    def queue_dir(self) -> Path:
        return REPO_ROOT / QUEUE_ROOT / self.tag

    @property
    def logs_dir(self) -> Path:
        return self.queue_dir / "logs"

    @property
    def status_file(self) -> Path:
        return self.queue_dir / "status.txt"

    @property
    def stop_file(self) -> Path:
        return self.queue_dir / "STOP"

    @property
    def lock_file(self) -> Path:
        return self.queue_dir / ".lock"

    def log_file(self, item: QueueItem) -> Path:
        return self.logs_dir / f"{item.run_name}.log"

    def local_run_dir(self, item: QueueItem) -> Path:
        return REPO_ROOT / CHECKPOINT_ROOT / item.run_name

    @property
    def backup_dir(self) -> Optional[Path]:
        return self.backup_root / self.tag if self.backup_root else None

    def backup_run_dir(self, item: QueueItem) -> Optional[Path]:
        return self.backup_dir / "checkpoints" / item.run_name if self.backup_dir else None

    @property
    def backup_queue_dir(self) -> Optional[Path]:
        return self.backup_dir / "queue" if self.backup_dir else None


PENDING = "pending"
RUNNING = "running"
DONE = "done"
DONE_NOT_BACKED_UP = "done*"
WANDB_ONLY = "wandb"
FAILED = "failed"
INTERRUPTED = "interrupted"

STATUS_LEGEND = (
    "done = results and checkpoints in the backup folder; done* = finished but not yet copied to the backup folder; "
    "wandb = finished in W&B only, no checkpoints (retrained in tier 9)"
)


def read_json(path: Optional[Path]) -> Optional[dict]:
    if path is None or not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def completed_summary(run_dir: Optional[Path]) -> Optional[dict]:
    """The run's summary if it finished with test metrics, else ``None``."""
    summary = read_json(run_dir / RUN_SUMMARY_FILE) if run_dir else None
    return summary if summary and DONE_METRIC in summary else None


@dataclass
class RunState:
    item: QueueItem
    status: str = PENDING
    summary: dict = field(default_factory=dict)
    duration_s: Optional[float] = None
    started_at: Optional[float] = None
    note: str = ""


def resolve_state(item: QueueItem, paths: QueuePaths, wandb_summaries: dict[str, dict]) -> RunState:
    """Status from the backup folder, then the local run directory, then W&B (original runs only)."""
    backup_dir = paths.backup_run_dir(item)
    if summary := completed_summary(backup_dir):
        record = read_json(backup_dir / QUEUE_RECORD_FILE) or {}
        return RunState(item, DONE, summary, record.get("duration_s"))
    local_dir = paths.local_run_dir(item)
    if summary := completed_summary(local_dir):
        record = read_json(local_dir / QUEUE_RECORD_FILE) or {}
        return RunState(item, DONE_NOT_BACKED_UP, summary, record.get("duration_s"))
    if not item.is_rerun and item.run_name in wandb_summaries:
        return RunState(item, WANDB_ONLY, wandb_summaries[item.run_name])
    return RunState(item, PENDING)


def resolve_queue(
    items: list[QueueItem],
    paths: QueuePaths,
    wandb_summaries: dict[str, dict],
    rerun_candidates: Iterable[QueueItem] = (),
) -> list[RunState]:
    """States of ``items``, followed by a tier-9 rerun of every candidate that is done only in W&B."""
    states = [resolve_state(item, paths, wandb_summaries) for item in items]
    reruns = [
        rerun_item(item) for item in rerun_candidates
        if resolve_state(item, paths, wandb_summaries).status == WANDB_ONLY
    ]
    return states + [resolve_state(item, paths, wandb_summaries) for item in reruns]


def finished_wandb_summaries(tag: str, entity: Optional[str]) -> dict[str, dict]:
    """Summary of every finished W&B run named ``...__<tag>`` that has test metrics."""
    from utils.wandb_export import runs_with_name_suffix

    summaries = {}
    for run in runs_with_name_suffix([tag], entity=entity):
        summary = {key: value for key, value in run.summary.items() if isinstance(value, (int, float))}
        if run.state == "finished" and DONE_METRIC in summary:
            summaries[run.name] = summary
    return summaries


# --------------------------------------------------------------------------- status file


def format_duration(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    return f"{minutes}m{seconds:02d}s"


def format_metric(summary: dict, key: str) -> str:
    value = summary.get(key)
    return f"{value:.4f}" if isinstance(value, (int, float)) else "-"


def mean_duration(states: list[RunState]) -> Optional[float]:
    durations = [state.duration_s for state in states if state.duration_s and state.status in (DONE, DONE_NOT_BACKED_UP)]
    return mean(durations) if durations else None


def run_progress(log_file: Path) -> str:
    """Current phase of a run read from the end of its log, e.g. ``epoch 37/100``."""
    try:
        with log_file.open("rb") as log:
            log.seek(max(log_file.stat().st_size - LOG_TAIL_BYTES, 0))
            tail = log.read().decode(errors="replace")
    except OSError:
        return "starting"
    phases = [(match.end(), f"epoch {match[1]}/{match[2]}") for match in re.finditer(r"Epoch (\d+)/(\d+) \[Train\]", tail)]
    phases += [(match.end(), "SWA validation") for match in re.finditer(r"SWA \[Val\]", tail)]
    phases += [(match.end(), "testing") for match in re.finditer(r"\[Test\]", tail)]
    return max(phases)[1] if phases else "starting"


@dataclass
class Launch:
    tag: str
    started: float
    max_hours: Optional[float]
    epochs: int
    git: str
    state: str = "running"


def render_status(launch: Launch, states: list[RunState], paths: QueuePaths, current: Optional[RunState] = None) -> str:
    now = time.time()
    counts = {status: sum(state.status == status for state in states) for status in (DONE, DONE_NOT_BACKED_UP, WANDB_ONLY, FAILED, PENDING)}
    finished = counts[DONE] + counts[DONE_NOT_BACKED_UP] + counts[WANDB_ONLY]
    pending = len(states) - finished
    average = mean_duration(states)
    launched = datetime.fromtimestamp(launch.started).strftime("%Y-%m-%d %H:%M")
    budget = f", budget {launch.max_hours:g}h" if launch.max_hours else ""
    estimate = (
        f"{format_duration(average)} per run (mean of {sum(1 for s in states if s.duration_s and s.status in (DONE, DONE_NOT_BACKED_UP))}) · "
        f"~{format_duration(average * pending)} for the {pending} runs left"
        if average else "available after the first run finishes"
    )
    lines = [
        f"Queue:    {launch.tag}  ({len(states)} runs, {launch.epochs} epoch{'s' if launch.epochs != 1 else ''} each)",
        f"State:    {launch.state}",
        f"Updated:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Launched: {launched} ({format_duration(now - launch.started)} ago{budget})",
        f"Code:     {launch.git}",
        f"Progress: {finished}/{len(states)} finished ({counts[WANDB_ONLY]} in W&B only) · "
        f"{1 if current else 0} running · {counts[FAILED]} failed · {counts[PENDING] - (1 if current else 0)} pending",
        f"Estimate: {estimate}",
    ]
    if current:
        elapsed = format_duration(now - current.started_at) if current.started_at else "-"
        lines += [
            f"Current:  [tier {current.item.tier}] {current.item.run_name}",
            f"          {run_progress(paths.log_file(current.item))}, running for {elapsed}",
            f"Log:      {paths.log_file(current.item).relative_to(REPO_ROOT)}",
        ]
    lines.append(f"Backup:   {paths.backup_dir or 'disabled'}")
    lines += ["", "Tiers:"]
    descriptions = {tier.number: tier.description for tier in build_tiers()}
    descriptions[RERUN_TIER] = "reruns of runs done only in W&B, for their checkpoints"
    for number in sorted({state.item.tier for state in states}):
        tier_states = [state for state in states if state.item.tier == number]
        done = sum(state.status in (DONE, DONE_NOT_BACKED_UP, WANDB_ONLY) for state in tier_states)
        lines.append(f"  tier {number}  {done:>3}/{len(tier_states):<3} {descriptions[number]}")
    lines += ["", f"{'#':>3}  {'tier':>4}  {'status':<11} {'time':>7}  {'swa dice':>8}  {'live dice':>9}  run"]
    for index, state in enumerate(states, start=1):
        status = RUNNING if state is current else state.status
        duration = format_duration(state.duration_s) if state.duration_s and state is not current else "-"
        lines.append(
            f"{index:>3}  {state.item.tier:>4}  {status:<11} {duration:>7}  "
            f"{format_metric(state.summary, 'test_swa_dice'):>8}  {format_metric(state.summary, 'test_best_live_dice'):>9}  "
            f"{state.item.run_name}{'  ' + state.note if state.note else ''}"
        )
    lines += ["", STATUS_LEGEND, f"Stop after the current run: touch {paths.stop_file.relative_to(REPO_ROOT)}"]
    return "\n".join(lines) + "\n"


def write_text_atomically(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


# --------------------------------------------------------------------------- backup


def copy_tree_atomically(source: Path, destination: Path) -> None:
    """Copy ``source`` to ``destination`` through a temporary name, replacing any earlier copy."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    shutil.rmtree(temporary, ignore_errors=True)
    shutil.copytree(source, temporary)
    for file in source.rglob("*"):
        if file.is_file() and (temporary / file.relative_to(source)).stat().st_size != file.stat().st_size:
            raise OSError(f"Size mismatch after copying {file}")
    shutil.rmtree(destination, ignore_errors=True)
    temporary.rename(destination)


def back_up_run(state: RunState, paths: QueuePaths) -> None:
    """Copy a finished run directory to the backup folder; on failure the run stays ``done*``."""
    backup_dir = paths.backup_run_dir(state.item)
    if backup_dir is None:
        return
    try:
        copy_tree_atomically(paths.local_run_dir(state.item), backup_dir)
        state.status, state.note = DONE, ""
    except OSError as error:
        state.status, state.note = DONE_NOT_BACKED_UP, f"(backup failed: {error})"


def back_up_queue_files(paths: QueuePaths, log_file: Optional[Path] = None) -> None:
    """Copy the status file and one log (or all logs) to the backup folder; errors are ignored."""
    if paths.backup_queue_dir is None:
        return
    try:
        (paths.backup_queue_dir / "logs").mkdir(parents=True, exist_ok=True)
        if paths.status_file.exists():
            shutil.copy2(paths.status_file, paths.backup_queue_dir / "status.txt")
        logs = [log_file] if log_file else list(paths.logs_dir.glob("*.log"))
        for log in logs:
            if log and log.exists():
                shutil.copy2(log, paths.backup_queue_dir / "logs" / log.name)
    except OSError as error:
        print(f"Warning: could not copy queue files to {paths.backup_queue_dir}: {error}", file=sys.stderr)


# --------------------------------------------------------------------------- running


def git_description() -> str:
    def git(*arguments: str) -> str:
        result = subprocess.run(["git", *arguments], cwd=REPO_ROOT, capture_output=True, text=True)
        return result.stdout.strip()

    commit = git("rev-parse", "--short", "HEAD") or "unknown"
    dirty = " + uncommitted changes" if git("status", "--porcelain", "--untracked-files=no") else ""
    return f"{git('rev-parse', '--abbrev-ref', 'HEAD')} @ {commit}{dirty}"


def gpu_name() -> str:
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True)
        return result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    except OSError:
        return ""


class Interrupted(Exception):
    """Ctrl-C or SIGTERM reached the queue."""


def raise_interrupted(*_) -> None:
    raise Interrupted


def run_one(state: RunState, launch: Launch, states: list[RunState], paths: QueuePaths) -> bool:
    """Train one item; returns ``False`` if the queue was interrupted during it."""
    item = state.item
    local_dir = paths.local_run_dir(item)
    shutil.rmtree(local_dir, ignore_errors=True)  # a leftover of an unfinished attempt
    log_file = paths.log_file(item)
    if log_file.exists():
        previous = paths.logs_dir / "previous" / f"{log_file.stem}.{datetime.now():%Y%m%d-%H%M%S}.log"
        previous.parent.mkdir(parents=True, exist_ok=True)
        log_file.rename(previous)

    command = [sys.executable, "train.py", *item.train_arguments(launch.epochs)]
    started = time.time()
    state.started_at, state.duration_s, state.note = started, None, ""
    environment = {**os.environ, **CHILD_ENVIRONMENT}
    interrupted = False
    with log_file.open("w") as log:
        log.write(f"# {datetime.now():%Y-%m-%d %H:%M:%S}  tier {item.tier}  {' '.join(command)}\n")
        log.flush()
        # A session of its own: Ctrl-C reaches only the queue, which forwards one SIGINT.
        process = subprocess.Popen(
            command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, env=environment, start_new_session=True
        )
        while process.poll() is None:
            try:
                write_text_atomically(paths.status_file, render_status(launch, states, paths, current=state))
                process.wait(timeout=STATUS_REFRESH_SECONDS)
            except subprocess.TimeoutExpired:
                continue
            except Interrupted:
                if interrupted:
                    os.killpg(process.pid, signal.SIGKILL)
                    break
                interrupted = True
                launch.state = "interrupted: stopping the current run (interrupt again to kill it)"
                process.send_signal(signal.SIGINT)
        process.wait()

    finished = time.time()
    state.duration_s = finished - started
    if process.returncode == 0 and completed_summary(local_dir):
        state.summary = completed_summary(local_dir)
        record = {
            "run_name": item.run_name, "tier": item.tier, "seed": item.seed, "tag": item.tag, "epochs": launch.epochs,
            "command": command, "started_at": datetime.fromtimestamp(started).isoformat(timespec="seconds"),
            "finished_at": datetime.fromtimestamp(finished).isoformat(timespec="seconds"),
            "duration_s": state.duration_s, "code": launch.git, "gpu": gpu_name(),
        }
        (local_dir / QUEUE_RECORD_FILE).write_text(json.dumps(record, indent=2))
        state.status = DONE_NOT_BACKED_UP
        back_up_run(state, paths)
    else:
        state.status = INTERRUPTED if interrupted else FAILED
        state.note = f"(exit code {process.returncode}, see log)"
    write_text_atomically(paths.status_file, render_status(launch, states, paths))
    back_up_queue_files(paths, log_file)
    return not interrupted


def run_queue(launch: Launch, states: list[RunState], paths: QueuePaths) -> None:
    signal.signal(signal.SIGINT, raise_interrupted)
    signal.signal(signal.SIGTERM, raise_interrupted)
    budget_s = launch.max_hours * 3600 if launch.max_hours else None
    try:
        for state in states:
            if state.status != PENDING:
                continue
            if paths.stop_file.exists():
                launch.state = "stopped by the STOP file"
                break
            average = mean_duration(states) or 0.0
            if budget_s and time.time() - launch.started + average > budget_s:
                launch.state = f"stopped: the next run would exceed the {launch.max_hours:g}h budget"
                break
            if not run_one(state, launch, states, paths):
                launch.state = "interrupted (Ctrl-C or SIGTERM)"
                break
        else:
            failed = sum(state.status == FAILED for state in states)
            launch.state = f"finished with {failed} failed runs (relaunch to retry them)" if failed else "finished"
    except Interrupted:
        launch.state = "interrupted (Ctrl-C or SIGTERM)"
    finally:
        write_text_atomically(paths.status_file, render_status(launch, states, paths))
        back_up_queue_files(paths)
    print(f"Queue {launch.state}. Status: {paths.status_file.relative_to(REPO_ROOT)}")


def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Kvasir-SEG ablation grid as a resumable queue")
    parser.add_argument("--tag", required=True, help="Run-name suffix of every run, e.g. showcase_runs")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--tiers", type=int, nargs="+", default=None, help=f"Tiers to run (1-8, {RERUN_TIER} = reruns); default all")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Fewer epochs add _e<epochs> to the tag (smoke tests)")
    parser.add_argument("--max_hours", type=float, default=None, help="Start no run that would end after this many hours")
    parser.add_argument("--dry_run", action="store_true", help="Print the queue and exit; writes nothing")
    parser.add_argument("--backup_root", type=Path, default=DEFAULT_BACKUP_ROOT, help="Permanent storage for finished runs")
    parser.add_argument("--no_backup", action="store_true", help="Keep finished runs only in checkpoints/")
    parser.add_argument("--skip_wandb_check", action="store_true", help="Do not look up finished runs in W&B")
    parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity; default: your default entity")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    arguments = parse_arguments(argv)
    os.chdir(REPO_ROOT)
    tag = queue_tag(arguments.tag, arguments.epochs)
    backup_root = None if arguments.no_backup else arguments.backup_root
    if backup_root and not backup_root.parent.is_dir() and not arguments.dry_run:
        sys.exit(f"Backup folder parent {backup_root.parent} does not exist; pass --backup_root or --no_backup.")
    paths = QueuePaths(tag, backup_root)
    selected_tiers = arguments.tiers or [tier.number for tier in build_tiers()] + [RERUN_TIER]
    items = build_queue(tag, arguments.seeds, [tier for tier in selected_tiers if tier != RERUN_TIER])
    rerun_candidates = build_queue(tag, arguments.seeds) if RERUN_TIER in selected_tiers else []

    wandb_summaries: dict[str, dict] = {}
    if not arguments.skip_wandb_check:
        try:
            wandb_summaries = finished_wandb_summaries(tag, arguments.wandb_entity)
            print(f"W&B: {len(wandb_summaries)} finished runs tagged {tag}")
        except Exception as error:
            message = f"Could not look up finished runs in W&B ({type(error).__name__}: {error})."
            if not arguments.dry_run:
                sys.exit(f"{message} Log in with `wandb login`, or pass --skip_wandb_check to retrain them.")
            print(f"Warning: {message}", file=sys.stderr)

    states = resolve_queue(items, paths, wandb_summaries, rerun_candidates)
    launch = Launch(tag, time.time(), arguments.max_hours, arguments.epochs, git_description())

    if arguments.dry_run:
        launch.state = "dry run (nothing started)"
        print(render_status(launch, states, paths))
        next_state = next((state for state in states if state.status == PENDING), None)
        if next_state:
            print("First command: python train.py " + " ".join(next_state.item.train_arguments(arguments.epochs)))
        return

    paths.logs_dir.mkdir(parents=True, exist_ok=True)
    lock = paths.lock_file.open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit(f"Another queue with tag {tag} is already running.")
    if paths.stop_file.exists():
        paths.stop_file.unlink()
        print("Removed a STOP file left from an earlier launch.")
    for state in states:
        if state.status == DONE_NOT_BACKED_UP:
            back_up_run(state, paths)
    run_queue(launch, states, paths)


if __name__ == "__main__":
    main()
