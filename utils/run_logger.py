"""Experiment logging seam: W&B and run-directory files in training runs, an in-memory recorder in tests."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol

import wandb

WANDB_PROJECT = "hybrid-qtn"

RUN_CONFIG_FILE = "config.json"
RUN_HISTORY_FILE = "history.jsonl"
RUN_SUMMARY_FILE = "summary.json"
WANDB_RUN_ID_FILE = "wandb_run_id.txt"

# How a run ended; only a completed run has a final summary.
COMPLETED = "completed"
OUT_OF_TIME = "out_of_time"
INTERRUPTED = "interrupted"


class RunLogger(Protocol):
    def start(self, name: str, config: Mapping[str, Any], resume_after_epoch: Optional[int] = None) -> None:
        """Begin a run, or with ``resume_after_epoch`` continue one whose state was saved after that epoch."""

    def log(self, metrics: Mapping[str, float]) -> None:
        """Append one step of metrics to the run history."""

    def summarize(self, metrics: Mapping[str, float]) -> None:
        """Record final results in the run summary."""

    def finish(self, outcome: str = COMPLETED) -> None:
        """End the session; ``outcome`` is :data:`COMPLETED`, :data:`OUT_OF_TIME` or :data:`INTERRUPTED`."""


class WandbLogger:
    """Logs to W&B. With a run ``directory``, the W&B run id is kept there, so a resumed
    training continues the same W&B run instead of opening a new one.

    Run names are not unique in W&B (restarts and reruns reuse them); the id is.
    """

    def __init__(self, directory: Optional[Path] = None, project: str = WANDB_PROJECT) -> None:
        self.id_file = Path(directory) / WANDB_RUN_ID_FILE if directory is not None else None
        self.project = project

    def start(self, name: str, config: Mapping[str, Any], resume_after_epoch: Optional[int] = None) -> None:
        run_id = self._saved_run_id() if resume_after_epoch is not None else None
        if run_id is not None:
            wandb.init(project=self.project, config=dict(config), name=name, id=run_id, resume="must")
            return
        run = wandb.init(project=self.project, config=dict(config), name=name)
        if self.id_file is not None:
            self.id_file.parent.mkdir(parents=True, exist_ok=True)
            self.id_file.write_text(run.id)

    def _saved_run_id(self) -> Optional[str]:
        if self.id_file is None or not self.id_file.exists():
            return None
        return self.id_file.read_text().strip() or None

    def log(self, metrics: Mapping[str, float]) -> None:
        wandb.log(dict(metrics))

    def summarize(self, metrics: Mapping[str, float]) -> None:
        wandb.run.summary.update(dict(metrics))

    def finish(self, outcome: str = COMPLETED) -> None:
        if outcome == COMPLETED:
            wandb.finish()
            return
        if outcome == OUT_OF_TIME and wandb.run is not None:
            wandb.run.mark_preempting()
        wandb.finish(exit_code=1)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path: Path) -> Optional[dict[str, Any]]:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write through a temporary file, so a reader never sees a half-written file."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str))
    os.replace(temporary, path)


class JsonFileLogger:
    """Mirrors a run into its directory, so results are available without W&B.

    Writes ``config.json`` at start, appends every logged step to ``history.jsonl`` and keeps
    ``summary.json`` current. Starting again in the same directory replaces the previous files,
    so a restarted run never mixes with an earlier attempt. Resuming instead keeps the history
    up to the resumed epoch, so the curves continue without repeated epochs.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self._summary: dict[str, Any] = {}

    def start(self, name: str, config: Mapping[str, Any], resume_after_epoch: Optional[int] = None) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self._summary = {}
        (self.directory / RUN_SUMMARY_FILE).unlink(missing_ok=True)
        if resume_after_epoch is None:
            (self.directory / RUN_HISTORY_FILE).write_text("")
            _write_json(self.directory / RUN_CONFIG_FILE, {"name": name, "started_at": _now(), "config": dict(config)})
            return
        self._rewind_history(resume_after_epoch)
        run_config = _read_json(self.directory / RUN_CONFIG_FILE) or {"name": name, "config": dict(config)}
        run_config["resumed_at"] = [*run_config.get("resumed_at", []), _now()]
        _write_json(self.directory / RUN_CONFIG_FILE, run_config)

    def _rewind_history(self, epoch: int) -> None:
        """Drop everything logged after ``epoch``: an interruption after logging an epoch but
        before saving its state means that epoch runs again and would appear twice."""
        path = self.directory / RUN_HISTORY_FILE
        lines = path.read_text().splitlines() if path.exists() else []
        kept: list[str] = []
        for position, line in enumerate(lines, start=1):
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                break  # a line cut off by a hard kill; nothing valid follows it
            if entry.get("epoch") == epoch:
                kept = lines[:position]
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text("".join(line + "\n" for line in kept))
        os.replace(temporary, path)

    def log(self, metrics: Mapping[str, float]) -> None:
        with (self.directory / RUN_HISTORY_FILE).open("a") as history:
            history.write(json.dumps(dict(metrics), default=str) + "\n")

    def summarize(self, metrics: Mapping[str, float]) -> None:
        self._summary.update(metrics)
        _write_json(self.directory / RUN_SUMMARY_FILE, self._summary)

    def finish(self, outcome: str = COMPLETED) -> None:
        if outcome == COMPLETED:
            _write_json(self.directory / RUN_SUMMARY_FILE, {**self._summary, "finished_at": _now()})
        else:
            # A partial summary (stopped during test reporting) must not pass for a finished run.
            (self.directory / RUN_SUMMARY_FILE).unlink(missing_ok=True)


class CombinedLogger:
    """Forwards every call to each logger in order."""

    def __init__(self, *loggers: RunLogger) -> None:
        self.loggers = loggers

    def start(self, name: str, config: Mapping[str, Any], resume_after_epoch: Optional[int] = None) -> None:
        for run_logger in self.loggers:
            run_logger.start(name, config, resume_after_epoch)

    def log(self, metrics: Mapping[str, float]) -> None:
        for run_logger in self.loggers:
            run_logger.log(metrics)

    def summarize(self, metrics: Mapping[str, float]) -> None:
        for run_logger in self.loggers:
            run_logger.summarize(metrics)

    def finish(self, outcome: str = COMPLETED) -> None:
        for run_logger in self.loggers:
            run_logger.finish(outcome)
