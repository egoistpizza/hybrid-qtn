"""Experiment logging seam: W&B and run-directory files in training runs, an in-memory recorder in tests."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol

import wandb

WANDB_PROJECT = "hybrid-qtn"

RUN_CONFIG_FILE = "config.json"
RUN_HISTORY_FILE = "history.jsonl"
RUN_SUMMARY_FILE = "summary.json"


class RunLogger(Protocol):
    def start(self, name: str, config: Mapping[str, Any]) -> None: ...

    def log(self, metrics: Mapping[str, float]) -> None:
        """Append one step of metrics to the run history."""

    def summarize(self, metrics: Mapping[str, float]) -> None:
        """Record final results in the run summary."""

    def finish(self) -> None: ...


class WandbLogger:
    def __init__(self, project: str = WANDB_PROJECT) -> None:
        self.project = project

    def start(self, name: str, config: Mapping[str, Any]) -> None:
        wandb.init(project=self.project, config=dict(config), name=name)

    def log(self, metrics: Mapping[str, float]) -> None:
        wandb.log(dict(metrics))

    def summarize(self, metrics: Mapping[str, float]) -> None:
        wandb.run.summary.update(dict(metrics))

    def finish(self) -> None:
        wandb.finish()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write through a temporary file, so a reader never sees a half-written file."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str))
    os.replace(temporary, path)


class JsonFileLogger:
    """Mirrors a run into its directory, so results are available without W&B.

    Writes ``config.json`` at start, appends every logged step to ``history.jsonl`` and keeps
    ``summary.json`` current. Starting again in the same directory replaces the previous files,
    so a restarted run never mixes with an earlier attempt.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self._summary: dict[str, Any] = {}

    def start(self, name: str, config: Mapping[str, Any]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self._summary = {}
        (self.directory / RUN_HISTORY_FILE).write_text("")
        (self.directory / RUN_SUMMARY_FILE).unlink(missing_ok=True)
        _write_json(self.directory / RUN_CONFIG_FILE, {"name": name, "started_at": _now(), "config": dict(config)})

    def log(self, metrics: Mapping[str, float]) -> None:
        with (self.directory / RUN_HISTORY_FILE).open("a") as history:
            history.write(json.dumps(dict(metrics), default=str) + "\n")

    def summarize(self, metrics: Mapping[str, float]) -> None:
        self._summary.update(metrics)
        _write_json(self.directory / RUN_SUMMARY_FILE, self._summary)

    def finish(self) -> None:
        _write_json(self.directory / RUN_SUMMARY_FILE, {**self._summary, "finished_at": _now()})


class CombinedLogger:
    """Forwards every call to each logger in order."""

    def __init__(self, *loggers: RunLogger) -> None:
        self.loggers = loggers

    def start(self, name: str, config: Mapping[str, Any]) -> None:
        for run_logger in self.loggers:
            run_logger.start(name, config)

    def log(self, metrics: Mapping[str, float]) -> None:
        for run_logger in self.loggers:
            run_logger.log(metrics)

    def summarize(self, metrics: Mapping[str, float]) -> None:
        for run_logger in self.loggers:
            run_logger.summarize(metrics)

    def finish(self) -> None:
        for run_logger in self.loggers:
            run_logger.finish()
