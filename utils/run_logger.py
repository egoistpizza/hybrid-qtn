"""Experiment logging seam: W&B in training runs, an in-memory recorder in tests."""

from __future__ import annotations

from typing import Any, Mapping, Protocol

import wandb

WANDB_PROJECT = "hybrid-qtn"


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
