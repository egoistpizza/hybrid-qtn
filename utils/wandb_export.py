"""Find W&B runs by name and export the configuration, summary and full history of a run to JSON."""

import argparse
import json
import re
from typing import Iterable, Optional

import wandb

from utils.run_logger import WANDB_PROJECT


def runs_with_name_suffix(suffixes: Iterable[str], project: str = WANDB_PROJECT, entity: Optional[str] = None) -> list:
    """Every run of ``project`` whose name ends in ``__<suffix>`` for one of ``suffixes``.

    ``entity`` defaults to the logged-in user's default entity.

    Raises:
        wandb.errors.CommError: If W&B cannot be reached or the login is missing.
    """
    api = wandb.Api(timeout=60)
    entity = entity or api.default_entity
    pattern = "__(" + "|".join(re.escape(suffix) for suffix in suffixes) + ")$"
    return list(api.runs(f"{entity}/{project}", filters={"display_name": {"$regex": pattern}}, per_page=200))


def export_run(entity: str, project: str, run_id: str, output: str) -> None:
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/runs/{run_id}")

    print(f"Run: {run.name} ({run.id})")
    print(f"State: {run.state}")
    print(f"Config: {json.dumps(run.config, indent=2)}")

    history = []
    for row in run.scan_history():
        history.append(row)

    summary = {
        "name": run.name,
        "id": run.id,
        "state": run.state,
        "config": run.config,
        "summary": dict(run.summary),
        "history": history,
    }

    with open(output, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"Exported {len(history)} rows to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export a W&B run to JSON")
    parser.add_argument("--entity", type=str, required=True, help="W&B entity")
    parser.add_argument("--project", type=str, default="hybrid-qtn", help="W&B project name")
    parser.add_argument("--run-id", type=str, required=True, help="W&B run ID")
    parser.add_argument("--output", type=str, default="wandb_export.json", help="Output JSON file path")
    args = parser.parse_args()

    export_run(args.entity, args.project, args.run_id, args.output)
