import argparse
import json

import wandb


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
