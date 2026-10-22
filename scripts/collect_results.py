"""Gather everything needed for comparison tables, graphs and slides of one queue tag.

Reads the run directories the queue backed up (``<backup_root>/<tag>/checkpoints/``) and the W&B
runs named ``...__<tag>`` or ``...__<tag>_rerun``, then writes to ``<backup_root>/<tag>/results/``:

    results_per_run.csv     one row per run: configuration, test metrics, params, VRAM, time
    results_by_config.csv   mean and std over seeds for each configuration
    results_by_config.md    the main columns of the above as a Markdown table
    history.csv             per-epoch training curves of every run (long format)
    wandb/<run>__<id>.json  W&B config, summary and full history of every run
    queue/                  the queue's status file and logs
    environment/            git commit and diff, configs/, pip freeze, GPU info

A run that was trained again for its checkpoints (``_rerun``) is listed in the per-run table,
but the per-config statistics use the original run of that seed when it has test metrics.

Usage (from the repository root):
    python -m scripts.collect_results --tag showcase_runs
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable, Optional, Sequence

from scripts.run_queue import (
    DEFAULT_BACKUP_ROOT,
    DEFAULT_SEEDS,
    DONE_METRIC,
    QUEUE_RECORD_FILE,
    QUEUE_ROOT,
    REPO_ROOT,
    RERUN_SUFFIX,
    QueueItem,
    build_queue,
    rerun_item,
)
from utils.checkpoints import BEST_LIVE_CHECKPOINT, SWA_CHECKPOINT
from utils.run_logger import RUN_CONFIG_FILE, RUN_HISTORY_FILE, RUN_SUMMARY_FILE

TEST_METRICS = ("dice", "iou", "hd95", "mae", "f2", "loss")
MODEL_PREFIXES = ("test_swa", "test_best_live")
RESOURCE_COLUMNS = ("total_params", "peak_vram_mb", "duration_min", "best_val_dice")
MARKDOWN_COLUMNS = (
    ("test_swa_dice", "Dice", 4), ("test_swa_iou", "IoU", 4), ("test_swa_hd95", "HD95", 2),
    ("test_swa_mae", "MAE", 4), ("test_swa_f2", "F2", 4), ("test_best_live_dice", "Dice (best live)", 4),
    ("total_params", "Params (M)", 2), ("peak_vram_mb", "Peak VRAM (GB)", 2), ("duration_min", "Time (min)", 1),
)


@dataclass
class RunRecord:
    """Everything known about one run, from its backed-up directory and/or W&B."""

    name: str
    item: Optional[QueueItem]
    config: dict = field(default_factory=dict)
    summary: dict = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)
    queue_record: dict = field(default_factory=dict)
    has_checkpoint: bool = False
    wandb_id: str = ""
    wandb_state: str = ""

    @property
    def complete(self) -> bool:
        return DONE_METRIC in self.summary

    def row(self) -> dict[str, Any]:
        item = self.item
        variant = item.config.variant if item and item.config.model != "vanilla" else None
        epoch_rows = [entry for entry in self.history if "epoch" in entry]
        vram = [entry["Peak_VRAM_MB"] for entry in epoch_rows if isinstance(entry.get("Peak_VRAM_MB"), (int, float))]
        val_dice = [entry["val_dice"] for entry in epoch_rows if isinstance(entry.get("val_dice"), (int, float))]
        duration_s = self.queue_record.get("duration_s") or self.summary.get("_runtime")
        row = {
            "run_name": self.name,
            "config": item.config.label if item else "",
            "model": item.config.model if item else self.config.get("model_type", ""),
            "bond_dim": item.config.bond_dim if variant else "",
            "axial_operator": variant.axial_operator if variant else "",
            "boundary": variant.boundary if variant else "",
            "positional_encoding": variant.positional_encoding if variant else "",
            "loss": item.config.loss if item else self.config.get("loss", ""),
            "seed": item.seed if item else self.config.get("seed", ""),
            "rerun": bool(item and item.is_rerun),
            "complete": self.complete,
            "has_checkpoint": self.has_checkpoint,
            "wandb_id": self.wandb_id,
            "wandb_state": self.wandb_state,
            "total_params": self.config.get("total_params", ""),
            "peak_vram_mb": max(vram) if vram else "",
            "duration_min": duration_s / 60 if isinstance(duration_s, (int, float)) else "",
            "best_val_dice": max(val_dice) if val_dice else "",
        }
        for prefix in MODEL_PREFIXES:
            for metric in TEST_METRICS:
                row[f"{prefix}_{metric}"] = self.summary.get(f"{prefix}_{metric}", "")
        return row


def expected_items(tag: str, seeds: Sequence[int]) -> dict[str, QueueItem]:
    """Every run name the queue can produce for ``tag``, originals and reruns."""
    originals = build_queue(tag, seeds)
    return {item.run_name: item for item in originals + [rerun_item(item) for item in originals]}


def read_history(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text()) if path.is_file() else {}
    except json.JSONDecodeError:
        return {}


def local_records(checkpoints_dir: Path, items: dict[str, QueueItem]) -> dict[str, RunRecord]:
    """Records of the run directories the queue backed up."""
    records = {}
    if not checkpoints_dir.is_dir():
        return records
    for run_dir in sorted(path for path in checkpoints_dir.iterdir() if path.is_dir() and not path.name.endswith(".partial")):
        records[run_dir.name] = RunRecord(
            name=run_dir.name,
            item=items.get(run_dir.name),
            config=read_json(run_dir / RUN_CONFIG_FILE).get("config", {}),
            summary=read_json(run_dir / RUN_SUMMARY_FILE),
            history=read_history(run_dir / RUN_HISTORY_FILE),
            queue_record=read_json(run_dir / QUEUE_RECORD_FILE),
            has_checkpoint=(run_dir / SWA_CHECKPOINT).exists() or (run_dir / BEST_LIVE_CHECKPOINT).exists(),
        )
    return records


def wandb_value(value: Any) -> Any:
    """W&B summary values as plain JSON types."""
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    if isinstance(value, dict) or hasattr(value, "items"):
        return {key: wandb_value(inner) for key, inner in value.items()}
    return str(value)


def merge_wandb_runs(records: dict[str, RunRecord], items: dict[str, QueueItem], tag: str, entity: Optional[str], wandb_dir: Path) -> int:
    """Export every W&B run of ``tag`` to ``wandb_dir`` and merge it into ``records``; returns the run count.

    When W&B has several runs of one name (e.g. a crashed attempt), the latest finished one
    with test metrics is used.
    """
    from utils.wandb_export import runs_with_name_suffix

    wandb_dir.mkdir(parents=True, exist_ok=True)
    runs = sorted(runs_with_name_suffix([tag, tag + RERUN_SUFFIX], entity=entity), key=lambda run: run.created_at)
    for run in runs:
        summary = wandb_value(dict(run.summary.items()))
        history = [wandb_value(row) for row in run.scan_history()]
        config = wandb_value(dict(run.config))
        payload = {"name": run.name, "id": run.id, "state": run.state, "created_at": run.created_at, "url": run.url,
                   "config": config, "summary": summary, "history": history}
        (wandb_dir / f"{run.name}__{run.id}.json").write_text(json.dumps(payload, indent=2, default=str))
        print(f"  W&B {run.state:<9} {run.name} ({len(history)} history rows)")

        record = records.get(run.name)
        usable = run.state == "finished" and DONE_METRIC in summary
        if record is None:
            records[run.name] = RunRecord(run.name, items.get(run.name), config, summary, history,
                                          wandb_id=run.id, wandb_state=run.state)
        elif usable or not record.wandb_id:
            record.wandb_id, record.wandb_state = run.id, run.state
            if not record.complete:  # no local results: take W&B's
                record.config, record.summary, record.history = config, summary, history
            if "_runtime" in summary:
                record.summary.setdefault("_runtime", summary["_runtime"])
    return len(runs)


def write_csv(path: Path, rows: list[dict], columns: Optional[Sequence[str]] = None) -> None:
    columns = list(columns or dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def aggregation_rows(rows: list[dict]) -> list[dict]:
    """The rows the per-config statistics use: complete runs, originals preferred over reruns."""
    originals = {(row["config"], row["seed"]) for row in rows if row["complete"] and not row["rerun"]}
    return [row for row in rows if row["complete"] and (not row["rerun"] or (row["config"], row["seed"]) not in originals)]


def numeric_columns() -> list[str]:
    return [f"{prefix}_{metric}" for prefix in MODEL_PREFIXES for metric in TEST_METRICS] + list(RESOURCE_COLUMNS)


def config_statistics(rows: list[dict], config_order: Iterable[str]) -> list[dict]:
    """Mean and sample std over seeds of every numeric column, one row per configuration."""
    grouped: dict[str, list[dict]] = {}
    for row in aggregation_rows(rows):
        grouped.setdefault(row["config"], []).append(row)
    order = [config for config in dict.fromkeys(config_order) if config in grouped]
    order += [config for config in grouped if config not in order]
    statistics = []
    for config in order:
        group = grouped[config]
        first = group[0]
        result = {key: first[key] for key in ("config", "model", "bond_dim", "axial_operator", "boundary", "positional_encoding", "loss")}
        result["n_seeds"] = len(group)
        result["seeds"] = " ".join(str(row["seed"]) for row in sorted(group, key=lambda row: row["seed"]))
        for column in numeric_columns():
            values = [row.get(column) for row in group]
            values = [value for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)]
            result[f"{column}_mean"] = mean(values) if values else ""
            result[f"{column}_std"] = stdev(values) if len(values) > 1 else ""
        statistics.append(result)
    return statistics


def markdown_table(statistics: list[dict]) -> str:
    scale = {"total_params": 1e-6, "peak_vram_mb": 1 / 1024}
    header = ["Configuration", "Loss", "Seeds"] + [title for _, title, _ in MARKDOWN_COLUMNS]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in statistics:
        cells = [row["config"].replace(f" ({row['loss']})", ""), row["loss"], str(row["n_seeds"])]
        for column, _, digits in MARKDOWN_COLUMNS:
            value, spread = row[f"{column}_mean"], row[f"{column}_std"]
            factor = scale.get(column, 1)
            if value == "":
                cells.append("-")
            elif spread == "":
                cells.append(f"{value * factor:.{digits}f}")
            else:
                cells.append(f"{value * factor:.{digits}f} ± {spread * factor:.{digits}f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def history_rows(records: Iterable[RunRecord]) -> list[dict]:
    rows = []
    for record in records:
        for entry in record.history:
            if "epoch" in entry:
                rows.append({"run_name": record.name, "config": record.item.config.label if record.item else "",
                             "seed": record.item.seed if record.item else "", **entry})
    return rows


def capture(command: Sequence[str]) -> str:
    try:
        result = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True)
        return result.stdout + result.stderr
    except OSError as error:
        return f"{command[0]} unavailable: {error}\n"


def write_environment(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "git.txt").write_text(
        capture(["git", "rev-parse", "--abbrev-ref", "HEAD"]) + capture(["git", "rev-parse", "HEAD"])
        + capture(["git", "status", "--short"]) + "\n" + capture(["git", "log", "--oneline", "-20"])
    )
    (directory / "git_diff.patch").write_text(capture(["git", "diff", "HEAD"]))
    (directory / "pip_freeze.txt").write_text(capture([sys.executable, "-m", "pip", "freeze"]))
    (directory / "nvidia_smi.txt").write_text(capture(["nvidia-smi"]))
    (directory / "python.txt").write_text(f"{sys.version}\n{platform.platform()}\n")
    shutil.copytree(REPO_ROOT / "configs", directory / "configs", dirs_exist_ok=True)


def copy_queue_files(tag: str, destination: Path) -> None:
    """Refresh the backed-up queue files from the local queue folder, if this pod has one."""
    source = REPO_ROOT / QUEUE_ROOT / tag
    if source.is_dir():
        shutil.copytree(source, destination, dirs_exist_ok=True, ignore=shutil.ignore_patterns(".lock", "*.tmp", "STOP"))


def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gather results, logs and environment of one queue tag")
    parser.add_argument("--tag", required=True, help="Queue tag, e.g. showcase_runs (or showcase_runs_e1)")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--backup_root", type=Path, default=DEFAULT_BACKUP_ROOT)
    parser.add_argument("--output_dir", type=Path, default=None, help="Default: <backup_root>/<tag>/results")
    parser.add_argument("--skip_wandb", action="store_true", help="Use only the backed-up run directories")
    parser.add_argument("--wandb_entity", type=str, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    arguments = parse_arguments(argv)
    tag_dir = arguments.backup_root / arguments.tag
    output_dir = arguments.output_dir or tag_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    items = expected_items(arguments.tag, arguments.seeds)

    records = local_records(tag_dir / "checkpoints", items)
    print(f"Backed-up run directories: {len(records)} in {tag_dir / 'checkpoints'}")
    if not arguments.skip_wandb:
        count = merge_wandb_runs(records, items, arguments.tag, arguments.wandb_entity, output_dir / "wandb")
        print(f"W&B runs exported: {count}")

    order = {name: index for index, name in enumerate(items)}
    ordered = sorted(records.values(), key=lambda record: (order.get(record.name, len(order)), record.name))
    rows = [record.row() for record in ordered]
    write_csv(output_dir / "results_per_run.csv", rows)
    statistics = config_statistics(rows, (item.config.label for item in items.values()))
    write_csv(output_dir / "results_by_config.csv", statistics)
    (output_dir / "results_by_config.md").write_text(markdown_table(statistics))
    write_csv(output_dir / "history.csv", history_rows(ordered))
    copy_queue_files(arguments.tag, output_dir / "queue")
    write_environment(output_dir / "environment")

    complete = sum(row["complete"] for row in rows)
    missing = sorted(name for name, item in items.items() if not item.is_rerun and name not in records)
    print(f"Runs: {len(rows)} found, {complete} with test metrics, {sum(row['has_checkpoint'] for row in rows)} with checkpoints")
    print(f"Configurations with results: {len(statistics)}; grid runs not found: {len(missing)}")
    print(f"Written to {output_dir}")


if __name__ == "__main__":
    main()
