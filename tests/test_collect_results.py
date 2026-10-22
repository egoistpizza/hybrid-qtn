import csv
import json

from scripts.collect_results import config_statistics, main, markdown_table
from scripts.run_queue import QUEUE_RECORD_FILE, build_queue, rerun_item
from utils.run_logger import RUN_CONFIG_FILE, RUN_HISTORY_FILE, RUN_SUMMARY_FILE


def row(config, seed, dice, rerun=False, complete=True):
    return {"config": config, "model": "hybrid", "bond_dim": 32, "axial_operator": "relative", "boundary": "zero",
            "positional_encoding": False, "loss": "focal_tversky", "seed": seed, "rerun": rerun,
            "complete": complete, "test_swa_dice": dice}


def test_statistics_prefer_the_original_run_of_a_seed():
    rows = [row("a", 42, 0.80), row("a", 42, 0.70, rerun=True), row("a", 43, 0.90), row("b", 44, 0.60, rerun=True)]
    by_config = {result["config"]: result for result in config_statistics(rows, ["a", "b"])}
    assert by_config["a"]["n_seeds"] == 2
    assert abs(by_config["a"]["test_swa_dice_mean"] - 0.85) < 1e-9
    assert by_config["b"]["test_swa_dice_mean"] == 0.60  # a rerun counts when there is no original
    assert by_config["b"]["test_swa_dice_std"] == ""


def test_statistics_skip_incomplete_runs():
    rows = [row("a", 42, 0.80), row("a", 43, "", complete=False)]
    assert config_statistics(rows, ["a"])[0]["n_seeds"] == 1


def test_markdown_shows_mean_and_std():
    rows = [row("a", 42, 0.80), row("a", 43, 0.90)]
    assert "0.8500 ± 0.0707" in markdown_table(config_statistics(rows, ["a"]))


def write_run(directory, seed, dice):
    directory.mkdir(parents=True)
    (directory / RUN_CONFIG_FILE).write_text(json.dumps({"config": {"total_params": 15_950_210}}))
    (directory / RUN_SUMMARY_FILE).write_text(json.dumps({"test_best_live_dice": dice, "test_swa_dice": dice}))
    history = [{"epoch": 1, "val_dice": 0.5, "Peak_VRAM_MB": 2048.0}, {"epoch": 2, "val_dice": 0.7, "Peak_VRAM_MB": 3072.0}]
    (directory / RUN_HISTORY_FILE).write_text("\n".join(json.dumps(entry) for entry in history))
    (directory / QUEUE_RECORD_FILE).write_text(json.dumps({"duration_s": 600, "seed": seed}))
    (directory / "best_swa_model.pth").write_bytes(b"weights")


def test_collect_builds_tables_from_backed_up_runs(tmp_path):
    items = [item for item in build_queue("t", seeds=[42, 43], tiers=[1]) if item.config.model == "hybrid"]
    checkpoints = tmp_path / "t" / "checkpoints"
    write_run(checkpoints / items[0].run_name, 42, 0.80)
    write_run(checkpoints / items[1].run_name, 43, 0.90)
    write_run(checkpoints / rerun_item(items[0]).run_name, 42, 0.10)

    main(["--tag", "t", "--seeds", "42", "43", "--backup_root", str(tmp_path), "--skip_wandb"])

    results = tmp_path / "t" / "results"
    per_run = list(csv.DictReader((results / "results_per_run.csv").open()))
    assert [entry["rerun"] for entry in per_run] == ["False", "False", "True"]
    assert per_run[0]["peak_vram_mb"] == "3072.0" and per_run[0]["duration_min"] == "10.0"
    (by_config,) = list(csv.DictReader((results / "results_by_config.csv").open()))
    assert by_config["config"] == "hybrid b32 rel no-PE" and by_config["n_seeds"] == "2"
    assert abs(float(by_config["test_swa_dice_mean"]) - 0.85) < 1e-9
    assert len(list(csv.DictReader((results / "history.csv").open()))) == 6
    assert (results / "environment" / "configs" / "kvasir_seg.yaml").exists()
