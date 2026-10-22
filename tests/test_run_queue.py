import argparse
import json
import textwrap
from collections import Counter

import pytest

import scripts.run_queue as run_queue
from scripts.run_queue import (
    DONE,
    DONE_METRIC,
    DONE_NOT_BACKED_UP,
    FAILED,
    PENDING,
    QUEUE_RECORD_FILE,
    RERUN_TIER,
    WANDB_ONLY,
    Launch,
    QueuePaths,
    RunState,
    build_queue,
    build_tiers,
    full_grid,
    queue_tag,
    render_status,
    resolve_queue,
)
from utils.run_logger import RUN_SUMMARY_FILE
from utils.run_spec import RunSpec

# Runs that finished before the pod crash (W&B only).
FINISHED_BEFORE = [
    "kvasir_seg__vanilla_unet__focal_tversky__seed42__showcase_runs",
    "kvasir_seg__vanilla_unet__focal_tversky__seed43__showcase_runs",
    "kvasir_seg__vanilla_unet__focal_tversky__seed44__showcase_runs",
    "kvasir_seg__hybrid_unet_b32_rel_nope__focal_tversky__seed42__showcase_runs",
    "kvasir_seg__hybrid_unet_b32_rel_nope__focal_tversky__seed43__showcase_runs",
    "kvasir_seg__hybrid_unet_b32_rel_nope__focal_tversky__seed44__showcase_runs",
    "kvasir_seg__hybrid_unet_b64_rel_nope__focal_tversky__seed42__showcase_runs",
]


def test_grid_has_34_distinct_configurations():
    grid = full_grid()
    assert len(grid) == len(set(grid)) == 34


def test_tiers_cover_the_grid_exactly_once():
    queued = [config for tier in build_tiers() for config in tier.configs]
    assert Counter(queued) == Counter(full_grid())


def test_queue_has_102_runs_in_the_planned_tiers():
    items = build_queue("showcase_runs")
    assert len(items) == len({item.run_name for item in items}) == 102
    assert [Counter(item.tier for item in items)[tier] for tier in range(1, 9)] == [6, 3, 9, 9, 6, 6, 18, 45]
    assert all(item.config.loss == "bce_dice" for item in items if item.tier == 8)


def test_each_tier_runs_every_config_once_before_the_next_seed():
    tier_one = [item for item in build_queue("t") if item.tier == 1]
    assert [item.seed for item in tier_one] == [42, 42, 43, 43, 44, 44]


def test_queue_names_match_the_runs_finished_before():
    names = {item.run_name for item in build_queue("showcase_runs")}
    assert set(FINISHED_BEFORE) <= names


@pytest.mark.parametrize("item", build_queue("showcase_runs"), ids=lambda item: item.run_name)
def test_train_arguments_select_the_queued_run(item):
    parser = argparse.ArgumentParser()
    RunSpec.add_arguments(parser)
    parser.add_argument("--epochs", type=int)
    arguments = parser.parse_args(item.train_arguments(epochs=100))
    assert RunSpec.from_arguments(arguments) == item.spec
    assert arguments.epochs == 100


def test_shortened_runs_get_their_own_tag():
    assert queue_tag("showcase_runs", 100) == "showcase_runs"
    assert queue_tag("showcase_runs", 1) == "showcase_runs_e1"


def write_run(directory, metric=0.9, record=None):
    directory.mkdir(parents=True)
    (directory / RUN_SUMMARY_FILE).write_text(json.dumps({DONE_METRIC: metric, "test_swa_dice": metric}))
    if record:
        (directory / QUEUE_RECORD_FILE).write_text(json.dumps(record))


@pytest.fixture
def repo(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(run_queue, "REPO_ROOT", root)
    return root


def test_resolve_reads_backup_then_local_then_wandb(repo, tmp_path):
    paths = QueuePaths("t", tmp_path / "backup")
    items = build_queue("t", seeds=[42], tiers=[1])
    vanilla, hybrid = items
    write_run(paths.backup_run_dir(vanilla), record={"duration_s": 600})
    write_run(paths.local_run_dir(hybrid))

    states = resolve_queue(items, paths, {})
    assert [state.status for state in states] == [DONE, DONE_NOT_BACKED_UP]
    assert states[0].duration_s == 600


def test_runs_done_only_in_wandb_are_retrained_last(repo, tmp_path):
    paths = QueuePaths("t", tmp_path / "backup")
    items = build_queue("t", seeds=[42], tiers=[1])
    wandb = {items[0].run_name: {DONE_METRIC: 0.8}}

    states = resolve_queue(items, paths, wandb, rerun_candidates=items)
    assert [(state.item.tier, state.status) for state in states] == [(1, WANDB_ONLY), (1, PENDING), (RERUN_TIER, PENDING)]
    rerun = states[-1].item
    assert rerun.run_name == items[0].run_name + "_rerun"

    write_run(paths.backup_run_dir(rerun))
    assert resolve_queue(items, paths, wandb, rerun_candidates=items)[-1].status == DONE


def test_rerun_finished_only_in_wandb_is_still_pending(repo, tmp_path):
    paths = QueuePaths("t", tmp_path / "backup")
    items = build_queue("t", seeds=[42], tiers=[1])
    rerun_name = items[0].run_name + "_rerun"
    states = resolve_queue(items, paths, {items[0].run_name: {DONE_METRIC: 0.8}, rerun_name: {DONE_METRIC: 0.8}}, items)
    assert states[-1].status == PENDING


def test_status_lists_every_run_and_tier(repo, tmp_path):
    paths = QueuePaths("t", tmp_path / "backup")
    states = resolve_queue(build_queue("t"), paths, {})
    text = render_status(Launch("t", 0.0, 20, 100, "main @ abc"), states, paths)
    assert "Progress: 0/102 finished" in text
    assert all(f"tier {tier} " in text for tier in range(1, 9))
    assert all(state.item.run_name in text for state in states)


FAKE_TRAIN = textwrap.dedent(
    """
    import json, os, sys
    from pathlib import Path
    print("Epoch 1/1 [Train]", flush=True)
    if os.environ["FAKE_OUTCOME"] == "fail":
        sys.exit(3)
    run_dir = Path(os.environ["FAKE_RUN_DIR"])
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "best_model.pth").write_bytes(b"weights")
    (run_dir / "summary.json").write_text(json.dumps({"test_best_live_dice": 0.91, "test_swa_dice": 0.92}))
    """
)


@pytest.mark.parametrize("outcome", ["success", "fail"])
def test_run_one_trains_and_backs_up_a_run(repo, tmp_path, monkeypatch, outcome):
    (repo / "train.py").write_text(FAKE_TRAIN)
    paths = QueuePaths("t", tmp_path / "backup")
    paths.logs_dir.mkdir(parents=True)
    item = build_queue("t", seeds=[42], tiers=[1])[0]
    monkeypatch.setenv("FAKE_OUTCOME", outcome)
    monkeypatch.setenv("FAKE_RUN_DIR", str(paths.local_run_dir(item)))
    state = RunState(item)

    assert run_queue.run_one(state, Launch("t", 0.0, None, 1, "main @ abc"), [state], paths)

    assert "Epoch 1/1 [Train]" in paths.log_file(item).read_text()
    assert (paths.backup_queue_dir / "logs" / paths.log_file(item).name).exists()
    if outcome == "success":
        assert state.status == DONE and state.summary["test_swa_dice"] == 0.92
        backup = paths.backup_run_dir(item)
        assert (backup / "best_model.pth").read_bytes() == b"weights"
        assert json.loads((backup / QUEUE_RECORD_FILE).read_text())["run_name"] == item.run_name
    else:
        assert state.status == FAILED and "exit code 3" in state.note
        assert not paths.backup_run_dir(item).exists()
