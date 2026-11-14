import json
import textwrap
from collections import Counter

import pytest

import scripts.run_queue as run_queue
import train
from scripts.run_queue import (
    DONE,
    DONE_METRIC,
    DONE_NOT_BACKED_UP,
    FAILED,
    PAUSED,
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
def test_train_arguments_select_the_queued_run(item, monkeypatch):
    monkeypatch.setattr("sys.argv", ["train.py", *item.train_arguments(epochs=100, max_hours=2.5)])
    arguments = train.parse_arguments()
    assert RunSpec.from_arguments(arguments) == item.spec
    assert arguments.epochs == 100
    assert arguments.resume and arguments.max_hours == 2.5


def test_a_queue_on_another_dataset_trains_that_dataset(monkeypatch):
    items = build_queue("paper_v1", seeds=[42], tiers=[1], dataset="duts")
    assert [item.run_name for item in items] == [
        "duts__vanilla_unet__focal_tversky__seed42__paper_v1",
        "duts__hybrid_unet_b32_rel_nope__focal_tversky__seed42__paper_v1",
    ]
    monkeypatch.setattr("sys.argv", ["train.py", *items[1].train_arguments(epochs=100)])
    assert RunSpec.from_arguments(train.parse_arguments()) == items[1].spec
    assert run_queue.rerun_item(items[1]).dataset == "duts"


def test_an_unknown_dataset_is_rejected_before_anything_starts():
    with pytest.raises(SystemExit, match="Unknown dataset 'dutz'"):
        run_queue.main(["--tag", "t", "--dataset", "dutz", "--dry_run", "--skip_wandb_check"])


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
    run_dir = Path(os.environ["FAKE_RUN_DIR"])
    if "--resume" in sys.argv and (run_dir / "last.pt").exists():
        print("Resuming after epoch 1", flush=True)
    print("Epoch 1/1 [Train]", flush=True)
    print("GPU", os.environ.get("CUDA_VISIBLE_DEVICES"), flush=True)
    if os.environ["FAKE_OUTCOME"] == "fail":
        sys.exit(3)
    run_dir.mkdir(parents=True, exist_ok=True)
    if os.environ["FAKE_OUTCOME"] == "pause":
        (run_dir / "last.pt").write_bytes(b"state")
        sys.exit(75)
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


def test_a_paused_run_keeps_its_state_and_resumes_on_the_next_launch(repo, tmp_path, monkeypatch):
    (repo / "train.py").write_text(FAKE_TRAIN)
    paths = QueuePaths("t", tmp_path / "backup")
    paths.logs_dir.mkdir(parents=True)
    item = build_queue("t", seeds=[42], tiers=[1])[0]
    monkeypatch.setenv("FAKE_RUN_DIR", str(paths.local_run_dir(item)))

    monkeypatch.setenv("FAKE_OUTCOME", "pause")
    paused = RunState(item)
    assert run_queue.run_one(paused, Launch("t", 0.0, None, 1, "main @ abc"), [paused], paths)
    assert paused.status == PAUSED
    assert (paths.local_run_dir(item) / "last.pt").exists()

    monkeypatch.setenv("FAKE_OUTCOME", "success")
    resumed = RunState(item)
    assert run_queue.run_one(resumed, Launch("t", 0.0, None, 1, "main @ abc"), [resumed], paths)
    assert resumed.status == DONE
    assert "Resuming after epoch 1" in paths.log_file(item).read_text()


def test_the_queue_stops_after_a_paused_run(tmp_path, monkeypatch):
    paths = QueuePaths("t", None)
    states = [RunState(item) for item in build_queue("t", seeds=[42], tiers=[1])]
    started = []

    def pause(state, launch, all_states, queue_paths):
        started.append(state.item.run_name)
        state.status = PAUSED
        return True

    monkeypatch.setattr(run_queue, "run_one", pause)
    monkeypatch.setattr(run_queue, "write_text_atomically", lambda path, text: None)
    monkeypatch.setattr(run_queue, "back_up_queue_files", lambda *arguments: None)
    monkeypatch.setattr(run_queue, "render_status", lambda *arguments, **keywords: "")
    launch = Launch("t", run_queue.time.time(), 10.0, 1, "main @ abc")
    monkeypatch.setattr(run_queue, "REPO_ROOT", tmp_path)
    paths.status_file.parent.mkdir(parents=True, exist_ok=True)

    run_queue.run_queue(launch, states, paths)

    assert started == [states[0].item.run_name]
    assert launch.state.startswith("paused")


def test_lanes_split_the_queue_without_overlap(repo, tmp_path):
    states = resolve_queue(build_queue("t", tiers=[1]), QueuePaths("t", None), {})
    lanes = [run_queue.lane_states(states, lane, 2) for lane in range(2)]

    names = [[state.item.run_name for state in lane] for lane in lanes]
    assert sorted(names[0] + names[1]) == sorted(state.item.run_name for state in states)
    assert not set(names[0]) & set(names[1])
    # The same lane gets the same runs in every session, so it resumes the runs it paused.
    again = resolve_queue(build_queue("t", tiers=[1]), QueuePaths("t", None), {})
    assert [state.item.run_name for state in run_queue.lane_states(again, 1, 2)] == names[1]
    assert run_queue.lane_states(states, 0, 1) == states


def test_each_lane_has_its_own_status_file_but_shares_logs_and_stop(repo):
    single, first, second = QueuePaths("t", None), QueuePaths("t", None, 0, 2), QueuePaths("t", None, 1, 2)
    assert single.status_file.name == "status.txt"
    assert len({single.status_file, first.status_file, second.status_file}) == 3
    assert first.lane_lock_file != second.lane_lock_file
    assert first.logs_dir == second.logs_dir and first.stop_file == second.stop_file


def test_lanes_run_together_but_never_beside_a_single_lane_queue(repo):
    first, second, single = QueuePaths("t", None, 0, 2), QueuePaths("t", None, 1, 2), QueuePaths("t", None)
    first.queue_dir.mkdir(parents=True)

    held = run_queue.acquire_locks(first) + run_queue.acquire_locks(second)
    with pytest.raises(SystemExit, match="lane 0"):
        run_queue.acquire_locks(first)
    with pytest.raises(SystemExit, match="already running"):
        run_queue.acquire_locks(single)

    for lock in held:
        lock.close()
    held = run_queue.acquire_locks(single)
    with pytest.raises(SystemExit, match="already running"):
        run_queue.acquire_locks(second)


def test_a_lane_trains_on_its_own_gpu(repo, tmp_path, monkeypatch):
    (repo / "train.py").write_text(FAKE_TRAIN)
    paths = QueuePaths("t", None, 1, 2)
    paths.logs_dir.mkdir(parents=True)
    item = build_queue("t", seeds=[42], tiers=[1])[1]
    monkeypatch.setenv("FAKE_OUTCOME", "success")
    monkeypatch.setenv("FAKE_RUN_DIR", str(paths.local_run_dir(item)))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    state = RunState(item)

    assert run_queue.run_one(state, Launch("t", 0.0, None, 1, "main @ abc", gpu="1"), [state], paths)
    assert "GPU 1\n" in paths.log_file(item).read_text()


def test_the_lane_gpu_comes_from_the_visible_devices(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2, 3")
    assert run_queue.visible_gpus() == ["2", "3"]


@pytest.mark.parametrize(
    "options, message",
    [(["--lanes", "2", "--lane", "2"], "--lane must be between 0 and 1"), (["--lanes", "2", "--lane", "1"], "needs one GPU per lane; found 1")],
)
def test_lanes_that_cannot_run_stop_before_anything_starts(monkeypatch, options, message):
    monkeypatch.setattr(run_queue, "visible_gpus", lambda: ["0"])
    with pytest.raises(SystemExit, match=message):
        run_queue.main(["--tag", "t", "--skip_wandb_check", "--no_backup", *options])


def test_a_lane_dry_run_lists_only_its_runs(capsys):
    run_queue.main(["--tag", "t", "--dataset", "duts", "--tiers", "1", "--seeds", "42", "--lanes", "2", "--lane", "1", "--dry_run", "--skip_wandb_check"])
    output = capsys.readouterr().out
    assert "duts__hybrid_unet_b32_rel_nope__focal_tversky__seed42__t" in output
    assert "duts__vanilla_unet" not in output
    assert "lane 1 of 2" in output
