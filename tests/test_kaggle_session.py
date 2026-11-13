import csv

import pytest

from scripts import kaggle_session
from scripts.kaggle_session import (
    UsageMonitor,
    link_data_directory,
    parse_arguments,
    previous_sessions,
    queue_command,
    restore_previous_session,
)


def make_session(root, files):
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return root


def test_the_previous_output_is_found_among_the_inputs(tmp_path):
    session = make_session(tmp_path / "input" / "hqtn-duts" / "hybrid-qtn", {"checkpoints/run/last.pt": "state"})
    make_session(tmp_path / "input" / "some-dataset" / "hybrid-qtn", {"README.md": "a copy without training state"})
    assert previous_sessions(tmp_path / "input", "hybrid-qtn") == [session]
    assert previous_sessions(tmp_path / "missing", "hybrid-qtn") == []


def test_restoring_copies_the_training_state_into_the_repository(tmp_path):
    session = make_session(
        tmp_path / "previous",
        {
            "checkpoints/run/last.pt": "epoch 40",
            "backup/paper_v1/checkpoints/done_run/summary.json": "{}",
            "outputs/queue/paper_v1/status.txt": "paused",
            "train.py": "code is not copied",
        },
    )
    repo = make_session(tmp_path / "repo", {"train.py": "current code"})

    assert restore_previous_session(repo, [session]) == session

    assert (repo / "checkpoints/run/last.pt").read_text() == "epoch 40"
    assert (repo / "backup/paper_v1/checkpoints/done_run/summary.json").exists()
    assert (repo / "outputs/queue/paper_v1/status.txt").read_text() == "paused"
    assert (repo / "train.py").read_text() == "current code"


def test_restoring_refuses_several_previous_sessions(tmp_path):
    sessions = [make_session(tmp_path / name, {"checkpoints/x": ""}) for name in ("v1", "v2")]
    with pytest.raises(RuntimeError, match="Attach only the latest version"):
        restore_previous_session(tmp_path / "repo", sessions)


def test_data_goes_to_scratch_but_a_local_data_directory_is_kept(tmp_path):
    repo, scratch = tmp_path / "repo", tmp_path / "scratch"
    repo.mkdir()
    link_data_directory(repo, scratch)
    assert (repo / "data").resolve() == scratch.resolve()
    link_data_directory(repo, scratch)  # idempotent across sessions

    local = tmp_path / "local"
    (local / "data" / "duts").mkdir(parents=True)
    link_data_directory(local, scratch)
    assert not (local / "data").is_symlink()


def test_the_queue_gets_the_time_left_and_the_extra_options(monkeypatch):
    arguments, extra = parse_arguments(["--dataset", "duts", "--tag", "paper_v1", "--tiers", "1", "--seeds", "42"])
    command = queue_command(arguments, extra, hours_left=10.25, wandb_logged_in=False)

    assert command[1:3] == ["-m", "scripts.run_queue"]
    options = command[3:]
    assert options[options.index("--dataset") + 1] == "duts"
    assert options[options.index("--max_hours") + 1] == "10.250"
    assert options[options.index("--backup_root") + 1].endswith("/backup")
    assert ["--tiers", "1", "--seeds", "42"] == options[options.index("--tiers"):options.index("--seeds") + 2]
    assert "--skip_wandb_check" in options
    assert "--skip_wandb_check" not in queue_command(arguments, extra, 10.25, wandb_logged_in=True)


def test_the_queue_looks_up_finished_runs_in_the_team_entity():
    arguments, extra = parse_arguments(["--dataset", "duts", "--tag", "paper_v1"])
    command = queue_command(arguments, extra, 10.0, wandb_logged_in=True, wandb_entity="team")
    assert command[command.index("--wandb_entity") + 1] == "team"
    assert "--wandb_entity" not in queue_command(arguments, extra, 10.0, wandb_logged_in=True)


def test_an_unknown_dataset_stops_the_session_before_any_set_up(monkeypatch):
    monkeypatch.setattr(kaggle_session, "environment_report", lambda: pytest.fail("set-up started"))
    with pytest.raises(SystemExit, match="No downloader for 'dutz'"):
        kaggle_session.main(["--dataset", "dutz", "--tag", "t"])


def test_usage_samples_are_appended_per_gpu(tmp_path):
    path = tmp_path / "usage.csv"
    with UsageMonitor(path, interval=3600, query=lambda: [["0", "97", "10240"], ["1", "0", "3"]]) as monitor:
        monitor.sample()
    rows = list(csv.DictReader(path.open()))
    assert [(row["gpu"], row["utilization_pct"]) for row in rows] == [("0", "97"), ("1", "0")]
    assert all(row["cpu_load_1min"] and row["cpus"] for row in rows)
