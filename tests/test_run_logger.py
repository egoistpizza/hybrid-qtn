import json

import torch

import train
from tests.test_train import RecordingLogger, build_trainer, evaluation_set, SquareDataset
from utils import run_logger as run_logger_module
from utils.run_logger import (
    RUN_CONFIG_FILE,
    RUN_HISTORY_FILE,
    RUN_SUMMARY_FILE,
    WANDB_RUN_ID_FILE,
    CombinedLogger,
    JsonFileLogger,
    WandbLogger,
)


def history(directory):
    return [json.loads(line) for line in (directory / RUN_HISTORY_FILE).read_text().splitlines()]


def test_file_logger_writes_config_history_and_summary(tmp_path):
    run_logger = JsonFileLogger(tmp_path / "run")
    run_logger.start("name", {"seed": 42})
    run_logger.log({"epoch": 1, "val_dice": 0.5})
    run_logger.summarize({"test_swa_dice": 0.9})
    run_logger.finish()

    assert json.loads((tmp_path / "run" / RUN_CONFIG_FILE).read_text())["config"] == {"seed": 42}
    assert history(tmp_path / "run") == [{"epoch": 1, "val_dice": 0.5}]
    summary = json.loads((tmp_path / "run" / RUN_SUMMARY_FILE).read_text())
    assert summary["test_swa_dice"] == 0.9 and "finished_at" in summary


def test_restarting_replaces_the_previous_attempt(tmp_path):
    first = JsonFileLogger(tmp_path)
    first.start("name", {})
    first.log({"epoch": 1})
    first.summarize({"test_swa_dice": 0.9})

    JsonFileLogger(tmp_path).start("name", {})
    assert history(tmp_path) == []
    assert not (tmp_path / RUN_SUMMARY_FILE).exists()


def test_resuming_keeps_the_history_up_to_the_resumed_epoch(tmp_path):
    first = JsonFileLogger(tmp_path)
    first.start("name", {"seed": 42})
    for epoch in (1, 2, 3):
        first.log({"epoch": epoch})
    first.log({"swa_val_dice": 0.8})

    JsonFileLogger(tmp_path).start("name", {"seed": 42}, resume_after_epoch=2)

    assert history(tmp_path) == [{"epoch": 1}, {"epoch": 2}]
    run_config = json.loads((tmp_path / RUN_CONFIG_FILE).read_text())
    assert "started_at" in run_config and len(run_config["resumed_at"]) == 1


def test_rewinding_skips_a_line_cut_off_by_a_hard_kill(tmp_path):
    first = JsonFileLogger(tmp_path)
    first.start("name", {})
    for epoch in (1, 2):
        first.log({"epoch": epoch})
    with (tmp_path / RUN_HISTORY_FILE).open("a") as history_file:
        history_file.write('{"epoch": 3, "train_lo')

    JsonFileLogger(tmp_path).start("name", {}, resume_after_epoch=2)

    assert history(tmp_path) == [{"epoch": 1}, {"epoch": 2}]


def test_only_a_completed_run_leaves_a_summary(tmp_path):
    run_logger = JsonFileLogger(tmp_path)
    run_logger.start("name", {})
    run_logger.summarize({"test_best_live_dice": 0.9})  # stopped while reporting test metrics
    run_logger.finish("interrupted")
    assert not (tmp_path / RUN_SUMMARY_FILE).exists()

    run_logger.start("name", {})
    run_logger.summarize({"test_best_live_dice": 0.9})
    run_logger.finish("completed")
    assert "finished_at" in json.loads((tmp_path / RUN_SUMMARY_FILE).read_text())


def test_an_epoch_logged_but_not_saved_appears_once_after_resuming(tmp_path, monkeypatch):
    def train_with_files(run_dir):
        trainer = build_trainer(run_dir, {"test": evaluation_set(SquareDataset(3))}, RecordingLogger())
        trainer.run_logger = JsonFileLogger(run_dir)
        return trainer

    uninterrupted_dir = tmp_path / "uninterrupted"
    train_with_files(uninterrupted_dir).fit(run_name="synthetic", total_params=0)

    resumed_dir = tmp_path / "resumed"
    save_training_state = train.save_training_state
    saves = 0

    def interrupt_second_save(state, path):
        nonlocal saves
        saves += 1
        if saves == 2:
            raise KeyboardInterrupt
        return save_training_state(state, path)

    monkeypatch.setattr(train, "save_training_state", interrupt_second_save)
    train_with_files(resumed_dir).fit(run_name="synthetic", total_params=0)
    assert [entry["epoch"] for entry in history(resumed_dir)] == [1, 2]

    monkeypatch.setattr(train, "save_training_state", save_training_state)
    state = torch.load(resumed_dir / "last.pt", weights_only=False)
    assert state["epoch"] == 1
    train_with_files(resumed_dir).fit(run_name="synthetic", total_params=0, resume_state=state)

    assert history(resumed_dir) == history(uninterrupted_dir)


class FakeWandbInit:
    """Records ``wandb.init`` calls and hands out run ids ``run-1``, ``run-2``, ..."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, **arguments):
        self.calls.append(arguments)
        return type("Run", (), {"id": arguments.get("id") or f"run-{len(self.calls)}"})()


def test_a_resumed_training_continues_the_same_wandb_run(tmp_path, monkeypatch):
    fake_init = FakeWandbInit()
    monkeypatch.setattr(run_logger_module.wandb, "init", fake_init)

    WandbLogger(tmp_path).start("name", {})
    assert (tmp_path / WANDB_RUN_ID_FILE).read_text() == "run-1"

    WandbLogger(tmp_path).start("name", {}, resume_after_epoch=40)
    assert fake_init.calls[-1]["id"] == "run-1"
    assert fake_init.calls[-1]["resume"] == "must"

    WandbLogger(tmp_path).start("name", {})
    assert "id" not in fake_init.calls[-1]
    assert (tmp_path / WANDB_RUN_ID_FILE).read_text() == "run-3"


def test_resuming_without_a_saved_wandb_id_opens_a_new_run(tmp_path, monkeypatch):
    fake_init = FakeWandbInit()
    monkeypatch.setattr(run_logger_module.wandb, "init", fake_init)

    WandbLogger(tmp_path).start("name", {}, resume_after_epoch=40)

    assert "resume" not in fake_init.calls[-1]
    assert (tmp_path / WANDB_RUN_ID_FILE).read_text() == "run-1"


def test_training_mirrors_metrics_into_the_run_directory(tmp_path):
    recorder = RecordingLogger()
    run_dir = tmp_path / "run"
    trainer = build_trainer(tmp_path, {"test": evaluation_set(SquareDataset(3))}, recorder)
    trainer.run_logger = CombinedLogger(recorder, JsonFileLogger(run_dir))
    trainer.fit(run_name="synthetic", total_params=123)

    summary = json.loads((run_dir / RUN_SUMMARY_FILE).read_text())
    assert summary["test_best_live_dice"] == recorder.summary["test_best_live_dice"]
    assert [entry["epoch"] for entry in history(run_dir) if "epoch" in entry] == [1, 2, 3, 4]
    assert json.loads((run_dir / RUN_CONFIG_FILE).read_text())["config"]["total_params"] == 123
