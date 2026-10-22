import json

from tests.test_train import RecordingLogger, build_trainer, evaluation_set, SquareDataset
from utils.run_logger import RUN_CONFIG_FILE, RUN_HISTORY_FILE, RUN_SUMMARY_FILE, CombinedLogger, JsonFileLogger


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
