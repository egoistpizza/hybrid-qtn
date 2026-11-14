from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler
from torch.utils.data import DataLoader, Dataset

from dataset import DatasetSplits, Denormalization, EvaluationSet
from tests.synthetic import MaskFileSamples, small_model
from train import SegmentationTrainer, TrainingConfig, compile_with_warmup, parse_arguments
from utils.loss import FocalTverskyLoss


class SquareDataset(Dataset):
    def __init__(self, size: int) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict:
        mask = torch.zeros(1, 16, 16)
        mask[:, 4:12, 4:12] = 1.0
        return {"image": mask.repeat(3, 1, 1) + 0.1 * torch.randn(3, 16, 16), "mask": mask}


class RecordingLogger:
    def __init__(self) -> None:
        self.logged: list[dict] = []
        self.summary: dict[str, float] = {}

    def start(self, name: str, config: dict, resume_after_epoch=None) -> None:
        self.config = dict(config)
        self.resume_after_epoch = resume_after_epoch

    def log(self, metrics: dict) -> None:
        self.logged.append(dict(metrics))

    def summarize(self, metrics: dict) -> None:
        self.summary.update(metrics)

    def finish(self, outcome="completed") -> None:
        self.outcome = outcome

    def keys(self) -> set[str]:
        return {key for entry in self.logged for key in entry}


def training_config(checkpoint_dir: Path) -> TrainingConfig:
    return TrainingConfig(
        model_type="vanilla", dataset="synthetic", loss="focal_tversky", seed=0, epochs=4, batch_size=2,
        bond_dim=4, fn_weight=0.3, fp_weight=0.7, focal_gamma=4.0 / 3.0, image_size=16,
        accumulation_steps=1, swa_start_pct=0.5, checkpoint_dir=str(checkpoint_dir),
    )


def evaluation_set(dataset: Dataset, foreground_maps: bool = False) -> EvaluationSet:
    return EvaluationSet(dataset, foreground_maps, Denormalization(offset=(0.0,) * 3, scale=(1.0,) * 3))


def build_trainer(checkpoint_dir: Path, test_sets: dict[str, EvaluationSet], run_logger: RecordingLogger) -> SegmentationTrainer:
    torch.manual_seed(0)
    model = small_model()
    config = training_config(checkpoint_dir)
    optimizer = optim.AdamW(model.parameters(), lr=1e-2)
    splits = DatasetSplits(
        train=SquareDataset(6), val=evaluation_set(SquareDataset(3)), test_sets=test_sets, image_size=16
    )
    return SegmentationTrainer(
        model=model,
        splits=splits,
        loader_factory=lambda dataset, shuffle: DataLoader(dataset, batch_size=2, shuffle=shuffle),
        criterion=FocalTverskyLoss(),
        optimizer=optimizer,
        scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs),
        device=torch.device("cpu"),
        config=config,
        run_logger=run_logger,
    )


@pytest.fixture
def recorder():
    return RecordingLogger()


@pytest.fixture
def trainer_and_recorder(recorder, tmp_path):
    return build_trainer(tmp_path, {"test": evaluation_set(SquareDataset(3))}, recorder), recorder


def test_every_epoch_validates_the_live_model(trainer_and_recorder, monkeypatch):
    trainer, _ = trainer_and_recorder
    evaluated_models: list[tuple[str, nn.Module]] = []
    original_evaluate = trainer.evaluate

    def spy(model, loader, description, compute_hd95, foreground_maps=False):
        evaluated_models.append((description, model))
        return original_evaluate(model, loader, description, compute_hd95, foreground_maps)

    monkeypatch.setattr(trainer, "evaluate", spy)
    trainer.fit(run_name="synthetic", total_params=0)

    per_epoch = [model for description, model in evaluated_models if "[Val]" in description and description.startswith("Epoch")]
    assert len(per_epoch) == 4
    assert all(model is trainer.model for model in per_epoch)
    assert any(model is trainer.swa_model for description, model in evaluated_models if description.startswith("SWA"))


def test_swa_averages_only_the_final_epochs(trainer_and_recorder):
    trainer, recorder = trainer_and_recorder
    trainer.fit(run_name="synthetic", total_params=0)
    assert trainer.swa_model.n_averaged.item() == 3
    assert [entry["is_swa_active"] for entry in recorder.logged if "is_swa_active" in entry] == [0.0, 1.0, 1.0, 1.0]


@pytest.mark.parametrize(("epochs", "anneal_epochs"), [(100, 10), (28, 3), (4, 1)])
def test_swa_anneal_scales_with_the_number_of_epochs(tmp_path, epochs, anneal_epochs):
    config = replace(training_config(tmp_path), epochs=epochs)
    assert config.swa_anneal_epochs == anneal_epochs


def test_swa_scheduler_uses_the_scaled_anneal(trainer_and_recorder):
    trainer, _ = trainer_and_recorder
    assert trainer.swa_scheduler.anneal_epochs == trainer.config.swa_anneal_epochs


def test_fit_logs_peak_vram_and_test_only_paper_metrics(trainer_and_recorder):
    trainer, recorder = trainer_and_recorder
    trainer.fit(run_name="synthetic", total_params=0)
    keys = recorder.keys()
    assert {"Peak_VRAM_MB", "val_dice", "swa_val_dice", "swa_val_hd95"} <= keys
    assert {"test_best_live_dice", "test_best_live_hd95", "test_swa_dice", "test_swa_hd95"} <= keys
    assert set(recorder.summary) == {key for key in keys if key.startswith("test_")}
    assert (Path(trainer.config.checkpoint_dir) / "best_model.pth").exists()
    assert (Path(trainer.config.checkpoint_dir) / "best_swa_model.pth").exists()


def test_named_test_sets_log_qualified_metrics(recorder, tmp_path):
    test_sets = {name: evaluation_set(SquareDataset(3)) for name in ("cod10k", "nc4k")}
    build_trainer(tmp_path, test_sets, recorder).fit(run_name="synthetic", total_params=0)
    keys = recorder.keys()
    for model_prefix in ("test_best_live", "test_swa"):
        for test_set in ("cod10k", "nc4k"):
            assert {f"{model_prefix}_{test_set}_dice", f"{model_prefix}_{test_set}_hd95", f"{model_prefix}_{test_set}_loss"} <= keys
    assert "test_swa_dice" not in keys


def test_foreground_map_metrics_are_reported_at_ground_truth_resolution(recorder, tmp_path):
    chameleon = evaluation_set(MaskFileSamples(tmp_path), foreground_maps=True)
    build_trainer(tmp_path, {"chameleon": chameleon}, recorder).fit(run_name="synthetic", total_params=0)
    expected = {f"test_swa_chameleon_{name}" for name in ("s_measure", "e_measure", "wf_measure", "map_mae")}
    assert expected <= set(recorder.summary)
    assert all(0.0 <= recorder.summary[key] <= 1.0 for key in expected)


def test_foreground_map_metrics_are_off_by_default(trainer_and_recorder):
    trainer, recorder = trainer_and_recorder
    trainer.fit(run_name="synthetic", total_params=0)
    assert not any(key.endswith("s_measure") for key in recorder.keys())


def load_training_state(checkpoint_dir: Path) -> dict:
    return torch.load(checkpoint_dir / "last.pt", weights_only=False)


def test_every_epoch_ends_with_a_complete_training_state(trainer_and_recorder):
    trainer, _ = trainer_and_recorder
    trainer.fit(run_name="synthetic", total_params=0)
    checkpoint_dir = Path(trainer.config.checkpoint_dir)
    state = load_training_state(checkpoint_dir)

    assert state["epoch"] == 4
    assert state["best_val_dice"] == trainer.best_val_dice
    assert state["swa_n_averaged"].item() == 3
    assert state["scheduler"]["last_epoch"] == trainer.scheduler.last_epoch
    assert {"model", "optimizer", "scaler", "swa_model", "swa_scheduler", "train_loader_generator", "rng"} <= set(state)
    assert not (checkpoint_dir / "last.pt.tmp").exists()


def test_an_interrupted_validation_leaves_the_previous_epoch_state(trainer_and_recorder, monkeypatch):
    trainer, _ = trainer_and_recorder
    original_evaluate = trainer.evaluate
    validations = 0

    def interrupt_second_validation(model, loader, description, compute_hd95, foreground_maps=False):
        nonlocal validations
        validations += 1
        if validations == 2:
            raise KeyboardInterrupt
        return original_evaluate(model, loader, description, compute_hd95, foreground_maps)

    monkeypatch.setattr(trainer, "evaluate", interrupt_second_validation)
    trainer.fit(run_name="synthetic", total_params=0)

    assert load_training_state(Path(trainer.config.checkpoint_dir))["epoch"] == 1


def interrupt_validation(trainer: SegmentationTrainer, monkeypatch, at: int) -> None:
    """Raise ``KeyboardInterrupt`` during the ``at``-th call of ``trainer.evaluate``."""
    original_evaluate = trainer.evaluate
    calls = 0

    def interrupting(model, loader, description, compute_hd95, foreground_maps=False):
        nonlocal calls
        calls += 1
        if calls == at:
            raise KeyboardInterrupt
        return original_evaluate(model, loader, description, compute_hd95, foreground_maps)

    monkeypatch.setattr(trainer, "evaluate", interrupting)


@pytest.mark.parametrize(
    "swa_start_pct, finished_epochs",
    [(0.75, 1), (0.5, 2)],
    ids=["resume-before-swa", "resume-during-swa"],
)
def test_a_resumed_run_matches_an_uninterrupted_one(tmp_path, monkeypatch, swa_start_pct, finished_epochs):
    def trainer_in(directory: Path, run_logger: RecordingLogger) -> SegmentationTrainer:
        trainer = build_trainer(directory, {"test": evaluation_set(SquareDataset(3))}, run_logger)
        trainer.config.swa_start_pct = swa_start_pct
        return trainer

    uninterrupted_log = RecordingLogger()
    uninterrupted = trainer_in(tmp_path / "uninterrupted", uninterrupted_log)
    uninterrupted.fit(run_name="synthetic", total_params=0)

    resumed_dir = tmp_path / "resumed"
    first_part = trainer_in(resumed_dir, RecordingLogger())
    interrupt_validation(first_part, monkeypatch, at=finished_epochs + 1)
    first_part.fit(run_name="synthetic", total_params=0)
    state = load_training_state(resumed_dir)
    assert state["epoch"] == finished_epochs

    resumed_log = RecordingLogger()
    resumed = trainer_in(resumed_dir, resumed_log)
    resumed.fit(run_name="synthetic", total_params=0, resume_state=state)

    for name, expected in uninterrupted.model.state_dict().items():
        assert torch.equal(resumed.model.state_dict()[name], expected), name
    for name, expected in uninterrupted.swa_model.state_dict().items():
        assert torch.equal(resumed.swa_model.state_dict()[name], expected), name
    assert resumed.best_val_dice == uninterrupted.best_val_dice
    assert resumed_log.summary == uninterrupted_log.summary
    epochs_after_resume = [entry for entry in uninterrupted_log.logged if entry.get("epoch", 0) > finished_epochs]
    assert [entry for entry in resumed_log.logged if "epoch" in entry] == epochs_after_resume


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_the_time_budget_stops_before_an_epoch_that_would_not_finish(trainer_and_recorder, monkeypatch):
    trainer, recorder = trainer_and_recorder
    clock = FakeClock()
    trainer.clock = clock
    trainer.config.max_hours = 2.5
    run_epoch = trainer._run_epoch

    def one_hour_epoch(epoch: int) -> None:
        run_epoch(epoch)
        clock.now += 3600

    monkeypatch.setattr(trainer, "_run_epoch", one_hour_epoch)
    trainer.fit(run_name="synthetic", total_params=0)

    checkpoint_dir = Path(trainer.config.checkpoint_dir)
    assert load_training_state(checkpoint_dir)["epoch"] == 2
    assert not any(key.startswith("test_") for key in recorder.keys())
    assert not (checkpoint_dir / "best_swa_model.pth").exists()

    next_session = RecordingLogger()
    resumed = build_trainer(checkpoint_dir, {"test": evaluation_set(SquareDataset(3))}, next_session)
    resumed.fit(run_name="synthetic", total_params=0, resume_state=load_training_state(checkpoint_dir))
    assert [entry["epoch"] for entry in next_session.logged if "epoch" in entry] == [3, 4]
    assert "test_swa_dice" in next_session.summary


def test_resuming_with_a_changed_epoch_count_is_refused_before_anything_starts(tmp_path, monkeypatch):
    first_part = build_trainer(tmp_path, {"test": evaluation_set(SquareDataset(3))}, RecordingLogger())
    interrupt_validation(first_part, monkeypatch, at=3)
    first_part.fit(run_name="synthetic", total_params=0)

    run_logger = RecordingLogger()
    resumed = build_trainer(tmp_path, {"test": evaluation_set(SquareDataset(3))}, run_logger)
    resumed.config.epochs = 8
    resumed.config.max_hours = 5.0  # a session setting; may change between sessions
    with pytest.raises(ValueError, match="epochs: saved 4, now 8"):
        resumed.fit(run_name="synthetic", total_params=0, resume_state=load_training_state(tmp_path))
    assert not hasattr(run_logger, "config")


def test_a_fresh_start_removes_the_state_of_an_earlier_attempt(tmp_path, monkeypatch):
    earlier = build_trainer(tmp_path, {"test": evaluation_set(SquareDataset(3))}, RecordingLogger())
    interrupt_validation(earlier, monkeypatch, at=3)
    earlier.fit(run_name="synthetic", total_params=0)
    assert (tmp_path / "last.pt").exists()

    fresh = build_trainer(tmp_path, {"test": evaluation_set(SquareDataset(3))}, RecordingLogger())
    interrupt_validation(fresh, monkeypatch, at=1)
    fresh.fit(run_name="synthetic", total_params=0)

    assert not (tmp_path / "last.pt").exists()


def test_resuming_rewrites_the_best_checkpoint_of_the_saved_state(tmp_path, monkeypatch):
    first_part = build_trainer(tmp_path, {"test": evaluation_set(SquareDataset(3))}, RecordingLogger())
    interrupt_validation(first_part, monkeypatch, at=3)
    first_part.fit(run_name="synthetic", total_params=0)
    state = load_training_state(tmp_path)
    # An epoch that was interrupted after writing a new best checkpoint but before saving its state.
    torch.save({name: torch.zeros_like(tensor) for name, tensor in state["best_weights"].items()}, tmp_path / "best_model.pth")

    resumed = build_trainer(tmp_path, {"test": evaluation_set(SquareDataset(3))}, RecordingLogger())
    resumed.restore(state)

    rewritten = torch.load(tmp_path / "best_model.pth", weights_only=True)
    assert all(torch.equal(rewritten[name], tensor) for name, tensor in state["best_weights"].items())


def test_a_state_saved_without_amp_resumes_with_amp(tmp_path, monkeypatch):
    first_part = build_trainer(tmp_path, {"test": evaluation_set(SquareDataset(3))}, RecordingLogger())
    interrupt_validation(first_part, monkeypatch, at=3)
    first_part.fit(run_name="synthetic", total_params=0)
    state = load_training_state(tmp_path)
    assert state["scaler"] == {}

    resumed = build_trainer(tmp_path, {"test": evaluation_set(SquareDataset(3))}, RecordingLogger())
    resumed.scaler = GradScaler("cpu", enabled=True)  # as on a CUDA machine, where AMP is on
    assert resumed.restore(state) == 2


def test_fit_reports_how_the_run_ended(trainer_and_recorder, monkeypatch, tmp_path):
    trainer, recorder = trainer_and_recorder
    assert trainer.fit(run_name="synthetic", total_params=0) == "completed"
    assert recorder.outcome == "completed"

    stopped_log = RecordingLogger()
    stopped = build_trainer(tmp_path / "stopped", {"test": evaluation_set(SquareDataset(3))}, stopped_log)
    clock = FakeClock()
    stopped.clock = clock
    stopped.config.max_hours = 0.5
    run_epoch = stopped._run_epoch

    def one_hour_epoch(epoch: int) -> None:
        run_epoch(epoch)
        clock.now += 3600

    monkeypatch.setattr(stopped, "_run_epoch", one_hour_epoch)
    assert stopped.fit(run_name="synthetic", total_params=0) == "out_of_time"
    assert stopped_log.outcome == "out_of_time"
    assert "max_hours" not in stopped_log.config


def test_failed_compile_warmup_leaves_no_gradients(monkeypatch, tmp_path):
    model = small_model()

    def failing_compile(module: nn.Module):
        def run(inputs: torch.Tensor) -> torch.Tensor:
            module(inputs).sum().backward()
            raise RuntimeError("backend unavailable")

        return run

    monkeypatch.setattr(torch, "compile", failing_compile)
    assert compile_with_warmup(model, training_config(tmp_path), torch.device("cpu")) is model
    assert all(parameter.grad is None for parameter in model.parameters())


def test_fold_option_defaults_to_the_holdout_split(monkeypatch):
    monkeypatch.setattr("sys.argv", ["train.py", "--dataset", "cvc_clinicdb"])
    assert parse_arguments().fold is None
    monkeypatch.setattr("sys.argv", ["train.py", "--dataset", "cvc_clinicdb", "--fold", "3"])
    assert parse_arguments().fold == 3
