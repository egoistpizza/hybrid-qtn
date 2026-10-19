from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
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

    def start(self, name: str, config: dict) -> None:
        self.config = dict(config)

    def log(self, metrics: dict) -> None:
        self.logged.append(dict(metrics))

    def summarize(self, metrics: dict) -> None:
        self.summary.update(metrics)

    def finish(self) -> None:
        pass

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
