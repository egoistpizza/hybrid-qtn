import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import train
from train import SegmentationTrainer, TrainingConfig
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


class RecordingWandb:
    def __init__(self) -> None:
        self.logged: list[dict] = []
        self.run = SimpleNamespace(summary={})

    def init(self, **_: object) -> None:
        pass

    def log(self, data: dict) -> None:
        self.logged.append(dict(data))

    def finish(self) -> None:
        pass

    def keys(self) -> set[str]:
        return {key for entry in self.logged for key in entry}


@pytest.fixture
def trainer_and_recorder(monkeypatch, tmp_path):
    recorder = RecordingWandb()
    monkeypatch.setattr(train, "wandb", recorder)
    torch.manual_seed(0)
    model = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4), nn.ReLU(), nn.Conv2d(4, 1, 1))
    config = TrainingConfig(
        model_type="vanilla", dataset="synthetic", loss="focal_tversky", seed=0, epochs=4, batch_size=2,
        bond_dim=4, fn_weight=0.3, fp_weight=0.7, focal_gamma=4.0 / 3.0, image_size=16,
        accumulation_steps=1, swa_start_pct=0.5, checkpoint_dir=str(tmp_path),
    )
    optimizer = optim.AdamW(model.parameters(), lr=1e-2)
    trainer = SegmentationTrainer(
        model=model,
        train_loader=DataLoader(SquareDataset(6), batch_size=2, shuffle=True),
        val_loader=DataLoader(SquareDataset(3), batch_size=2),
        test_loader=DataLoader(SquareDataset(3), batch_size=2),
        criterion=FocalTverskyLoss(),
        optimizer=optimizer,
        scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs),
        device=torch.device("cpu"),
        config=config,
    )
    return trainer, recorder


def test_every_epoch_validates_the_live_model(trainer_and_recorder, monkeypatch):
    trainer, _ = trainer_and_recorder
    evaluated_models: list[tuple[str, nn.Module]] = []
    original_evaluate = trainer.evaluate

    def spy(model, loader, description, compute_hd95):
        evaluated_models.append((description, model))
        return original_evaluate(model, loader, description, compute_hd95)

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
    assert set(recorder.run.summary) == {key for key in keys if key.startswith("test_")}
    assert (Path(trainer.config.checkpoint_dir) / "best_model.pth").exists()
    assert (Path(trainer.config.checkpoint_dir) / "best_swa_model.pth").exists()
