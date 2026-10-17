import random

import numpy as np
import pytest
import torch

from utils.hardware import reset_peak_vram, vram_metrics
from utils.run_naming import build_run_name
from utils.seed import seed_everything, seeded_generator


@pytest.fixture(autouse=True)
def restore_global_determinism():
    yield
    torch.use_deterministic_algorithms(False)


def _draw_everything() -> tuple[float, float, torch.Tensor]:
    return random.random(), float(np.random.rand()), torch.rand(3)


def test_seed_everything_reproduces_all_generators():
    seed_everything(123)
    first = _draw_everything()
    seed_everything(123)
    second = _draw_everything()
    assert first[:2] == second[:2]
    assert torch.equal(first[2], second[2])


def test_seed_everything_fixes_cudnn_and_leaves_deterministic_algorithms_off_by_default():
    seed_everything(0)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
    assert not torch.are_deterministic_algorithms_enabled()


def test_seed_everything_enables_deterministic_algorithms_on_request():
    seed_everything(0, deterministic_algorithms=True)
    assert torch.are_deterministic_algorithms_enabled()
    seed_everything(0)
    assert not torch.are_deterministic_algorithms_enabled()


def test_different_seeds_produce_different_streams():
    seed_everything(42)
    first = torch.rand(8)
    seed_everything(43)
    assert not torch.equal(first, torch.rand(8))


def test_seeded_generator_reproduces_shuffling():
    first = torch.randperm(50, generator=seeded_generator(7))
    assert torch.equal(first, torch.randperm(50, generator=seeded_generator(7)))


def test_vram_metrics_on_cpu_are_zero_and_complete():
    device = torch.device("cpu")
    reset_peak_vram(device)
    assert vram_metrics(device) == {"Peak_VRAM_MB": 0.0, "vram_allocated_mb": 0.0, "vram_reserved_mb": 0.0}


def test_run_name_encodes_seed_and_tag():
    assert build_run_name("kvasir_seg", "hybrid", 32, "focal_tversky", 43, "ablation") == (
        "kvasir_seg__hybrid_unet_b32__focal_tversky__seed43__ablation"
    )
    assert build_run_name("kvasir_seg", "vanilla", 32, "bce_dice", 42) == "kvasir_seg__vanilla_unet__bce_dice__seed42"
