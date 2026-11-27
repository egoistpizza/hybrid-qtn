import pytest
import torch
from torch.optim.swa_utils import AveragedModel

from tests.synthetic import small_model
from utils.checkpoints import (
    BEST_LIVE_CHECKPOINT,
    SWA_CHECKPOINT,
    clean_state_dict,
    evaluation_checkpoint,
    load_weights,
    save_weights,
)

CPU = torch.device("cpu")


def test_wrapper_prefixes_and_the_swa_counter_are_removed():
    cleaned = clean_state_dict({"n_averaged": torch.tensor(3), "module._orig_mod.head.weight": torch.tensor([1.0])})
    assert list(cleaned) == ["head.weight"]


@pytest.mark.parametrize("wrap", [torch.compile, AveragedModel], ids=["compiled", "averaged"])
def test_wrapped_models_save_plain_keys(tmp_path, wrap):
    model = small_model()
    path = save_weights(wrap(model), tmp_path / "weights.pth")
    assert list(torch.load(path, weights_only=True)) == list(model.state_dict())


def test_weights_load_into_eager_and_compiled_models(tmp_path):
    source = small_model()
    path = save_weights(torch.compile(source), tmp_path / "weights.pth")
    eager, compiled = small_model(), torch.compile(small_model())
    load_weights(eager, path, CPU)
    load_weights(compiled, path, CPU)
    for loaded in (eager.state_dict(), compiled._orig_mod.state_dict()):
        assert all(torch.equal(loaded[key], value) for key, value in source.state_dict().items())


def test_legacy_prefixed_checkpoints_still_load(tmp_path):
    source = small_model()
    path = tmp_path / "legacy.pth"
    torch.save({"n_averaged": torch.tensor(2), **{f"module._orig_mod.{key}": value for key, value in source.state_dict().items()}}, path)
    target = small_model()
    load_weights(target, path, CPU)
    assert torch.equal(target[0].weight, source[0].weight)


def test_missing_checkpoint_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Checkpoint required but not found"):
        load_weights(small_model(), tmp_path / "missing.pth", CPU)


def test_evaluation_prefers_the_swa_checkpoint(tmp_path):
    assert evaluation_checkpoint(tmp_path) == tmp_path / BEST_LIVE_CHECKPOINT
    (tmp_path / SWA_CHECKPOINT).touch()
    assert evaluation_checkpoint(tmp_path) == tmp_path / SWA_CHECKPOINT
