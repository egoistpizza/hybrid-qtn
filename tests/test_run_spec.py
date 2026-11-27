import sys

import pytest
import torch

import evaluate
import train
from dataset import DatasetSplits, Denormalization, EvaluationSet, build_loader
from export import export_filename
from models.variant import MPSVariant
from tests.synthetic import MaskFileSamples
from utils.checkpoints import SWA_CHECKPOINT, save_weights
from utils.evaluation import evaluate_model, metric_suites
from utils.run_naming import build_run_name
from utils.run_spec import CHECKPOINT_ROOT, RunSpec

RUN_ARGUMENTS = [
    "--model", "hybrid", "--dataset", "cvc_clinicdb", "--loss", "bce_dice", "--seed", "43", "--bond_dim", "16",
    "--axial_operator", "circular", "--fold", "2", "--tag", "kaggle",
]


def parse(module_parser, arguments, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["script.py", *arguments])
    return RunSpec.from_arguments(module_parser())


@pytest.mark.parametrize("arguments", [[], RUN_ARGUMENTS])
def test_training_and_evaluation_resolve_the_same_run(arguments, monkeypatch):
    trained = parse(train.parse_arguments, arguments, monkeypatch)
    evaluated = parse(evaluate.parse_args, arguments, monkeypatch)
    assert trained == evaluated
    assert trained.checkpoint_dir == evaluated.checkpoint_dir == CHECKPOINT_ROOT / trained.run_name


def test_run_spec_names_runs_like_build_run_name(monkeypatch):
    spec = parse(train.parse_arguments, RUN_ARGUMENTS, monkeypatch)
    assert spec.run_name == build_run_name("cvc_clinicdb", "hybrid", 16, "bce_dice", 43, "kaggle", MPSVariant("circular"), 2)
    assert spec.run_name == "cvc_clinicdb__hybrid_unet_b16_circ_nope__bce_dice__seed43__fold2__kaggle"


def test_default_run_is_the_hybrid_with_the_default_variant(monkeypatch):
    assert parse(evaluate.parse_args, [], monkeypatch) == RunSpec()
    assert RunSpec().run_name == "kvasir_seg__hybrid_unet_b32_rel_nope__focal_tversky__seed42"


def test_evaluation_rejects_an_unknown_loss(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--loss", "dice"])
    with pytest.raises(SystemExit):
        evaluate.parse_args()


@pytest.mark.parametrize(
    "model_type, variant, expected",
    [
        ("vanilla", MPSVariant(), "vanilla_unet_s512.onnx"),
        ("hybrid", MPSVariant(), "hybrid_unet_b32_rel_nope_s512.onnx"),
        ("hybrid", MPSVariant("absolute", positional_encoding=True), "hybrid_unet_b32_s512.onnx"),
    ],
)
def test_export_files_are_named_after_the_architecture(model_type, variant, expected):
    assert export_filename(model_type, 32, variant, 512) == expected


def test_evaluate_run_loads_the_checkpoint_training_writes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    spec = RunSpec(model_type="vanilla", datasets=("synthetic",))
    spec.checkpoint_dir.mkdir(parents=True)
    torch.manual_seed(0)
    trained = spec.build_model(16)
    save_weights(trained, spec.checkpoint_dir / SWA_CHECKPOINT)
    identity = Denormalization(offset=(0.0,) * 3, scale=(1.0,) * 3)
    test_set = EvaluationSet(MaskFileSamples(tmp_path), False, identity)
    splits = DatasetSplits(train=[], val=test_set, test_sets={"test": test_set}, image_size=16)

    results = evaluate.evaluate_run(spec, splits, torch.device("cpu"), num_samples=5, output_dir=tmp_path / "panels")

    loader = build_loader(test_set.dataset, evaluate.EVALUATION_BATCH_SIZE, shuffle=False, seed=spec.seed)
    expected = evaluate_model(trained, loader, torch.device("cpu"), metric_suites(compute_hd95=True))
    assert results == {"test": expected}
    assert len(list((tmp_path / "panels" / "test").glob("*.png"))) == len(test_set.dataset)
