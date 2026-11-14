"""Evaluate a trained checkpoint on the validation split or on every test set of a dataset.

Prints the per-image metrics of each set, renders qualitative panels and, with
``--save_maps``, writes probability maps at ground-truth resolution.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset

from dataset import DatasetSplits, Denormalization, EvaluationSet, build_loader, build_splits
from utils import get_device
from utils.checkpoints import evaluation_checkpoint, load_weights
from utils.evaluation import Batch, evaluate_model, metric_suites
from utils.foreground_maps import full_resolution_maps
from utils.metrics import MetricAccumulator, binarize
from utils.run_spec import CHECKPOINT_ROOT, RunSpec

TEST_SPLIT = "test"
VALIDATION_SPLIT = "val"
EVALUATION_SPLITS = (TEST_SPLIT, VALIDATION_SPLIT)
EVALUATION_BATCH_SIZE = 4
VISUALIZATION_SEED = 123


def save_panel(
    image: torch.Tensor,
    mask: torch.Tensor,
    logits: torch.Tensor,
    sample_id: str,
    path: Path,
    denormalization: Denormalization,
) -> None:
    """Save an image / ground truth / prediction panel titled with the per-image metrics.

    Args:
        image: Normalized ``(3, H, W)`` network input; ``denormalization`` restores it for display.
        mask: ``(1, H, W)`` ground truth.
        logits: ``(1, H, W)`` prediction logits.
    """
    accumulator = MetricAccumulator(compute_hd95=True)
    accumulator.update(logits[None], mask[None])
    metrics = accumulator.compute()
    figure, axes = plt.subplots(1, 3, figsize=(16, 5))
    panels = (
        (denormalization(image), None, f"Original RGB\n({sample_id})"),
        (mask.cpu().squeeze().numpy(), "gray", "Ground Truth Mask"),
        (
            binarize(logits).cpu().squeeze().numpy(),
            "gray",
            f"Prediction | Dice: {metrics['dice']:.4f} | IoU: {metrics['iou']:.4f}\n"
            f"HD95: {metrics['hd95']:.2f} | MAE: {metrics['mae']:.4f} | F2: {metrics['f2']:.4f}",
        ),
    )
    for axis, (panel, colormap, title) in zip(axes, panels):
        axis.imshow(panel, cmap=colormap)
        axis.set_title(title)
        axis.axis("off")

    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(figure)


class PanelWriter:
    """Saves a panel for every sample it sees, numbered in loader order."""

    def __init__(self, output_dir: Path, denormalization: Denormalization) -> None:
        self.output_dir = output_dir
        self.denormalization = denormalization
        self._rendered = 0

    def update(self, logits: torch.Tensor, masks: torch.Tensor, batch: Batch) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for index, sample_id in enumerate(batch["metadata"]["sample_id"]):
            self._rendered += 1
            path = self.output_dir / f"viz_{self._rendered:03d}_{sample_id}.png"
            save_panel(batch["image"][index], masks[index], logits[index], sample_id, path, self.denormalization)

    def compute(self) -> dict[str, float]:
        return {}


class ProbabilityMapWriter:
    """Saves each sample's uint8 probability map at ground-truth resolution as ``<sample_id>.png``."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir

    def update(self, logits: torch.Tensor, masks: torch.Tensor, batch: Batch) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        maps = full_resolution_maps(logits, batch["metadata"]["mask_path"])
        for (prediction, _), sample_id in zip(maps, batch["metadata"]["sample_id"], strict=True):
            Image.fromarray(prediction).save(self.output_dir / f"{sample_id}.png")

    def compute(self) -> dict[str, float]:
        return {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate segmentation models")
    RunSpec.add_arguments(parser)
    parser.add_argument("--split", type=str, default="test", choices=EVALUATION_SPLITS)
    parser.add_argument("--num_samples", type=int, default=10, choices=range(5, 11))
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--run", type=str, default=None, help="Checkpoint run name; overrides the derived name")
    parser.add_argument("--save_maps", action="store_true", help="Save probability maps at ground-truth resolution")
    return parser.parse_args()


def evaluation_sets(splits: DatasetSplits, split: str) -> dict[str, EvaluationSet]:
    """Named datasets to evaluate: every test set, or the validation split."""
    return dict(splits.test_sets) if split == TEST_SPLIT else {VALIDATION_SPLIT: splits.val}


def evaluate_run(
    spec: RunSpec,
    splits: DatasetSplits,
    device: torch.device,
    split: str = TEST_SPLIT,
    num_samples: int = 10,
    output_dir: Path | None = None,
    run_name: str | None = None,
    save_maps: bool = False,
) -> dict[str, dict[str, float]]:
    """Evaluate the checkpoint of ``spec`` (or of ``run_name``) on every set of ``split``.

    Writes qualitative panels to ``output_dir`` and, with ``save_maps``, probability maps;
    returns the metrics of each evaluated set.
    """
    run_name = run_name or spec.run_name
    model = spec.build_model(splits.image_size).to(device).to(memory_format=torch.channels_last)
    checkpoint = evaluation_checkpoint(CHECKPOINT_ROOT / run_name)
    print(f"Loading weights from: {checkpoint}")
    load_weights(model, checkpoint, device)

    output_dir = output_dir or Path("outputs") / "visualizations" / run_name
    results = {}
    for set_name, evaluation_set in evaluation_sets(splits, split).items():
        dataset = evaluation_set.dataset
        print(f"Dataset '{spec.dataset_slug}': {len(dataset)} {set_name} samples on {device}")
        loader = build_loader(dataset, EVALUATION_BATCH_SIZE, shuffle=False, seed=spec.seed)
        suites = metric_suites(compute_hd95=True, foreground_maps=evaluation_set.foreground_maps)
        if save_maps:
            suites.append(ProbabilityMapWriter(Path("outputs") / "prediction_maps" / run_name / set_name))
        results[set_name] = evaluate_model(model, loader, device, suites, description=f"Evaluating {set_name}")
        print(" | ".join(f"{set_name} {name}: {value:.4f}" for name, value in results[set_name].items()))

        # Panels use a separate batch-size-1 pass: logits near the threshold depend on the batch size.
        panel_positions = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(VISUALIZATION_SEED))
        panel_loader = DataLoader(Subset(dataset, panel_positions[:num_samples].tolist()), batch_size=1, shuffle=False)
        panel_writer = PanelWriter(output_dir / set_name, evaluation_set.denormalization)
        evaluate_model(model, panel_loader, device, [panel_writer], description="Saving visualizations")
    print(f"Visualizations saved to: {output_dir.absolute()}")
    return results


def main() -> None:
    args = parse_args()
    spec = RunSpec.from_arguments(args)
    splits = build_splits(list(spec.datasets), fold=spec.fold)
    evaluate_run(
        spec,
        splits,
        get_device(),
        split=args.split,
        num_samples=args.num_samples,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        run_name=args.run,
        save_maps=args.save_maps,
    )


if __name__ == "__main__":
    main()
