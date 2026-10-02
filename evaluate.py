import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset import build_splits, discover_dataset_configs, parse_dataset_arg, slugify_dataset_arg
from models.factory import MODEL_TYPES, build_segmentation_model
from utils import get_device
from utils.metrics import MetricAccumulator, binarize, calculate_all_metrics
from utils.run_naming import build_run_name

IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)
IMAGE_SIZE = 512
EVALUATION_SPLITS = ("test", "val")
CHECKPOINT_PREFIXES_TO_STRIP = ("_orig_mod.", "module.")
SWA_METADATA_KEY = "n_averaged"
VISUALIZATION_SEED = 123


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGE_MEAN, device=tensor.device).view(3, 1, 1)
    std = torch.tensor(IMAGE_STD, device=tensor.device).view(3, 1, 1)
    return torch.clamp(tensor * std + mean, 0.0, 1.0).cpu().permute(1, 2, 0).numpy()


def save_visualizations(
    images: torch.Tensor,
    masks: torch.Tensor,
    logits: torch.Tensor,
    sample_ids: list[str],
    output_dir: Path,
    start_index: int = 0,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = binarize(logits)

    for index in range(images.size(0)):
        metrics = calculate_all_metrics(logits[index : index + 1], masks[index : index + 1], compute_hd95=True)
        figure, axes = plt.subplots(1, 3, figsize=(16, 5))
        panels = (
            (denormalize(images[index]), None, f"Original RGB\n({sample_ids[index]})"),
            (masks[index].cpu().squeeze().numpy(), "gray", "Ground Truth Mask"),
            (
                predictions[index].cpu().squeeze().numpy(),
                "gray",
                f"Prediction | Dice: {metrics['dice']:.4f} | IoU: {metrics['iou']:.4f}\n"
                f"HD95: {metrics['hd95']:.2f} | MAE: {metrics['mae']:.4f} | F2: {metrics['f2']:.4f}",
            ),
        )
        for axis, (image, colormap, title) in zip(axes, panels):
            axis.imshow(image, cmap=colormap)
            axis.set_title(title)
            axis.axis("off")

        plt.tight_layout()
        plt.savefig(output_dir / f"viz_{start_index + index + 1:03d}_{sample_ids[index]}.png", bbox_inches="tight", dpi=300)
        plt.close(figure)


def clean_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state_dict.items():
        if SWA_METADATA_KEY in key:
            continue
        for prefix in CHECKPOINT_PREFIXES_TO_STRIP:
            key = key.replace(prefix, "")
        cleaned[key] = value
    return cleaned


def load_model_weights(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint required but not found at: {checkpoint_path}")
    print(f"Loading weights from: {checkpoint_path}")
    model.load_state_dict(clean_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True)))
    print("Weights loaded successfully!")


def resolve_checkpoint(checkpoint_dir: Path) -> Path:
    swa_checkpoint = checkpoint_dir / "best_swa_model.pth"
    return swa_checkpoint if swa_checkpoint.exists() else checkpoint_dir / "best_model.pth"


def parse_args() -> argparse.Namespace:
    available_datasets = ", ".join(sorted(discover_dataset_configs())) or "(none)"
    parser = argparse.ArgumentParser(description="Evaluate segmentation models")
    parser.add_argument("--model", type=str, default="vanilla", choices=MODEL_TYPES)
    parser.add_argument("--bond_dim", type=int, default=32)
    parser.add_argument("--loss", type=str, default="focal_tversky")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="kvasir_seg", help=f"Available: {available_datasets}")
    parser.add_argument("--split", type=str, default="test", choices=EVALUATION_SPLITS)
    parser.add_argument("--num_samples", type=int, default=10, choices=range(5, 11))
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--run", type=str, default=None, help="Checkpoint run name; overrides the derived name")
    return parser.parse_args()


@torch.inference_mode()
def evaluate_split(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    use_amp = device.type == "cuda"
    accumulator = MetricAccumulator(compute_hd95=True)
    for batch in tqdm(loader, desc="Evaluating"):
        images = batch["image"].to(device, non_blocking=True, memory_format=torch.channels_last)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
        accumulator.update(logits, batch["mask"].to(device, non_blocking=True))
    return accumulator.compute()


@torch.inference_mode()
def render_samples(model: torch.nn.Module, loader: DataLoader, device: torch.device, output_dir: Path) -> None:
    use_amp = device.type == "cuda"
    rendered = 0
    for batch in tqdm(loader, desc="Saving visualizations"):
        images = batch["image"].to(device, non_blocking=True, memory_format=torch.channels_last)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
        save_visualizations(images, batch["mask"].to(device), logits, batch["metadata"]["sample_id"], output_dir, rendered)
        rendered += images.size(0)


def main() -> None:
    args = parse_args()
    device = get_device()
    dataset_names = parse_dataset_arg(args.dataset)
    dataset_slug = slugify_dataset_arg(dataset_names)

    evaluation_dataset = getattr(build_splits(dataset_names), args.split)
    print(f"Dataset '{dataset_slug}': {len(evaluation_dataset)} {args.split} samples on {device}")

    run_name = args.run or build_run_name(dataset_slug, args.model, args.bond_dim, args.loss, args.seed, args.tag)
    model = build_segmentation_model(args.model, args.bond_dim, IMAGE_SIZE).to(device).to(memory_format=torch.channels_last)
    load_model_weights(model, resolve_checkpoint(Path("checkpoints") / run_name), device)
    model.eval()

    metrics = evaluate_split(model, DataLoader(evaluation_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True), device)
    print(" | ".join(f"{args.split} {name}: {value:.4f}" for name, value in metrics.items()))

    visualization_indices = torch.randperm(len(evaluation_dataset), generator=torch.Generator().manual_seed(VISUALIZATION_SEED))
    visualization_loader = DataLoader(
        Subset(evaluation_dataset, visualization_indices[: args.num_samples].tolist()), batch_size=1, shuffle=False
    )
    output_dir = Path(args.output_dir) if args.output_dir else Path("outputs") / "visualizations" / run_name
    render_samples(model, visualization_loader, device, output_dir)
    print(f"Visualizations saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
