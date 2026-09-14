"""Benchmark: AMP (FP16) vs. FP32 inference precision.

Loads one checkpoint once, then runs the full validation set twice —
once under torch.autocast(enabled=True) and once with it disabled — and
reports the deviation in Dice/IoU/MAE/F2/HD95 between the two.

Usage (same --model/--bond_dim/--dataset flags as evaluate.py; --run must
name the exact checkpoint directory under checkpoints/, e.g. what train.py
printed as config["checkpoint_dir"]):

    python benchmark_precision.py --model hybrid --bond_dim 32 \\
        --dataset kvasir_seg \\
        --run kvasir_seg__hybrid_unet_b32_ckpt__focal_tversky__hybrid_sota_baseline
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import (
    build_train_val,
    discover_dataset_configs,
    parse_dataset_arg,
    slugify_dataset_arg,
)
from evaluate import build_model, load_model_weights
from utils import get_device
from utils.metrics import calculate_all_metrics

METRIC_KEYS = ("dice", "iou", "mae", "f2", "hd95")


def run_validation_pass(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    compute_hd95: bool,
) -> Dict[str, float]:
    aggregated = {k: 0.0 for k in METRIC_KEYS}
    label = "AMP (FP16)" if use_amp else "FP32"

    with torch.inference_mode():
        for batch in tqdm(val_loader, desc=f"Validating [{label}]"):
            images = batch["image"].to(device, non_blocking=True, memory_format=torch.channels_last)
            masks = batch["mask"].to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images)

            batch_metrics = calculate_all_metrics(outputs, masks, compute_hd95=compute_hd95)
            for k in aggregated:
                aggregated[k] += batch_metrics[k]

    for k in aggregated:
        aggregated[k] /= len(val_loader)
    return aggregated


def parse_args() -> argparse.Namespace:
    available_datasets = ", ".join(sorted(discover_dataset_configs())) or "(none)"

    parser = argparse.ArgumentParser(
        description="Benchmark AMP (FP16) vs FP32 inference precision on a fixed checkpoint."
    )
    parser.add_argument("--model", type=str, default="hybrid", choices=["vanilla", "hybrid", "deep_hybrid"])
    parser.add_argument("--bond_dim", type=int, default=32)
    parser.add_argument("--dataset", type=str, default="kvasir_seg", help=f"Available: {available_datasets}")
    parser.add_argument(
        "--run", type=str, required=True,
        help="Exact checkpoint directory name under checkpoints/ (see evaluate.py --run / config['checkpoint_dir']).",
    )
    parser.add_argument(
        "--skip_hd95", action="store_true",
        help="Skip HD95 (slow, CPU/scipy-based) to speed up the benchmark.",
    )
    parser.add_argument("--output_dir", type=str, default="outputs/precision_benchmark")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()

    if device.type != "cuda":
        print(
            f"[!] WARNING: device is '{device.type}', not 'cuda'. This codebase's AMP path "
            "(torch.autocast + GradScaler, see train.py/evaluate.py) is designed and used "
            "for CUDA. Numbers collected here do not reflect that AMP behavior."
        )

    dataset_names = parse_dataset_arg(args.dataset)
    dataset_slug = slugify_dataset_arg(dataset_names)
    _, val_ds = build_train_val(dataset_names)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)
    print(f"Dataset '{dataset_slug}': {len(val_ds)} validation samples")

    model = build_model(args.model, args.bond_dim)
    model = model.to(device).to(memory_format=torch.channels_last)

    checkpoint_dir = Path("checkpoints") / args.run
    swa_path = checkpoint_dir / "best_swa_model.pth"
    standard_path = checkpoint_dir / "best_model.pth"
    load_path = swa_path if swa_path.exists() else standard_path
    load_model_weights(model, load_path, device)
    model.eval()

    compute_hd95 = not args.skip_hd95

    amp_metrics = run_validation_pass(model, val_loader, device, use_amp=True, compute_hd95=compute_hd95)
    fp32_metrics = run_validation_pass(model, val_loader, device, use_amp=False, compute_hd95=compute_hd95)

    print("\n" + "=" * 70)
    print(f"{'Metric':<10} {'AMP (FP16)':>15} {'FP32':>15} {'Delta':>15} {'Delta %':>10}")
    print("-" * 70)
    deltas = {}
    for k in METRIC_KEYS:
        if k == "hd95" and not compute_hd95:
            continue
        amp_value, fp32_value = amp_metrics[k], fp32_metrics[k]
        delta = amp_value - fp32_value
        delta_pct = (delta / fp32_value * 100.0) if fp32_value != 0 else float("nan")
        deltas[k] = {"absolute": delta, "percent": delta_pct}
        print(f"{k:<10} {amp_value:>15.6f} {fp32_value:>15.6f} {delta:>+15.6f} {delta_pct:>+9.3f}%")
    print("=" * 70)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"{args.run}.json"
    with result_path.open("w") as fh:
        json.dump(
            {
                "run": args.run,
                "model": args.model,
                "bond_dim": args.bond_dim,
                "dataset": dataset_slug,
                "checkpoint": str(load_path),
                "device": str(device),
                "amp_metrics": amp_metrics,
                "fp32_metrics": fp32_metrics,
                "deltas": deltas,
            },
            fh,
            indent=2,
        )
    print(f"\nResults saved to: {result_path.absolute()}")


if __name__ == "__main__":
    main()
