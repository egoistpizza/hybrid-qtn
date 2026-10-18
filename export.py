"""ONNX export of a segmentation model and a PyTorch vs ONNX Runtime throughput benchmark."""

import argparse
import time
from pathlib import Path
from typing import Callable

import numpy as np
import onnxruntime as ort
import torch

from models.factory import MODEL_TYPES, add_mps_arguments, build_segmentation_model
from utils import get_device
from utils.run_naming import mps_variant_suffix


def measure_throughput(
    run_once: Callable[[], object], iterations: int, warmup: int = 10, synchronize: Callable[[], None] = lambda: None
) -> float:
    """Calls of ``run_once`` per second over ``iterations`` timed calls, after ``warmup`` untimed ones.

    ``synchronize`` waits for queued asynchronous work, so the timer brackets completed calls.
    """
    for _ in range(warmup):
        run_once()
    synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        run_once()
    synchronize()
    return iterations / (time.perf_counter() - start)


def benchmark_pytorch_throughput(model: torch.nn.Module, dummy_input: torch.Tensor, total_iterations: int = 100) -> float:
    """Forward passes per second in PyTorch."""
    model.eval()
    device = dummy_input.device
    synchronize = (lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else (lambda: None)
    with torch.inference_mode():
        return measure_throughput(lambda: model(dummy_input), total_iterations, synchronize=synchronize)


def benchmark_onnx_throughput(onnx_file_path: Path, numpy_dummy_input: np.ndarray, total_iterations: int = 100) -> float:
    """Forward passes per second in ONNX Runtime, on CUDA if available."""
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(onnx_file_path), sess_options=session_options, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    feed = {session.get_inputs()[0].name: numpy_dummy_input}
    return measure_throughput(lambda: session.run(None, feed), total_iterations)


def export_model_to_onnx(model: torch.nn.Module, dummy_input: torch.Tensor, output_file_path: Path) -> None:
    """Export with opset 18 and a dynamic batch dimension; the spatial size stays fixed."""
    model.eval()
    
    torch.onnx.export(
        model,
        dummy_input,
        str(output_file_path),
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input_image"],
        output_names=["segmentation_mask"],
        dynamic_axes={
            "input_image": {0: "batch_size"},
            "segmentation_mask": {0: "batch_size"}
        }
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="hybrid", choices=MODEL_TYPES)
    parser.add_argument("--bond_dim", type=int, default=32)
    add_mps_arguments(parser)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="outputs/onnx")
    return parser.parse_args()

def main() -> None:
    args = parse_arguments()
    compute_device = get_device()
    
    output_directory = Path(args.output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    
    variant = mps_variant_suffix(args.axial_operator, args.positional_encoding)
    onnx_filename = f"{args.model}_b{args.bond_dim}{variant}_s{args.image_size}.onnx"
    onnx_file_path = output_directory / onnx_filename
    
    model = build_segmentation_model(
        args.model, args.bond_dim, args.image_size, args.axial_operator, args.positional_encoding
    ).to(compute_device)
    model = model.to(memory_format=torch.channels_last)
    
    dummy_input_tensor = torch.randn(args.batch_size, 3, args.image_size, args.image_size, device=compute_device)
    dummy_input_tensor = dummy_input_tensor.to(memory_format=torch.channels_last)
    
    export_model_to_onnx(model, dummy_input_tensor, onnx_file_path)
    
    pytorch_fps = benchmark_pytorch_throughput(model, dummy_input_tensor, args.iterations)
    
    numpy_dummy_input = dummy_input_tensor.cpu().numpy()
    onnx_fps = benchmark_onnx_throughput(onnx_file_path, numpy_dummy_input, args.iterations)
    
    speedup_ratio = onnx_fps / pytorch_fps

    print(f"\n--- BENCHMARK RESULTS ({args.iterations} Iterations) ---")
    print(f"PyTorch FPS : {pytorch_fps:.2f}")
    print(f"ONNX FPS    : {onnx_fps:.2f}")
    print(f"Speedup     : {speedup_ratio:.2f}x")

if __name__ == "__main__":
    main()
