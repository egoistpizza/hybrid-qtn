import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from utils import get_device


def build_export_model(model_type: str, bond_dim: int) -> torch.nn.Module:
    if model_type == "deep_hybrid":
        from models.unet_hybrid import UNetDeepHybrid
        from models.mps_layer import MPSBottleneck
        
        mps_512 = MPSBottleneck(in_channels=512, bond_dim=bond_dim, height=64, width=64)
        mps_1024 = MPSBottleneck(in_channels=1024, bond_dim=bond_dim, height=32, width=32)
        return UNetDeepHybrid(in_channels=3, out_channels=1, transform_512=mps_512, transform_1024=mps_1024)
        
    if model_type == "hybrid":
        from models.unet_hybrid import UNetDeepHybrid
        from models.mps_layer import MPSBottleneck
        
        mps_1024 = MPSBottleneck(in_channels=1024, bond_dim=bond_dim, height=32, width=32)
        return UNetDeepHybrid(in_channels=3, out_channels=1, transform_512=None, transform_1024=mps_1024)
        
    from models.unet_classic import UNet
    return UNet(in_channels=3, out_channels=1)

def benchmark_pytorch_throughput(model: torch.nn.Module, dummy_input: torch.Tensor, total_iterations: int = 100) -> float:
    model.eval()
    compute_device = dummy_input.device
    
    with torch.inference_mode():
        for _ in range(10):
            _ = model(dummy_input)
            
        if compute_device.type == "cuda":
            torch.cuda.synchronize(compute_device)
            
        start_timestamp = time.perf_counter()
        
        for _ in range(total_iterations):
            _ = model(dummy_input)
            
        if compute_device.type == "cuda":
            torch.cuda.synchronize(compute_device)
            
        end_timestamp = time.perf_counter()
        
    total_execution_time = end_timestamp - start_timestamp
    frames_per_second = total_iterations / total_execution_time
    
    return frames_per_second

def benchmark_onnx_throughput(onnx_file_path: Path, numpy_dummy_input: np.ndarray, total_iterations: int = 100) -> float:
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    execution_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    inference_session = ort.InferenceSession(str(onnx_file_path), sess_options=session_options, providers=execution_providers)
    
    input_tensor_name = inference_session.get_inputs()[0].name
    
    for _ in range(10):
        _ = inference_session.run(None, {input_tensor_name: numpy_dummy_input})
        
    start_timestamp = time.perf_counter()
    
    for _ in range(total_iterations):
        _ = inference_session.run(None, {input_tensor_name: numpy_dummy_input})
        
    end_timestamp = time.perf_counter()
    
    total_execution_time = end_timestamp - start_timestamp
    frames_per_second = total_iterations / total_execution_time
    
    return frames_per_second

def export_model_to_onnx(model: torch.nn.Module, dummy_input: torch.Tensor, output_file_path: Path) -> None:
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
    parser.add_argument("--model", type=str, default="hybrid", choices=["vanilla", "hybrid", "deep_hybrid"])
    parser.add_argument("--bond_dim", type=int, default=32)
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
    
    onnx_filename = f"{args.model}_b{args.bond_dim}_s{args.image_size}.onnx"
    onnx_file_path = output_directory / onnx_filename
    
    model = build_export_model(args.model, args.bond_dim).to(compute_device)
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
