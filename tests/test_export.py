import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import onnxruntime as ort
import pytest
import torch

from export import (
    build_export_model,
    export_model_to_onnx,
    benchmark_pytorch_throughput,
    benchmark_onnx_throughput
)


@pytest.fixture
def dummy_input_tensor() -> torch.Tensor:
    return torch.randn(1, 3, 64, 64)


@pytest.fixture
def minimal_pytorch_model() -> torch.nn.Module:
    class MinimalModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 1, kernel_size=3, padding=1)
            
        def forward(self, x):
            return self.conv(x)
            
    return MinimalModel()


def test_build_export_model_returns_correct_instances():
    vanilla_model = build_export_model("vanilla", bond_dim=16)
    assert vanilla_model.__class__.__name__ == "UNet"
    
    hybrid_model = build_export_model("hybrid", bond_dim=16)
    assert hybrid_model.__class__.__name__ == "UNetDeepHybrid"
    assert getattr(hybrid_model, "transform_1024", None) is not None
    
    deep_hybrid_model = build_export_model("deep_hybrid", bond_dim=16)
    assert getattr(deep_hybrid_model, "transform_512", None) is not None


def test_export_model_to_onnx_creates_valid_file(minimal_pytorch_model, dummy_input_tensor, tmp_path):
    output_onnx_path = tmp_path / "test_model.onnx"
    
    export_model_to_onnx(minimal_pytorch_model, dummy_input_tensor, output_onnx_path)
    
    assert output_onnx_path.exists()
    assert output_onnx_path.stat().st_size > 0
    
    session = ort.InferenceSession(str(output_onnx_path), providers=["CPUExecutionProvider"])
    assert session is not None
    
    input_details = session.get_inputs()[0]
    assert input_details.name == "input_image"


def test_benchmark_pytorch_throughput_returns_valid_fps(minimal_pytorch_model, dummy_input_tensor):
    iterations = 5
    fps = benchmark_pytorch_throughput(minimal_pytorch_model, dummy_input_tensor, total_iterations=iterations)
    
    assert isinstance(fps, float)
    assert fps > 0.0


def test_benchmark_onnx_throughput_returns_valid_fps(minimal_pytorch_model, dummy_input_tensor, tmp_path):
    output_onnx_path = tmp_path / "benchmark_test.onnx"
    export_model_to_onnx(minimal_pytorch_model, dummy_input_tensor, output_onnx_path)
    
    numpy_input = dummy_input_tensor.numpy()
    iterations = 5
    fps = benchmark_onnx_throughput(output_onnx_path, numpy_input, total_iterations=iterations)
    
    assert isinstance(fps, float)
    assert fps > 0.0
