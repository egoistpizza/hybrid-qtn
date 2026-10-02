import torch

BYTES_PER_MEGABYTE = 1024**2


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def reset_peak_vram(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def vram_metrics(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {"Peak_VRAM_MB": 0.0, "vram_allocated_mb": 0.0, "vram_reserved_mb": 0.0}
    return {
        "Peak_VRAM_MB": torch.cuda.max_memory_allocated(device) / BYTES_PER_MEGABYTE,
        "vram_allocated_mb": torch.cuda.memory_allocated(device) / BYTES_PER_MEGABYTE,
        "vram_reserved_mb": torch.cuda.memory_reserved(device) / BYTES_PER_MEGABYTE,
    }
