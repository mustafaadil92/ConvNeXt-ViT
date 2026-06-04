from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def _torch():
    # Lazy import avoids loading CUDA DLLs in spawned DataLoader workers.
    import torch

    return torch


def get_device(requested: str = "auto") -> "torch.device":
    """
    Resolve device from user request.

    requested:
        - "auto": use CUDA if available, else CPU
        - "cuda": force CUDA (raises if unavailable)
        - "cpu": force CPU
    """
    requested = (requested or "auto").strip().lower()

    torch = _torch()

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")

    if requested == "cpu":
        return torch.device("cpu")

    raise ValueError(f"Unsupported device option: {requested}")


def describe_device(device: "torch.device") -> dict:
    """
    Return a small dictionary with device info for logging/debugging.
    """
    torch = _torch()

    info = {
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_build": torch.version.cuda,
    }

    if device.type == "cuda" and torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()

    return info
