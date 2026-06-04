from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any,
    scaler: torch.amp.GradScaler | None,
    config: dict,
    out_path: str | Path,
    epoch: int | None = None,
    extra: dict | None = None,
) -> Path:
    """
    Save a training checkpoint (.pt).
    """
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "config": config,
        "epoch": epoch,
        "extra": extra or {},
    }

    torch.save(payload, p)
    return p


def load_checkpoint(
    path: str | Path,
    device: torch.device | str | None = None,
) -> dict:
    """
    Load a training checkpoint (.pt).
    """
    map_location = device if device is not None else "cpu"
    return torch.load(path, map_location=map_location)
