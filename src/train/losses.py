from __future__ import annotations

import torch.nn as nn


def get_loss_config(config: dict) -> dict:
    """
    Normalize loss configuration from project config.
    """
    train_cfg = config.get("train", {}) or {}
    loss_cfg = train_cfg.get("loss", {}) or {}

    if not loss_cfg:
        return {
            "name": "cross_entropy",
            "label_smoothing": 0.0,
        }

    return {
        "name": str(loss_cfg.get("name", "cross_entropy")).lower(),
        "label_smoothing": float(loss_cfg.get("label_smoothing", 0.0)),
    }


def build_loss_fn(config: dict):
    """
    Build a torch loss function from config.
    """
    loss_cfg = get_loss_config(config)
    name = loss_cfg["name"]

    if name == "cross_entropy":
        return nn.CrossEntropyLoss(
            label_smoothing=float(loss_cfg.get("label_smoothing", 0.0))
        )

    raise ValueError(f"Unsupported loss function: {name}")