from __future__ import annotations

import torch


def get_optimizer_config(config: dict) -> dict:
    """
    Return normalized optimizer config from the project config.
    """
    opt_cfg = config.get("optimizer", {}) or {}

    return {
        "name": str(opt_cfg.get("name", "adamw")).lower(),
        "lr": float(opt_cfg.get("lr", 1e-4)),
        "weight_decay": float(opt_cfg.get("weight_decay", 0.0)),
    }


def get_scheduler_config(config: dict) -> dict:
    """
    Return normalized scheduler config from the project config.
    """
    sch_cfg = config.get("scheduler", {}) or {}

    out = {"name": str(sch_cfg.get("name", "none")).lower()}

    for key, value in sch_cfg.items():
        if key == "name":
            continue
        out[str(key)] = value

    return out


def build_optimizer(model, config: dict):
    """
    Build torch optimizer from config.
    """
    opt_cfg = get_optimizer_config(config)
    name = opt_cfg["name"]

    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=float(opt_cfg["lr"]),
            weight_decay=float(opt_cfg["weight_decay"]),
        )

    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(optimizer, config: dict):
    """
    Build torch scheduler from config.
    Supported:
    - none
    - cosine (CosineAnnealingLR)
    """
    sch_cfg = get_scheduler_config(config)
    name = sch_cfg["name"]

    if name in {"none", "", "null"}:
        return None

    if name == "cosine":
        t_max = int(sch_cfg.get("t_max", 10))
        eta_min = float(sch_cfg.get("min_lr", 0.0))
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=t_max,
            eta_min=eta_min,
        )

    raise ValueError(f"Unsupported scheduler: {name}")