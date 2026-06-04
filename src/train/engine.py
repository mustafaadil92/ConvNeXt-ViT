from __future__ import annotations

import gc
from typing import Any
from time import perf_counter

import torch
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - fallback when tqdm is unavailable
    tqdm = None

from .metrics import summarize_epoch_metrics


def _amp_enabled(config: dict, device: torch.device) -> bool:
    train_cfg = config.get("train", {}) or {}
    return bool(train_cfg.get("amp", True)) and device.type == "cuda"


def _cleanup_interval(config: dict, key: str, default: int) -> int:
    train_cfg = config.get("train", {}) or {}
    value = int(train_cfg.get(key, default))
    return max(0, value)


def _cleanup_memory(config: dict, device: torch.device, step: int) -> None:
    gc_every = _cleanup_interval(config, "memory_cleanup_interval", 10)
    cuda_cache_every = _cleanup_interval(config, "empty_cuda_cache_interval", 50)

    if gc_every > 0 and step % gc_every == 0:
        gc.collect()

    if device.type == "cuda" and cuda_cache_every > 0 and step % cuda_cache_every == 0:
        torch.cuda.empty_cache()


def train_one_epoch(
    model: Any,
    dataloader: Any,
    optimizer: Any,
    loss_fn: Any,
    device: Any,
    epoch: int,
    config: dict,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, float]:
    """
    Minimal real training loop.
    """
    model.train()

    use_amp = _amp_enabled(config, device)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    iterator = enumerate(dataloader, start=1)
    total_steps = len(dataloader) if hasattr(dataloader, "__len__") else None

    if tqdm is not None:
        iterator = tqdm(
            iterator,
            total=total_steps,
            desc=f"Epoch {epoch} [train]",
            leave=False,
            dynamic_ncols=True,
        )

    for batch_idx, (images, labels) in iterator:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast("cuda"):
                logits = model(images)
                loss = loss_fn(logits, labels)

            if scaler is None:
                scaler = torch.amp.GradScaler("cuda")

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

        batch_size = labels.size(0)
        total_loss += float(loss.detach().item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_samples += int(batch_size)
        if tqdm is not None:
            running_loss = total_loss / total_samples if total_samples > 0 else 0.0
            running_acc = total_correct / total_samples if total_samples > 0 else 0.0
            iterator.set_postfix(
                step=batch_idx,
                loss=f"{running_loss:.4f}",
                acc=f"{running_acc:.4f}",
            )

        # Free per-batch tensor references promptly to reduce RAM pressure.
        del images, labels, logits, loss
        _cleanup_memory(config=config, device=device, step=batch_idx)

    train_loss = total_loss / total_samples if total_samples > 0 else 0.0
    train_acc = total_correct / total_samples if total_samples > 0 else 0.0

    metrics = {
        "epoch": float(epoch),
        "train_loss": float(train_loss),
        "train_acc": float(train_acc),
    }
    return summarize_epoch_metrics(metrics)


@torch.no_grad()
def validate_one_epoch(
    model: Any,
    dataloader: Any,
    loss_fn: Any,
    device: Any,
    epoch: int,
    config: dict,
) -> dict[str, float]:
    """
    Minimal real validation loop.
    """
    _ = config
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    iterator = dataloader
    total_steps = len(dataloader) if hasattr(dataloader, "__len__") else None
    if tqdm is not None:
        iterator = tqdm(
            dataloader,
            total=total_steps,
            desc=f"Epoch {epoch} [val]",
            leave=False,
            dynamic_ncols=True,
        )

    for batch_idx, (images, labels) in enumerate(iterator, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = loss_fn(logits, labels)

        batch_size = labels.size(0)
        total_loss += float(loss.detach().item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_samples += int(batch_size)
        if tqdm is not None:
            running_loss = total_loss / total_samples if total_samples > 0 else 0.0
            running_acc = total_correct / total_samples if total_samples > 0 else 0.0
            iterator.set_postfix(loss=f"{running_loss:.4f}", acc=f"{running_acc:.4f}")

        del images, labels, logits, loss
        _cleanup_memory(config=config, device=device, step=batch_idx)

    val_loss = total_loss / total_samples if total_samples > 0 else 0.0
    val_acc = total_correct / total_samples if total_samples > 0 else 0.0

    metrics = {
        "epoch": float(epoch),
        "val_loss": float(val_loss),
        "val_acc": float(val_acc),
    }
    return summarize_epoch_metrics(metrics)


def fit(
    model: Any,
    train_loader: Any,
    val_loader: Any,
    optimizer: Any,
    scheduler: Any,
    loss_fn: Any,
    device: Any,
    config: dict,
    start_epoch: int = 1,
    initial_history: list[dict[str, float]] | None = None,
    scaler_state_dict: dict | None = None,
    on_epoch_end: Any | None = None,
) -> list[dict[str, float]]:
    """
    Minimal fit loop orchestrator.
    """
    epochs = int(config.get("train", {}).get("epochs", 1))
    history: list[dict[str, float]] = list(initial_history or [])

    scaler = torch.amp.GradScaler("cuda") if _amp_enabled(config, device) else None
    if scaler is not None and scaler_state_dict:
        scaler.load_state_dict(scaler_state_dict)

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = perf_counter()
        print(f"\nEpoch {epoch}/{epochs}", flush=True)

        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            epoch=epoch,
            config=config,
            scaler=scaler,
        )

        val_metrics = {}
        if val_loader is not None:
            val_metrics = validate_one_epoch(
                model=model,
                dataloader=val_loader,
                loss_fn=loss_fn,
                device=device,
                epoch=epoch,
                config=config,
            )

        if scheduler is not None:
            scheduler.step()

        row = {**train_metrics, **val_metrics}
        history.append(row)

        elapsed = perf_counter() - epoch_start
        train_loss = row.get("train_loss", 0.0)
        train_acc = row.get("train_acc", 0.0)
        if "val_loss" in row and "val_acc" in row:
            print(
                f"Epoch {epoch}/{epochs} done in {elapsed:.1f}s | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={row['val_loss']:.4f} val_acc={row['val_acc']:.4f}",
                flush=True,
            )
        else:
            print(
                f"Epoch {epoch}/{epochs} done in {elapsed:.1f}s | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}",
                flush=True,
            )

        if on_epoch_end is not None:
            on_epoch_end(
                epoch=epoch,
                history=history,
                scaler=scaler,
                metrics=row,
            )

    return history
