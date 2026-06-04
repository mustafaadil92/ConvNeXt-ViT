from __future__ import annotations

import argparse
from pathlib import Path
from pprint import pprint

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def format_millions(n: int) -> str:
    return f"{n / 1_000_000:.3f}M"

def parse_args():
    parser = argparse.ArgumentParser(description="ConvNeXt-ViT PyTorch training entry")
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    parser.add_argument("--device", type=str, default=None, choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint (.pt/.pth) to resume from")
    parser.add_argument("--smoke-data", action="store_true", help="Build datasets/loaders and print info")
    parser.add_argument(
        "--smoke-train",
        action="store_true",
        help="Run a real training smoke test (model/loss/optimizer/fit)",
    )
    return parser.parse_args()


def main():
    from src.utils import (
        get_device,
        describe_device,
        seed_everything,
        load_yaml_config,
    )
    from src.datasets import build_datasets, build_dataloaders
    from src.models import build_model
    from src.train import (
        build_loss_fn,
        build_optimizer,
        build_scheduler,
        fit,
        save_history_csv,
        save_checkpoint,
        load_checkpoint,
    )

    args = parse_args()

    config = load_yaml_config(args.config)

    # CLI overrides (if provided)
    cfg_seed = int(config.get("seed", 42))
    cfg_device = str(config.get("device", "auto"))
    cfg_deterministic = bool(config.get("deterministic", False))

    seed = args.seed if args.seed is not None else cfg_seed
    device_request = args.device if args.device is not None else cfg_device
    deterministic = bool(args.deterministic or cfg_deterministic)

    seed_everything(seed=seed, deterministic=deterministic)
    device = get_device(device_request)
    info = describe_device(device)

    print("Training entry is working.")
    print("\nCLI arguments:")
    pprint(vars(args))

    print("\nLoaded config:")
    pprint(config)

    print("\nResolved runtime settings:")
    pprint(
        {
            "seed": seed,
            "device_request": device_request,
            "deterministic": deterministic,
        }
    )

    print("\nDevice info:")
    pprint(info)

    datasets = {}
    loaders = {}

    if args.smoke_data or args.smoke_train:
        print("\n[Smoke] Building datasets and dataloaders...")
        datasets = build_datasets(config)
        loaders = build_dataloaders(config)

        print("Dataset splits:", {k: len(v) for k, v in datasets.items()})
        print("Loader splits:", list(loaders.keys()))

        if "train" in loaders:
            batch = next(iter(loaders["train"]))
            images, labels = batch
            print("Train batch image shape:", tuple(images.shape))
            print("Train batch label shape:", tuple(labels.shape))

    if args.smoke_train:
        if "train" not in loaders:
            raise RuntimeError("Smoke train requested, but no train dataloader was built.")
        if "val" not in loaders:
            print("[Smoke] No val loader found; proceeding with train-only smoke run.")

        print("\n[Smoke] Building model/loss/optimizer/scheduler...")
        model = build_model(config).to(device)
        loss_fn = build_loss_fn(config)
        optimizer = build_optimizer(model, config)
        scheduler = build_scheduler(optimizer, config)

        total_params, trainable_params = count_parameters(model)
        print("Model parameters:")
        print(f"  total:     {total_params:,} ({format_millions(total_params)})")
        print(f"  trainable: {trainable_params:,} ({format_millions(trainable_params)})")

        print("Model class:", model.__class__.__name__)
        print("Loss:", loss_fn.__class__.__name__)
        print("Optimizer:", optimizer.__class__.__name__)
        print("Scheduler:", None if scheduler is None else scheduler.__class__.__name__)

        print("\n[Smoke] Running fit(...)")

        start_epoch = 1
        initial_history = []
        scaler_state_dict = None
        exp_name = str(config.get("experiment_name", "experiment"))
        log_dir = str(config.get("output", {}).get("log_dir", "outputs/logs"))
        ckpt_dir = str(config.get("output", {}).get("checkpoint_dir", "outputs/checkpoints"))
        last_ckpt_path = args.resume if args.resume else f"{ckpt_dir}/{exp_name}_last.pt"

        if args.resume:
            resume_path = Path(args.resume)
            if resume_path.exists():
                print(f"[Resume] Loading checkpoint: {args.resume}")
                ckpt = load_checkpoint(args.resume, device=device)
                model.load_state_dict(ckpt["model_state_dict"])
                if ckpt.get("optimizer_state_dict") is not None:
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                if ckpt.get("scheduler_state_dict") is not None and scheduler is not None:
                    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                start_epoch = int(ckpt.get("epoch") or 0) + 1
                initial_history = list(ckpt.get("extra", {}).get("history", []) or [])
                scaler_state_dict = ckpt.get("scaler_state_dict")
                print(f"[Resume] Starting from epoch {start_epoch}.")
            else:
                print(f"[Resume] Checkpoint not found: {args.resume}")
                print("[Resume] Saving initial checkpoint before training.")
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=None,
                    config=config,
                    out_path=last_ckpt_path,
                    epoch=0,
                    extra={"history": []},
                )

        train_cfg = config.get("train", {}) or {}
        checkpoint_interval = int(train_cfg.get("checkpoint_interval", 1))
        checkpoint_interval = max(1, checkpoint_interval)

        total_epochs = int(train_cfg.get("epochs", 1))
        if start_epoch > total_epochs:
            print(f"[Resume] Start epoch {start_epoch} exceeds total epochs {total_epochs}. Nothing to do.")
            return

        # Ensure there is always a baseline "last" checkpoint before the first epoch.
        if not Path(last_ckpt_path).exists():
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=None,
                config=config,
                out_path=last_ckpt_path,
                epoch=max(0, start_epoch - 1),
                extra={"history": initial_history},
            )

        def _on_epoch_end(epoch: int, history: list[dict[str, float]], scaler, metrics: dict[str, float]) -> None:
            _ = metrics
            csv_path = f"{log_dir}/{exp_name}_history.csv"
            save_history_csv(history, csv_path)

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                config=config,
                out_path=last_ckpt_path,
                epoch=epoch,
                extra={"history": history},
            )

            if epoch % checkpoint_interval == 0:
                ckpt_epoch = f"{ckpt_dir}/{exp_name}_epoch{epoch}.pt"
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    config=config,
                    out_path=ckpt_epoch,
                    epoch=epoch,
                    extra={"history": history},
                )

        # One explicit forward sanity check before fit
        model.eval()
        with __import__("torch").no_grad():
            sample_images, _ = next(iter(loaders["train"]))
            sample_images = sample_images.to(device, non_blocking=True)
            sample_logits = model(sample_images)
            print("Forward sanity check:")
            print("  input shape :", tuple(sample_images.shape))
            print("  logits shape:", tuple(sample_logits.shape))
        model.train()
        
        history = fit(
            model=model,
            train_loader=loaders["train"],
            val_loader=loaders.get("val"),
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            device=device,
            config=config,
            start_epoch=start_epoch,
            initial_history=initial_history,
            scaler_state_dict=scaler_state_dict,
            on_epoch_end=_on_epoch_end,
        )

        print("[Smoke] Training history:")
        pprint(history)
        print(f"[Smoke] Final history rows: {len(history)}")


if __name__ == "__main__":
    main()
