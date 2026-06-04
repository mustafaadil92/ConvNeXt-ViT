from __future__ import annotations

import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LOGS_ROOT = Path("outputs/logs/LC25000")
FIGURES_ROOT = Path("outputs/aggregate_figures/LC25000")
REQUIRED_COLUMNS = ("train_loss", "train_acc", "val_loss", "val_acc")
FILENAME_PATTERN = re.compile(r"(?P<exp>.+)_fold(?P<fold>\d+)_history\.csv$")


def discover_history_groups(logs_root: Path) -> dict[str, dict[str, list[tuple[int, Path]]]]:
    groups: dict[str, dict[str, list[tuple[int, Path]]]] = {}

    for csv_path in logs_root.rglob("*_history.csv"):
        if csv_path.name.endswith("_smoke_history.csv"):
            continue

        match = FILENAME_PATTERN.match(csv_path.name)
        if not match:
            continue

        try:
            task_name = str(csv_path.parent.relative_to(logs_root))
        except ValueError:
            task_name = ""

        task_name = task_name if task_name else "multiclass"
        model_name = csv_path.parent.name
        fold_number = int(match.group("fold"))

        groups.setdefault(task_name, {}).setdefault(model_name, []).append((fold_number, csv_path))

    return groups


def read_histories(files: list[tuple[int, Path]]) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    train_acc, val_acc = [], []
    train_loss, val_loss = [], []

    for _, csv_path in sorted(files, key=lambda item: item[0]):
        df = pd.read_csv(csv_path)
        missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns {missing} in {csv_path}")

        train_acc.append(df["train_acc"].to_numpy(dtype=float))
        val_acc.append(df["val_acc"].to_numpy(dtype=float))
        train_loss.append(df["train_loss"].to_numpy(dtype=float))
        val_loss.append(df["val_loss"].to_numpy(dtype=float))

    return train_acc, val_acc, train_loss, val_loss


def pad_histories(arrays: list[np.ndarray], target_len: int) -> np.ndarray:
    padded_rows = []
    for arr in arrays:
        if arr.shape[0] > target_len:
            raise ValueError("Found history longer than expected max epoch length.")
        if arr.shape[0] == target_len:
            padded_rows.append(arr)
            continue
        padded = np.full(target_len, np.nan, dtype=float)
        padded[: arr.shape[0]] = arr
        padded_rows.append(padded)
    return np.vstack(padded_rows)


def compute_stats(arrays: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.nanmean(arrays, axis=0),
        np.nanmin(arrays, axis=0),
        np.nanmax(arrays, axis=0),
    )


def plot_accuracy(
    epochs: np.ndarray,
    train_stats: tuple[np.ndarray, np.ndarray, np.ndarray],
    val_stats: tuple[np.ndarray, np.ndarray, np.ndarray],
    out_path: Path,
) -> None:
    train_mean, train_min, train_max = train_stats
    val_mean, val_min, val_max = val_stats

    plt.figure(figsize=(4, 3))
    plt.plot(epochs, train_mean)
    plt.fill_between(epochs, train_min, train_max, alpha=0.2)
    plt.plot(epochs, val_mean)
    plt.fill_between(epochs, val_min, val_max, alpha=0.2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.xlim(1, int(epochs[-1]))
    plt.xticks(range(1, int(epochs[-1]) + 1, max(1, int(np.ceil(len(epochs) / 10)))), fontsize=12)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_loss(
    epochs: np.ndarray,
    train_stats: tuple[np.ndarray, np.ndarray, np.ndarray],
    val_stats: tuple[np.ndarray, np.ndarray, np.ndarray],
    global_max_loss: float,
    out_path: Path,
) -> None:
    train_mean, train_min, train_max = train_stats
    val_mean, val_min, val_max = val_stats

    plt.figure(figsize=(4, 3))
    plt.plot(epochs, train_mean)
    plt.fill_between(epochs, train_min, train_max, alpha=0.2)
    plt.plot(epochs, val_mean)
    plt.fill_between(epochs, val_min, val_max, alpha=0.2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.xlim(1, int(epochs[-1]))
    plt.xticks(range(1, int(epochs[-1]) + 1, max(1, int(np.ceil(len(epochs) / 10)))), fontsize=12)
    plt.ylim(0, global_max_loss if global_max_loss > 0 else 1.0)
    plt.grid(True)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def create_legend(out_path: Path) -> None:
    plt.figure(figsize=(6, 2.5))
    (line1,) = plt.plot([], [])
    patch1 = plt.fill_between([], [], [], alpha=0.2)
    (line2,) = plt.plot([], [])
    patch2 = plt.fill_between([], [], [], alpha=0.2)
    plt.legend(
        [line1, patch1, line2, patch2],
        ["Train Mean", "Train Range", "Validation Mean", "Validation Range"],
        loc="center",
        frameon=False,
        fontsize=12,
        ncol=2,
    )
    plt.axis("off")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def generate_figures() -> list[Path]:
    created_files: list[Path] = []
    history_groups = discover_history_groups(LOGS_ROOT)
    if not history_groups:
        raise ValueError(f"No history CSV files found under {LOGS_ROOT}")

    for task_name, models in history_groups.items():
        task_root = FIGURES_ROOT / task_name
        accuracy_dir = task_root / "accuracy"
        loss_dir = task_root / "loss"

        global_max_loss = 0.0
        model_arrays: dict[str, tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]] = {}
        max_epochs = 0

        for model_name, files in models.items():
            train_acc_list, val_acc_list, train_loss_list, val_loss_list = read_histories(files)
            model_max_epochs = max(len(arr) for arr in train_acc_list)
            max_epochs = max(max_epochs, model_max_epochs)
            global_max_loss = max(
                global_max_loss,
                max(float(np.nanmax(arr)) for arr in train_loss_list),
                max(float(np.nanmax(arr)) for arr in val_loss_list),
            )
            model_arrays[model_name] = (train_acc_list, val_acc_list, train_loss_list, val_loss_list)

        epochs = np.arange(1, max_epochs + 1)

        for model_name, arrays in model_arrays.items():
            train_acc_list, val_acc_list, train_loss_list, val_loss_list = arrays

            train_acc = pad_histories(train_acc_list, max_epochs)
            val_acc = pad_histories(val_acc_list, max_epochs)
            train_loss = pad_histories(train_loss_list, max_epochs)
            val_loss = pad_histories(val_loss_list, max_epochs)

            accuracy_path = accuracy_dir / f"{model_name} accuracy.png"
            loss_path = loss_dir / f"{model_name} loss.png"

            plot_accuracy(
                epochs=epochs,
                train_stats=compute_stats(train_acc),
                val_stats=compute_stats(val_acc),
                out_path=accuracy_path,
            )
            plot_loss(
                epochs=epochs,
                train_stats=compute_stats(train_loss),
                val_stats=compute_stats(val_loss),
                global_max_loss=global_max_loss,
                out_path=loss_path,
            )

            created_files.extend([accuracy_path, loss_path])

        legend_path = task_root / "legend.png"
        create_legend(legend_path)
        created_files.append(legend_path)

    return created_files


def main() -> None:
    created_files = generate_figures()
    print("Generated LC25000 aggregate figures:")
    for path in created_files:
        print(path.as_posix())


if __name__ == "__main__":
    main()
