from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PREDICTIONS_ROOT = Path("outputs/predictions/LC25000")
FIGURES_ROOT = Path("outputs/aggregate_figures/LC25000")
MODEL_ORDER = ("convnext", "convnext-vit", "vit-s")
TASK_CONFIG = {
    "multiclass": {
        "pred_root": PREDICTIONS_ROOT,
        "labels": [
            "0",
            "1",
            "2",
            "3",
            "4",
        ],
    },
}


def _read_confusion_matrix(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        rows = list(reader)

    header = rows[0][1:]
    labels = []
    matrix_rows = []
    for row in rows[1:]:
        labels.append(row[0])
        matrix_rows.append([float(value) for value in row[1:]])

    if labels != header:
        raise ValueError(f"Row/column label mismatch in {path}")

    return labels, np.array(matrix_rows, dtype=float)


def _remap_matrix(labels: list[str], matrix: np.ndarray, target_labels: list[str]) -> np.ndarray:
    remapped = np.zeros((len(target_labels), len(target_labels)), dtype=float)
    source_to_index = {label: idx for idx, label in enumerate(labels)}

    for dst_i, row_label in enumerate(target_labels):
        if row_label not in source_to_index:
            raise ValueError(f"Missing row label '{row_label}' in source matrix.")
        for dst_j, col_label in enumerate(target_labels):
            if col_label not in source_to_index:
                raise ValueError(f"Missing column label '{col_label}' in source matrix.")
            remapped[dst_i, dst_j] = matrix[source_to_index[row_label], source_to_index[col_label]]

    return remapped


def _load_model_fold_matrices(model_dir: Path, target_labels: list[str]) -> np.ndarray:
    matrix_paths = sorted(model_dir.glob("*_test_confusion_matrix.csv"))
    if not matrix_paths:
        raise FileNotFoundError(f"No confusion matrix CSV files found in {model_dir}")

    matrices = []
    for path in matrix_paths:
        labels, matrix = _read_confusion_matrix(path)
        matrices.append(_remap_matrix(labels, matrix, target_labels))
    return np.stack(matrices, axis=0)


def _plot_confusion_matrix(
    mean_matrix: np.ndarray,
    std_matrix: np.ndarray,
    labels: list[str],
    title: str,
    out_path: Path,
) -> None:
    n = len(labels)
    fig_size = 5 if n <= 2 else 12
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    ax.imshow(mean_matrix, cmap="Blues")
    numeric_labels = [str(index) for index in range(n)]

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    label_fontsize = 22 if n <= 2 else 25
    axis_fontsize = 24 if n <= 2 else 27
    title_fontsize = 24 if n <= 2 else 28
    cell_fontsize = 22 if n <= 2 else 23

    ax.set_xticklabels(numeric_labels, rotation=0, ha="center", fontsize=label_fontsize)
    ax.set_yticklabels(numeric_labels, fontsize=label_fontsize)
    ax.set_xlabel("Predicted label", fontsize=axis_fontsize)
    ax.set_ylabel("True label", fontsize=axis_fontsize)
    ax.set_title(title, fontsize=title_fontsize)

    threshold = float(np.nanmax(mean_matrix)) / 2.0 if np.size(mean_matrix) else 0.0
    for i in range(n):
        for j in range(n):
            mean_value = mean_matrix[i, j]
            std_value = std_matrix[i, j]
            text_color = "white" if mean_value > threshold else "black"
            ax.text(
                j,
                i,
                f"{mean_value:.1f}\n± {std_value:.1f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=cell_fontsize,
            )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def generate_confusion_matrices() -> list[Path]:
    created_files: list[Path] = []

    for task_name, task_config in TASK_CONFIG.items():
        pred_root = Path(task_config["pred_root"])
        target_labels = list(task_config["labels"])
        output_dir = FIGURES_ROOT / task_name / "confusion_matrix"

        for model_name in MODEL_ORDER:
            model_dir = pred_root / model_name
            fold_matrices = _load_model_fold_matrices(model_dir, target_labels)
            mean_matrix = np.mean(fold_matrices, axis=0)
            std_matrix = np.std(fold_matrices, axis=0, ddof=1) if fold_matrices.shape[0] > 1 else np.zeros_like(mean_matrix)

            out_path = output_dir / f"{model_name} confusion matrix.png"
            _plot_confusion_matrix(
                mean_matrix=mean_matrix,
                std_matrix=std_matrix,
                labels=target_labels,
                title=f"{model_name} {task_name} confusion matrix",
                out_path=out_path,
            )
            created_files.append(out_path)

    return created_files


def main() -> None:
    created_files = generate_confusion_matrices()
    print("Generated LC25000 aggregate confusion matrices:")
    for path in created_files:
        print(path.as_posix())


if __name__ == "__main__":
    main()
