from __future__ import annotations

import csv
import math
from pathlib import Path


PREDICTIONS_ROOT = Path("outputs/predictions/LC25000")
COMPARISON_ROOT = PREDICTIONS_ROOT / "comparisons"
MODEL_ORDER = ("convnext", "convnext-vit", "vit-s")
TASK_MODEL_DIRS = {
    "multiclass": {
        "convnext": PREDICTIONS_ROOT / "convnext",
        "convnext-vit": PREDICTIONS_ROOT / "convnext-vit",
        "vit-s": PREDICTIONS_ROOT / "vit-s",
    },
}
METRICS_FILENAME_SUFFIX = "_test_metrics.csv"
METRIC_RENAMES = {
    "ÃŽÂº": "kappa",
    "Îº": "kappa",
    "κ": "kappa",
}


def _read_single_row_csv(path: Path) -> dict[str, str]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one row in {path}, found {len(rows)}.")
    return rows[0]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _sample_std(values: list[float], mean_value: float) -> float:
    if len(values) <= 1:
        return 0.0
    variance = sum((value - mean_value) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def _format_mean_std(values: list[float]) -> str:
    mean_value = _mean(values)
    std_value = _sample_std(values, mean_value)
    return f"{mean_value:.6f} ± {std_value:.6f}"


def summarize_model(model_dir: Path) -> dict[str, str]:
    metric_files = sorted(model_dir.glob(f"*{METRICS_FILENAME_SUFFIX}"))
    if not metric_files:
        raise FileNotFoundError(f"No metrics CSV files found in {model_dir}")

    rows = [_read_single_row_csv(path) for path in metric_files]
    metric_names: list[str] = []
    seen_metrics: set[str] = set()
    for raw_name in rows[0].keys():
        metric_name = METRIC_RENAMES.get(raw_name, raw_name)
        if metric_name in seen_metrics:
            continue
        seen_metrics.add(metric_name)
        metric_names.append(metric_name)

    summary: dict[str, str] = {}
    for metric in metric_names:
        source_metric = metric
        for candidate, renamed in METRIC_RENAMES.items():
            if renamed == metric and candidate in rows[0]:
                source_metric = candidate
                break
        if metric in rows[0]:
            source_metric = metric
        values = [float(row[source_metric]) for row in rows]
        summary[metric] = _format_mean_std(values)
    return summary


def build_comparison_rows(task_name: str) -> tuple[list[str], list[dict[str, str]]]:
    model_dirs = TASK_MODEL_DIRS[task_name]
    summaries = {model_name: summarize_model(model_dirs[model_name]) for model_name in MODEL_ORDER}

    metric_names = list(summaries[MODEL_ORDER[0]].keys())
    comparison_rows: list[dict[str, str]] = []
    for metric in metric_names:
        row = {"metric": metric}
        for model_name in MODEL_ORDER:
            row[model_name] = summaries[model_name][metric]
        comparison_rows.append(row)

    fieldnames = ["metric", *MODEL_ORDER]
    return fieldnames, comparison_rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    for task_name in TASK_MODEL_DIRS:
        fieldnames, rows = build_comparison_rows(task_name)
        out_path = COMPARISON_ROOT / f"{task_name}_model_metrics_comparison.csv"
        write_csv(out_path, fieldnames, rows)
        print(f"Wrote {out_path.as_posix()}")


if __name__ == "__main__":
    main()
