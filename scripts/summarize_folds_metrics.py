import argparse
import csv
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize fold metrics CSVs into mean/std and mean +- std."
    )
    parser.add_argument(
        "--pred-dir",
        default="outputs/predictions",
        help="Directory containing per-fold metrics CSV files.",
    )
    parser.add_argument(
        "--exp-prefix",
        default="breakhis_fold",
        help="Experiment prefix before fold index. Example: breakhis_fold",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Evaluated split name used in metrics filenames (e.g., test or val).",
    )
    parser.add_argument(
        "--start-fold",
        type=int,
        default=1,
        help="First fold index to include.",
    )
    parser.add_argument(
        "--end-fold",
        type=int,
        default=5,
        help="Last fold index to include.",
    )
    parser.add_argument(
        "--out-csv",
        default="outputs/predictions/breakhis_5fold_metrics_summary.csv",
        help="Output CSV path.",
    )
    return parser.parse_args()


def _read_single_row_csv(path: Path) -> dict[str, str]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if len(rows) != 1:
        raise ValueError(f"Expected exactly 1 data row in {path}, found {len(rows)}.")
    return rows[0]


def _to_float(text: str) -> float:
    return float(str(text).strip())


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _sample_std(values: list[float], mean_value: float) -> float:
    if len(values) <= 1:
        return 0.0
    var = sum((v - mean_value) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(var)


def main() -> None:
    args = parse_args()

    if args.start_fold > args.end_fold:
        raise ValueError("--start-fold must be <= --end-fold")

    pred_dir = Path(args.pred_dir)
    out_csv = Path(args.out_csv)
    fold_ids = list(range(args.start_fold, args.end_fold + 1))

    per_fold_rows: list[dict[str, str]] = []
    for fold in fold_ids:
        metrics_path = pred_dir / f"{args.exp_prefix}{fold}_{args.split}_metrics.csv"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing metrics CSV: {metrics_path}")
        per_fold_rows.append(_read_single_row_csv(metrics_path))

    metric_names = list(per_fold_rows[0].keys())
    summary_rows: list[dict[str, str]] = []

    for metric in metric_names:
        values = [_to_float(row[metric]) for row in per_fold_rows]
        m = _mean(values)
        s = _sample_std(values, m)
        row: dict[str, str] = {
            "metric": metric,
            "mean": f"{m:.6f}",
            "std": f"{s:.6f}",
            "mean+-std": f"{m:.6f} +- {s:.6f}",
        }
        for idx, value in enumerate(values, start=args.start_fold):
            row[f"fold_{idx}"] = f"{value:.6f}"
        summary_rows.append(row)

    fieldnames = (
        ["metric", "mean", "std", "mean+-std"]
        + [f"fold_{i}" for i in range(args.start_fold, args.end_fold + 1)]
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"[Summary] Wrote: {out_csv}")


if __name__ == "__main__":
    main()
