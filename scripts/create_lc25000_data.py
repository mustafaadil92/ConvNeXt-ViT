from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEFAULT_LABEL_DIRS = [
    Path("colon_image_sets/colon_aca"),
    Path("colon_image_sets/colon_n"),
    Path("lung_image_sets/lung_aca"),
    Path("lung_image_sets/lung_n"),
    Path("lung_image_sets/lung_scc"),
]


MODEL_CONFIGS = {
    "convnext": {
        "project_name": "convnext_torch",
        "experiment_prefix": "lc25000_convnext_fold",
        "batch_size": 16,
        "extra_train": [
            "  memory_cleanup_interval: 1",
            "  empty_cuda_cache_interval: 10",
        ],
        "model": [
            "  name: convnext",
            "  variant: convnext_tiny",
            "  pretrained: false",
            "  dropout: 0.1",
        ],
    },
    "convnext-vit": {
        "project_name": "convnext_vit_torch",
        "experiment_prefix": "lc25000_fold",
        "batch_size": 64,
        "extra_train": [],
        "model": [
            "  name: convnext_vit_hybrid",
            "  patch_size: 16",
            "  dropout: 0.1",
            "  num_layers : 4",
        ],
    },
    "vit-s": {
        "project_name": "vit_s_torch",
        "experiment_prefix": "lc25000_vits_fold",
        "batch_size": 64,
        "extra_train": [],
        "model": [
            "  name: vit-s",
            "  patch_size: 16",
            "  dropout: 0.1",
            "  embed_dim: 384",
            "  num_heads: 6",
            "  depth: 12",
            "  mlp_ratio: 4.0",
        ],
    },
}


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(rows)


def _collect_rows(source_root: Path, seed: int) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    all_rows: list[dict[str, object]] = []
    class_map: list[dict[str, object]] = []
    rng = random.Random(seed)

    for label_idx, rel_label_dir in enumerate(DEFAULT_LABEL_DIRS):
        label_dir = source_root / rel_label_dir
        if not label_dir.is_dir():
            raise FileNotFoundError(f"Missing label directory: {label_dir}")

        class_name = label_dir.name
        image_paths = sorted(
            p for p in label_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not image_paths:
            raise ValueError(f"No images found in label directory: {label_dir}")

        shuffled = image_paths[:]
        rng.shuffle(shuffled)
        for idx, image_path in enumerate(shuffled):
            fold = (idx % 5) + 1
            all_rows.append(
                {
                    "image_path": str(image_path),
                    "label": label_idx,
                    "class_name": class_name,
                    "fold": fold,
                }
            )

        class_map.append({"class_name": class_name, "label": label_idx, "count": len(image_paths)})

    return all_rows, class_map


def _split_rows(all_rows: list[dict[str, object]], fold: int) -> tuple[list[dict[str, object]], ...]:
    test_fold = fold
    val_fold = (fold % 5) + 1
    train_rows = [row for row in all_rows if row["fold"] not in {test_fold, val_fold}]
    val_rows = [row for row in all_rows if row["fold"] == val_fold]
    test_rows = [row for row in all_rows if row["fold"] == test_fold]
    return train_rows, val_rows, test_rows


def _yaml_text(model_name: str, cfg: dict[str, object], fold: int, num_classes: int) -> str:
    exp_name = f"{cfg['experiment_prefix']}{fold}"
    train_csv = f"data/LC25000/lc25000_{fold}_train.csv"
    val_csv = f"data/LC25000/lc25000_{fold}_val.csv"
    test_csv = f"data/LC25000/lc25000_{fold}_test.csv"

    lines = [
        f"project_name: {cfg['project_name']}",
        f"experiment_name: {exp_name}",
        "",
        "seed: 42",
        "device: cuda   # auto | cpu | cuda",
        "deterministic: false",
        "",
        "data:",
        f'  train_csv: "{train_csv}"',
        f'  val_csv: "{val_csv}"',
        f'  test_csv: "{test_csv}"',
        "  image_col: image_path",
        "  label_col: label",
        f"  num_classes: {num_classes}",
        "  input_size: 224",
        "  num_workers: 8",
        "  pin_memory: true",
        "  persistent_workers: false",
        "  prefetch_factor: 4",
        "",
        "train:",
        f"  batch_size: {cfg['batch_size']}",
        "  epochs: 50",
        "  amp: true",
        "  drop_last: true",
        "  gradient_clip_norm: null",
        *cfg["extra_train"],
        "",
        "optimizer:",
        "  name: adamw",
        "  lr: 0.0001",
        "  weight_decay: 0.0001",
        "",
        "scheduler:",
        "  name: cosine",
        "  t_max: 100",
        "  min_lr: 0.000001",
        "",
        "model:",
        *cfg["model"],
        "",
        "output:",
        f"  checkpoint_dir: outputs/checkpoints/LC25000/{model_name}",
        f"  log_dir: outputs/logs/LC25000/{model_name}",
        f"  figure_dir: outputs/figures/LC25000/{model_name}",
        f"  predictions_dir: outputs/predictions/LC25000/{model_name}",
        "  save_best_only: true",
        "  monitor: val_loss",
        "  mode: min",
        "",
    ]
    return "\n".join(lines)


def build_lc25000(source_root: Path, out_dir: Path, seed: int) -> None:
    all_rows, class_map = _collect_rows(source_root=source_root, seed=seed)
    fieldnames = ["image_path", "label", "class_name", "fold"]

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "lc25000_full.csv", all_rows, fieldnames)
    _write_csv(out_dir / "lc25000_class_map.csv", class_map, ["class_name", "label", "count"])

    summary_rows = []
    for fold in range(1, 6):
        train_rows, val_rows, test_rows = _split_rows(all_rows, fold)
        _write_csv(out_dir / f"lc25000_{fold}_train.csv", train_rows, fieldnames)
        _write_csv(out_dir / f"lc25000_{fold}_val.csv", val_rows, fieldnames)
        _write_csv(out_dir / f"lc25000_{fold}_test.csv", test_rows, fieldnames)
        summary_rows.append(
            {
                "fold": fold,
                "train_count": len(train_rows),
                "val_count": len(val_rows),
                "test_count": len(test_rows),
                "train_csv": out_dir / f"lc25000_{fold}_train.csv",
                "val_csv": out_dir / f"lc25000_{fold}_val.csv",
                "test_csv": out_dir / f"lc25000_{fold}_test.csv",
            }
        )

    _write_csv(
        out_dir / "lc25000_fold_summary.csv",
        summary_rows,
        ["fold", "train_count", "val_count", "test_count", "train_csv", "val_csv", "test_csv"],
    )

    configs_dir = out_dir / "configs"
    for model_name, cfg in MODEL_CONFIGS.items():
        model_dir = configs_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        for fold in range(1, 6):
            yaml_path = model_dir / f"lc25000_{model_name.replace('-', '_')}_fold{fold}.yaml"
            yaml_path.write_text(_yaml_text(model_name, cfg, fold, len(class_map)), encoding="utf-8")

    print(f"Created LC25000 metadata in {out_dir}")
    print(f"Images: {len(all_rows)}")
    for row in class_map:
        print(f"{row['class_name']}: {row['count']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create LC25000 CSV folds and model YAML files.")
    parser.add_argument("--source-root", default=r"E:\lung_colon_image_set")
    parser.add_argument("--out-dir", default="data/LC25000")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_lc25000(Path(args.source_root), Path(args.out_dir), args.seed)
