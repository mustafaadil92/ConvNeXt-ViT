from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def _norm_text(x: object) -> str:
    return str(x).strip().lower()


def _norm_key_from_pathlike(x: object) -> str:
    s = str(x).strip().replace("\\", "/")
    return Path(s).name.lower()


def _to_image_path(value: object, images_root: Path | None) -> str:
    s = str(value).strip()
    p = Path(s)
    if p.is_absolute():
        return str(p)
    if images_root is None:
        return s
    return str((images_root / p).resolve())


def _sanitize_fold_name(x: object) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(x).strip())


def _is_center_crop(value: object) -> bool:
    s = _norm_text(value)
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    return "center" in s


def _is_no_flip(value: object) -> bool:
    s = _norm_text(value)
    return s in {"0", "false", "f", "no", "none", "", "nan"}


def _is_zero_rotation(value: object) -> bool:
    s = _norm_text(value)
    if s in {"0", "0.0", "none", "no", "false", "f", "", "nan"}:
        return True
    try:
        return abs(float(s)) < 1e-9
    except Exception:
        return False


def _split_test_group_to_val_test(
    eval_pool: pd.DataFrame,
    origin_col: str,
    class_col: str,
    val_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split filtered test-group data into val/test at origin level with class stratification.
    """
    rng = np.random.default_rng(seed)

    origin_class = (
        eval_pool[[origin_col, class_col]]
        .drop_duplicates(subset=[origin_col])
        .reset_index(drop=True)
    )

    val_origins: set[str] = set()
    for _, grp in origin_class.groupby(class_col):
        origins = grp[origin_col].tolist()
        n = len(origins)
        if n <= 1:
            continue
        n_val = int(round(n * val_ratio))
        n_val = max(1, min(n - 1, n_val))
        picked = rng.choice(origins, size=n_val, replace=False).tolist()
        val_origins.update(picked)

    val_df = eval_pool[eval_pool[origin_col].isin(val_origins)].copy()
    test_df = eval_pool[~eval_pool[origin_col].isin(val_origins)].copy()
    return val_df, test_df


def build_fold_csvs(args: argparse.Namespace) -> None:
    aug_df = pd.read_csv(args.aug_csv).copy()
    folds_df = pd.read_csv(args.folds_csv).copy()

    required_aug = [
        args.aug_filename_col,
        args.aug_class_col,
        args.aug_origin_col,
        args.aug_crop_col,
        args.aug_flip_col,
        args.aug_rotation_col,
        args.aug_mag_col,
    ]
    required_folds = [
        args.folds_fold_col,
        args.folds_mag_col,
        args.folds_group_col,
        args.folds_filename_col,
    ]
    missing_aug = [c for c in required_aug if c not in aug_df.columns]
    missing_folds = [c for c in required_folds if c not in folds_df.columns]
    if missing_aug:
        raise ValueError(f"Missing columns in aug CSV: {missing_aug}")
    if missing_folds:
        raise ValueError(f"Missing columns in folds CSV: {missing_folds}")

    aug_df["_origin_key_"] = aug_df[args.aug_origin_col].map(_norm_key_from_pathlike)
    aug_df["_mag_key_"] = aug_df[args.aug_mag_col].map(_norm_text)
    folds_df["_file_key_"] = folds_df[args.folds_filename_col].map(_norm_key_from_pathlike)
    folds_df["_mag_key_"] = folds_df[args.folds_mag_col].map(_norm_text)
    folds_df["_grp_norm_"] = folds_df[args.folds_group_col].map(_norm_text)

    class_names = sorted(aug_df[args.aug_class_col].astype(str).str.strip().unique().tolist())
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    aug_df["_class_name_"] = aug_df[args.aug_class_col].astype(str).str.strip()
    aug_df["label"] = aug_df["_class_name_"].map(class_to_idx).astype(int)

    images_root = Path(args.images_root).resolve() if args.images_root else None
    aug_df["image_path"] = aug_df[args.aug_filename_col].map(
        lambda x: _to_image_path(x, images_root)
    )

    # Filter condition for val/test: center crop + no flip + zero rotation.
    aug_df["_eval_ok_"] = (
        aug_df[args.aug_crop_col].map(_is_center_crop)
        & aug_df[args.aug_flip_col].map(_is_no_flip)
        & aug_df[args.aug_rotation_col].map(_is_zero_rotation)
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    folds = sorted(folds_df[args.folds_fold_col].dropna().unique().tolist(), key=str)
    summary_rows: list[dict[str, object]] = []

    for fold_value in folds:
        fold_meta = folds_df[folds_df[args.folds_fold_col] == fold_value].copy()
        train_meta = fold_meta[fold_meta["_grp_norm_"] == _norm_text(args.train_group_value)].copy()
        eval_meta = fold_meta[fold_meta["_grp_norm_"] == _norm_text(args.test_group_value)].copy()

        # Build pair keys (origin filename + magnification) for exact membership.
        train_pairs = set(zip(train_meta["_file_key_"], train_meta["_mag_key_"]))
        eval_pairs = set(zip(eval_meta["_file_key_"], eval_meta["_mag_key_"]))

        # Training: all augmentations matching originals in train group.
        in_train_group = aug_df.apply(
            lambda r: (r["_origin_key_"], r["_mag_key_"]) in train_pairs, axis=1
        )
        train_pool = aug_df[in_train_group].copy()

        # Eval pool from original test group, then enforce strict no-aug condition.
        in_eval_group = aug_df.apply(
            lambda r: (r["_origin_key_"], r["_mag_key_"]) in eval_pairs, axis=1
        )
        eval_pool = aug_df[in_eval_group & aug_df["_eval_ok_"]].copy()

        # Split test-group-derived pool into val/test.
        val_df, test_df = _split_test_group_to_val_test(
            eval_pool=eval_pool,
            origin_col="_origin_key_",
            class_col="_class_name_",
            val_ratio=args.val_from_test_ratio,
            seed=args.seed,
        )

        cols = ["image_path", "label"]
        train_out = train_pool[cols].reset_index(drop=True)
        val_out = val_df[cols].reset_index(drop=True)
        test_out = test_df[cols].reset_index(drop=True)

        fold_name = _sanitize_fold_name(fold_value)
        train_path = out_dir / f"{args.prefix}_{fold_name}_train.csv"
        val_path = out_dir / f"{args.prefix}_{fold_name}_val.csv"
        test_path = out_dir / f"{args.prefix}_{fold_name}_test.csv"

        train_out.to_csv(train_path, index=False)
        val_out.to_csv(val_path, index=False)
        test_out.to_csv(test_path, index=False)

        summary_rows.append(
            {
                "fold": str(fold_value),
                "train_rows": int(len(train_out)),
                "val_rows": int(len(val_out)),
                "test_rows": int(len(test_out)),
                "train_csv": str(train_path),
                "val_csv": str(val_path),
                "test_csv": str(test_path),
            }
        )

    class_map_path = out_dir / f"{args.prefix}_class_map.csv"
    pd.DataFrame([{"class_name": k, "label": v} for k, v in class_to_idx.items()]).to_csv(
        class_map_path, index=False
    )

    summary_path = out_dir / f"{args.prefix}_fold_summary.csv"
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_path, index=False)

    print("Done. Created per-fold CSVs with columns: image_path,label")
    print(f"Class map: {class_map_path}")
    print(f"Summary: {summary_path}")
    print(summary_df.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create 5-fold train/val/test CSVs aligned to folds.csv grouping."
    )
    parser.add_argument("--aug-csv", required=True, help="Path to augmented manifest CSV")
    parser.add_argument("--folds-csv", required=True, help="Path to folds CSV")
    parser.add_argument(
        "--images-root",
        default="",
        help="Optional root folder to prepend when aug filename values are relative",
    )
    parser.add_argument("--out-dir", default="data/folds", help="Output directory")
    parser.add_argument("--prefix", default="breakhis", help="Output file prefix")
    parser.add_argument(
        "--val-from-test-ratio",
        type=float,
        default=0.5,
        help="Fraction of test-group origins routed to val (rest stay in test)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # aug columns
    parser.add_argument("--aug-filename-col", default="filename")
    parser.add_argument("--aug-crop-col", default="crop")
    parser.add_argument("--aug-flip-col", default="flip")
    parser.add_argument("--aug-rotation-col", default="rotation")
    parser.add_argument("--aug-origin-col", default="origin")
    parser.add_argument("--aug-class-col", default="disease_name")
    parser.add_argument("--aug-mag-col", default="mag")

    # folds columns
    parser.add_argument("--folds-fold-col", default="fold")
    parser.add_argument("--folds-mag-col", default="mag")
    parser.add_argument("--folds-group-col", default="grp")
    parser.add_argument("--folds-filename-col", default="filename")
    parser.add_argument("--train-group-value", default="train")
    parser.add_argument("--test-group-value", default="test")
    return parser.parse_args()


if __name__ == "__main__":
    build_fold_csvs(parse_args())
