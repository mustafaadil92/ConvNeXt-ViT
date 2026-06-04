from __future__ import annotations

from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

from .dataset import ImageClassificationDataset
from .transforms import build_train_transforms, build_eval_transforms


def _normalize_text(x) -> str:
    return str(x).strip()


def _normalize_split_text(x) -> str:
    return str(x).strip().lower()


def _normalize_fold_value(x):
    """
    Try to normalize fold values so both '0' and 0 behave consistently.
    """
    s = str(x).strip()
    try:
        # handles "0", "1", "2" ... and numeric values
        return int(float(s))
    except Exception:
        return s.lower()


def _build_dataset_from_df(
    df: pd.DataFrame,
    image_col: str,
    label_col: str,
    transform,
    class_to_idx: dict[str, int] | None = None,
):
    """
    Requires ImageClassificationDataset to support dataframe=...
    If it doesn't yet, patch dataset.py next (small patch).
    """
    return ImageClassificationDataset(
        dataframe=df.reset_index(drop=True),
        image_col=image_col,
        label_col=label_col,
        transform=transform,
        class_to_idx=class_to_idx,
    )


def _drop_cross_split_origin_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    origin_col: str,
):
    """
    Remove rows from val/test if their origin appears in an earlier-priority split.
    Priority: train > val > test
    """
    if not origin_col:
        return train_df, val_df, test_df, {
            "train_removed": 0,
            "val_removed": 0,
            "test_removed": 0,
        }

    # Normalize origin text for robust matching
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["_origin_norm_"] = train_df[origin_col].map(_normalize_text)
    val_df["_origin_norm_"] = val_df[origin_col].map(_normalize_text)
    test_df["_origin_norm_"] = test_df[origin_col].map(_normalize_text)

    # Optional: remove duplicates *within* each split by origin (keep first)
    # (Comment out if you want to keep multiple augmentations of same origin inside train)
    # train_before = len(train_df)
    # val_before = len(val_df)
    # test_before = len(test_df)
    # train_df = train_df.drop_duplicates(subset=["_origin_norm_"], keep="first")
    # val_df = val_df.drop_duplicates(subset=["_origin_norm_"], keep="first")
    # test_df = test_df.drop_duplicates(subset=["_origin_norm_"], keep="first")

    train_origins = set(train_df["_origin_norm_"].tolist())
    val_origins = set(val_df["_origin_norm_"].tolist())

    val_before = len(val_df)
    test_before = len(test_df)

    # Remove from val if origin exists in train
    val_df = val_df[~val_df["_origin_norm_"].isin(train_origins)].copy()

    # Recompute val origins after filtering
    val_origins = set(val_df["_origin_norm_"].tolist())

    # Remove from test if origin exists in train OR val
    forbidden_test_origins = train_origins.union(val_origins)
    test_df = test_df[~test_df["_origin_norm_"].isin(forbidden_test_origins)].copy()

    val_removed = val_before - len(val_df)
    test_removed = test_before - len(test_df)

    # Drop helper column
    for df in (train_df, val_df, test_df):
        if "_origin_norm_" in df.columns:
            df.drop(columns=["_origin_norm_"], inplace=True)

    return train_df, val_df, test_df, {
        "train_removed": 0,
        "val_removed": int(val_removed),
        "test_removed": int(test_removed),
    }


def _prepare_label_encoder_from_all_df(all_df: pd.DataFrame, label_col: str) -> dict[str, int]:
    """
    Stable label encoding from the full CSV, not per-split.
    This is critical so train/val/test use the same class IDs.
    """
    classes = sorted(all_df[label_col].astype(str).map(str.strip).unique().tolist())
    return {cls_name: idx for idx, cls_name in enumerate(classes)}


def build_datasets(config: dict) -> dict:
    """
    Build dataset objects from config.

    Supported modes:
    A) Separate CSVs (train_csv / val_csv / test_csv)
    B) Single CSV with explicit split column (all_csv + split_col)
    C) Single CSV with fold column (all_csv + fold_col + train_folds/val_folds/test_folds OR val_fold/test_fold)
    """
    data_cfg = config.get("data", {}) or {}

    image_col = str(data_cfg.get("image_col", "image_path"))
    label_col = str(data_cfg.get("label_col", "label"))

    train_csv = data_cfg.get("train_csv", "")
    val_csv = data_cfg.get("val_csv", "")
    test_csv = data_cfg.get("test_csv", "")

    all_csv = data_cfg.get("all_csv", "")
    split_col = str(data_cfg.get("split_col", "split"))

    fold_col = str(data_cfg.get("fold_col", "fold"))
    origin_col = str(data_cfg.get("origin_col", "origin")) if data_cfg.get("origin_col", "origin") else ""

    train_tfms = build_train_transforms(config)
    eval_tfms = build_eval_transforms(config)

    datasets = {}

    # ---------------------------------------------------------
    # Mode B/C: Single CSV (split column OR fold column)
    # ---------------------------------------------------------
    if str(all_csv).strip():
        all_df = pd.read_csv(all_csv)

        required_basic = {image_col, label_col}
        missing_basic = [c for c in required_basic if c not in all_df.columns]
        if missing_basic:
            raise ValueError(
                f"Missing required columns in all_csv ({all_csv}): {missing_basic}. "
                f"Found columns: {list(all_df.columns)}"
            )

        # Stable class mapping from full CSV
        class_to_idx = _prepare_label_encoder_from_all_df(all_df, label_col)

        # -------- Mode C: fold-based split --------
        if fold_col in all_df.columns and (
            data_cfg.get("val_fold", None) is not None
            or data_cfg.get("test_fold", None) is not None
            or data_cfg.get("train_folds", None) is not None
            or data_cfg.get("val_folds", None) is not None
            or data_cfg.get("test_folds", None) is not None
        ):
            all_df = all_df.copy()
            all_df["_fold_norm_"] = all_df[fold_col].map(_normalize_fold_value)

            # Flexible config:
            # Option 1: explicit fold lists
            # Option 2: val_fold + test_fold, train = rest
            train_folds_cfg = data_cfg.get("train_folds", None)
            val_folds_cfg = data_cfg.get("val_folds", None)
            test_folds_cfg = data_cfg.get("test_folds", None)

            val_fold_cfg = data_cfg.get("val_fold", None)
            test_fold_cfg = data_cfg.get("test_fold", None)

            if train_folds_cfg is not None or val_folds_cfg is not None or test_folds_cfg is not None:
                train_folds = set(_normalize_fold_value(x) for x in (train_folds_cfg or []))
                val_folds = set(_normalize_fold_value(x) for x in (val_folds_cfg or []))
                test_folds = set(_normalize_fold_value(x) for x in (test_folds_cfg or []))
            else:
                if val_fold_cfg is None:
                    raise ValueError("Fold mode requires 'val_fold' (or 'val_folds').")
                val_fold = _normalize_fold_value(val_fold_cfg)
                test_fold = _normalize_fold_value(test_fold_cfg) if test_fold_cfg is not None else None

                all_folds_present = set(all_df["_fold_norm_"].unique().tolist())
                val_folds = {val_fold}
                test_folds = {test_fold} if test_fold is not None else set()
                train_folds = all_folds_present - val_folds - test_folds

            train_df = all_df[all_df["_fold_norm_"].isin(train_folds)].copy()
            val_df = all_df[all_df["_fold_norm_"].isin(val_folds)].copy()
            test_df = all_df[all_df["_fold_norm_"].isin(test_folds)].copy()

            if "_fold_norm_" in train_df.columns:
                train_df.drop(columns=["_fold_norm_"], inplace=True)
            if "_fold_norm_" in val_df.columns:
                val_df.drop(columns=["_fold_norm_"], inplace=True)
            if "_fold_norm_" in test_df.columns:
                test_df.drop(columns=["_fold_norm_"], inplace=True)

            # Cross-split origin leakage filtering
            if origin_col and origin_col in all_df.columns:
                train_df, val_df, test_df, leakage_stats = _drop_cross_split_origin_leakage(
                    train_df=train_df,
                    val_df=val_df,
                    test_df=test_df,
                    origin_col=origin_col,
                )
                print("[Data] Origin leakage filtering:", leakage_stats)
            elif origin_col:
                print(f"[Data] origin_col='{origin_col}' not found in CSV. Skipping origin leakage filtering.")

            if len(train_df) > 0:
                datasets["train"] = _build_dataset_from_df(
                    train_df, image_col=image_col, label_col=label_col, transform=train_tfms, class_to_idx=class_to_idx
                )
            if len(val_df) > 0:
                datasets["val"] = _build_dataset_from_df(
                    val_df, image_col=image_col, label_col=label_col, transform=eval_tfms, class_to_idx=class_to_idx
                )
            if len(test_df) > 0:
                datasets["test"] = _build_dataset_from_df(
                    test_df, image_col=image_col, label_col=label_col, transform=eval_tfms, class_to_idx=class_to_idx
                )

            return datasets

        # -------- Mode B: explicit split column --------
        if split_col in all_df.columns:
            split_series = all_df[split_col].map(_normalize_split_text)

            train_aliases = {"train", "training"}
            val_aliases = {"val", "valid", "validation", "dev"}
            test_aliases = {"test", "testing"}

            train_df = all_df[split_series.isin(train_aliases)].copy()
            val_df = all_df[split_series.isin(val_aliases)].copy()
            test_df = all_df[split_series.isin(test_aliases)].copy()

            if origin_col and origin_col in all_df.columns:
                train_df, val_df, test_df, leakage_stats = _drop_cross_split_origin_leakage(
                    train_df=train_df,
                    val_df=val_df,
                    test_df=test_df,
                    origin_col=origin_col,
                )
                print("[Data] Origin leakage filtering:", leakage_stats)

            if len(train_df) > 0:
                datasets["train"] = _build_dataset_from_df(
                    train_df, image_col=image_col, label_col=label_col, transform=train_tfms, class_to_idx=class_to_idx
                )
            if len(val_df) > 0:
                datasets["val"] = _build_dataset_from_df(
                    val_df, image_col=image_col, label_col=label_col, transform=eval_tfms, class_to_idx=class_to_idx
                )
            if len(test_df) > 0:
                datasets["test"] = _build_dataset_from_df(
                    test_df, image_col=image_col, label_col=label_col, transform=eval_tfms, class_to_idx=class_to_idx
                )

            return datasets

        raise ValueError(
            f"all_csv was provided ({all_csv}) but neither split_col='{split_col}' nor usable fold mode was found. "
            f"Available columns: {list(all_df.columns)}"
        )

    # ---------------------------------------------------------
    # Mode A: Separate CSV files (backward compatible)
    # ---------------------------------------------------------
    if str(train_csv).strip():
        datasets["train"] = ImageClassificationDataset(
            csv_path=Path(train_csv),
            image_col=image_col,
            label_col=label_col,
            transform=train_tfms,
        )

    if str(val_csv).strip():
        datasets["val"] = ImageClassificationDataset(
            csv_path=Path(val_csv),
            image_col=image_col,
            label_col=label_col,
            transform=eval_tfms,
        )

    if str(test_csv).strip():
        datasets["test"] = ImageClassificationDataset(
            csv_path=Path(test_csv),
            image_col=image_col,
            label_col=label_col,
            transform=eval_tfms,
        )

    return datasets


def build_dataloaders(config: dict) -> dict:
    """
    Build PyTorch DataLoaders for available splits.
    """
    data_cfg = config.get("data", {}) or {}
    train_cfg = config.get("train", {}) or {}

    batch_size = int(train_cfg.get("batch_size", 16))
    num_workers = int(data_cfg.get("num_workers", 0))
    pin_memory = bool(data_cfg.get("pin_memory", True))
    persistent_workers = bool(data_cfg.get("persistent_workers", num_workers > 0))
    prefetch_factor = int(data_cfg.get("prefetch_factor", 2))
    train_drop_last = bool(train_cfg.get("drop_last", False))

    datasets = build_datasets(config)
    loaders = {}

    common_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        common_kwargs["persistent_workers"] = persistent_workers
        common_kwargs["prefetch_factor"] = prefetch_factor

    if "train" in datasets:
        loaders["train"] = DataLoader(
            datasets["train"],
            shuffle=True,
            drop_last=train_drop_last,
            **common_kwargs,
        )

    if "val" in datasets:
        loaders["val"] = DataLoader(
            datasets["val"],
            shuffle=False,
            drop_last=False,
            **common_kwargs,
        )

    if "test" in datasets:
        loaders["test"] = DataLoader(
            datasets["test"],
            shuffle=False,
            drop_last=False,
            **common_kwargs,
        )

    return loaders
