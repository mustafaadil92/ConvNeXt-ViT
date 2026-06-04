from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from pprint import pprint

import torch
from PIL import Image, ImageDraw, ImageFont

from src.utils import (
    get_device,
    describe_device,
    seed_everything,
    load_yaml_config,
)
from src.datasets import build_dataloaders
from src.models import build_model
from src.train import build_loss_fn


def parse_args():
    parser = argparse.ArgumentParser(description="ConvNeXt-ViT PyTorch evaluation entry")
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    parser.add_argument("--device", type=str, default=None, choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to checkpoint (.pt/.pth)")
    parser.add_argument(
        "--smoke-eval",
        action="store_true",
        help="Deprecated. Evaluation now runs without this flag.",
    )
    return parser.parse_args()


@torch.no_grad()
def evaluate_one_split(
    model,
    dataloader,
    loss_fn,
    device,
) -> tuple[dict[str, float], torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    labels_all: list[torch.Tensor] = []
    preds_all: list[torch.Tensor] = []
    probs_all: list[torch.Tensor] = []

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = loss_fn(logits, labels)
        preds = logits.argmax(dim=1)
        probs = torch.softmax(logits, dim=1)

        batch_size = labels.size(0)
        total_loss += float(loss.detach().item()) * int(batch_size)
        total_correct += int((preds == labels).sum().item())
        total_samples += int(batch_size)

        labels_all.append(labels.detach().cpu())
        preds_all.append(preds.detach().cpu())
        probs_all.append(probs.detach().cpu())

    if total_samples == 0:
        raise RuntimeError("Evaluation dataloader is empty.")

    y_true = torch.cat(labels_all, dim=0).to(torch.int64)
    y_pred = torch.cat(preds_all, dim=0).to(torch.int64)
    y_prob = torch.cat(probs_all, dim=0).to(torch.float32)
    metrics = {
        "eval_loss": float(total_loss / total_samples),
        "eval_acc": float(total_correct / total_samples),
        "num_samples": int(total_samples),
    }
    return metrics, y_true, y_pred, y_prob


def build_confusion_matrix(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int) -> torch.Tensor:
    if num_classes <= 0:
        raise ValueError("num_classes must be positive.")

    valid = (y_true >= 0) & (y_true < num_classes) & (y_pred >= 0) & (y_pred < num_classes)
    indices = y_true[valid] * num_classes + y_pred[valid]
    counts = torch.bincount(indices, minlength=num_classes * num_classes)
    return counts.reshape(num_classes, num_classes)


def save_metrics_files(metrics: dict, out_json: Path, out_csv: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)


def save_confusion_matrix_csv(cm: torch.Tensor, out_path: Path, class_names: list[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = cm.cpu().tolist()
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true/pred", *class_names])
        for i, row in enumerate(rows):
            writer.writerow([class_names[i], *row])


def save_confusion_matrix_image(cm: torch.Tensor, out_path: Path, class_names: list[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cm_np = cm.cpu().numpy()
    n = len(class_names)
    cell = 56
    left = 140
    top = 100
    right_pad = 20
    bottom_pad = 20
    width = left + n * cell + right_pad
    height = top + n * cell + bottom_pad

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    max_val = int(cm_np.max()) if cm_np.size > 0 else 0
    scale = max(1, max_val)

    for r in range(n):
        for c in range(n):
            v = int(cm_np[r, c])
            intensity = int(255 - (v / scale) * 200)
            color = (255, intensity, intensity)
            x0 = left + c * cell
            y0 = top + r * cell
            x1 = x0 + cell
            y1 = y0 + cell
            draw.rectangle([x0, y0, x1, y1], fill=color, outline=(180, 180, 180))
            text = str(v)
            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            draw.text((x0 + (cell - tw) / 2, y0 + (cell - th) / 2), text, fill="black", font=font)

    draw.text((left, 16), "Confusion Matrix (rows=true, cols=pred)", fill="black", font=font)

    for i, name in enumerate(class_names):
        bbox_x = draw.textbbox((0, 0), name, font=font)
        tw = bbox_x[2] - bbox_x[0]
        draw.text((left + i * cell + (cell - tw) / 2, top - 20), name, fill="black", font=font)
        draw.text((8, top + i * cell + 18), name, fill="black", font=font)

    image.save(out_path)


def _safe_div(n: float, d: float) -> float:
    return float(n / d) if d != 0 else 0.0


def _multiclass_mcc_from_cm(cm: torch.Tensor) -> float:
    cmf = cm.to(torch.float64)
    s = float(cmf.sum().item())
    if s <= 0:
        return 0.0
    c = float(torch.diag(cmf).sum().item())
    p = cmf.sum(dim=0)
    t = cmf.sum(dim=1)
    num = c * s - float((p * t).sum().item())
    den = math.sqrt(
        max(0.0, s * s - float((p * p).sum().item()))
        * max(0.0, s * s - float((t * t).sum().item()))
    )
    return _safe_div(num, den)


def _kappa_from_cm(cm: torch.Tensor) -> float:
    cmf = cm.to(torch.float64)
    s = float(cmf.sum().item())
    if s <= 0:
        return 0.0
    po = _safe_div(float(torch.diag(cmf).sum().item()), s)
    row = cmf.sum(dim=1)
    col = cmf.sum(dim=0)
    pe = _safe_div(float((row * col).sum().item()), s * s)
    return _safe_div(po - pe, 1.0 - pe)


def _roc_binary(y_true_bin: torch.Tensor, scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float] | None:
    y = y_true_bin.to(torch.int64)
    s = scores.to(torch.float64)
    p = int(y.sum().item())
    n = int(y.numel() - p)
    if p == 0 or n == 0:
        return None

    order = torch.argsort(s, descending=True)
    y_sorted = y[order]
    tp = torch.cumsum(y_sorted, dim=0).to(torch.float64)
    fp = torch.cumsum((1 - y_sorted), dim=0).to(torch.float64)

    tpr = torch.cat([torch.tensor([0.0], dtype=torch.float64), tp / p, torch.tensor([1.0], dtype=torch.float64)])
    fpr = torch.cat([torch.tensor([0.0], dtype=torch.float64), fp / n, torch.tensor([1.0], dtype=torch.float64)])
    auc = float(torch.trapz(tpr, fpr).item())
    return fpr, tpr, auc


def compute_classification_metrics(
    cm: torch.Tensor,
    y_true: torch.Tensor,
    y_prob: torch.Tensor,
    class_names: list[str],
) -> tuple[dict[str, float], list[dict[str, float]], list[dict[str, object]]]:
    cmf = cm.to(torch.float64)
    total = float(cmf.sum().item())
    tp = torch.diag(cmf)
    fp = cmf.sum(dim=0) - tp
    fn = cmf.sum(dim=1) - tp
    tn = total - (tp + fp + fn)

    precision_per = torch.where(tp + fp > 0, tp / (tp + fp), torch.zeros_like(tp))
    recall_per = torch.where(tp + fn > 0, tp / (tp + fn), torch.zeros_like(tp))
    f1_per = torch.where(
        precision_per + recall_per > 0,
        2 * precision_per * recall_per / (precision_per + recall_per),
        torch.zeros_like(tp),
    )
    tnr_per = torch.where(tn + fp > 0, tn / (tn + fp), torch.zeros_like(tp))

    accuracy = _safe_div(float(tp.sum().item()), total) if total > 0 else 0.0
    precision_macro = float(precision_per.mean().item()) if precision_per.numel() else 0.0
    recall_macro = float(recall_per.mean().item()) if recall_per.numel() else 0.0
    f1_macro = float(f1_per.mean().item()) if f1_per.numel() else 0.0
    tnr_macro = float(tnr_per.mean().item()) if tnr_per.numel() else 0.0
    j_stat = recall_macro + tnr_macro - 1.0
    mcc = _multiclass_mcc_from_cm(cm)
    kappa = _kappa_from_cm(cm)

    per_class_rows: list[dict[str, float]] = []
    roc_items: list[dict[str, object]] = []
    auc_values: list[float] = []
    for i, name in enumerate(class_names):
        p_val = float(precision_per[i].item())
        r_val = float(recall_per[i].item())
        f1_val = float(f1_per[i].item())
        tnr_val = float(tnr_per[i].item())
        row = {
            "class_id": float(i),
            "precision": p_val,
            "recall": r_val,
            "f1_score": f1_val,
            "tnr": tnr_val,
            "support": float(cmf.sum(dim=1)[i].item()),
        }

        roc = _roc_binary((y_true == i).to(torch.int64), y_prob[:, i])
        if roc is not None:
            fpr, tpr, auc_i = roc
            row["auc"] = float(auc_i)
            auc_values.append(float(auc_i))
            roc_items.append(
                {
                    "class_id": i,
                    "label_name": name,
                    "auc": float(auc_i),
                    "fpr": [float(x) for x in fpr.tolist()],
                    "tpr": [float(x) for x in tpr.tolist()],
                }
            )
        else:
            row["auc"] = 0.0
        per_class_rows.append(row)

    auc_macro = float(sum(auc_values) / len(auc_values)) if auc_values else 0.0

    summary = {
        "Accuracy": accuracy,
        "Precision": precision_macro,
        "Recall": recall_macro,
        "F1 Score": f1_macro,
        "J": j_stat,
        "MCC": mcc,
        "\u03ba": kappa,
        "kappa": kappa,
        "TNR": tnr_macro,
        "AUC": auc_macro,
    }
    return summary, per_class_rows, roc_items


def _safe_int(value) -> int | None:
    try:
        return int(float(str(value).strip()))
    except Exception:
        return None


def _extract_subtype_from_image_path(image_path: str) -> str:
    path_norm = str(image_path).replace("\\", "/")
    parts = [p for p in path_norm.split("/") if p]
    for i, part in enumerate(parts):
        if part == "SOB" and i + 1 < len(parts):
            return parts[i + 1].strip()
    return ""


def infer_label_names_from_config_and_csv(config: dict, num_classes: int) -> list[str]:
    data_cfg = config.get("data", {}) or {}

    configured_names = data_cfg.get("class_names", None)
    if isinstance(configured_names, list) and len(configured_names) > 0:
        names = [str(x) for x in configured_names]
    else:
        label_col = str(data_cfg.get("label_col", "label"))
        candidate_csvs = [
            data_cfg.get("train_csv", ""),
            data_cfg.get("val_csv", ""),
            data_cfg.get("test_csv", ""),
            data_cfg.get("all_csv", ""),
        ]
        labels_raw: set[str] = set()
        image_col = str(data_cfg.get("image_col", "image_path"))
        label_to_name_counts: dict[int, dict[str, int]] = {}
        for csv_path in candidate_csvs:
            csv_text = str(csv_path).strip()
            if not csv_text:
                continue
            p = Path(csv_text)
            if not p.exists() or not p.is_file():
                continue
            with p.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames or label_col not in reader.fieldnames:
                    continue
                for row in reader:
                    raw_label = str(row.get(label_col, "")).strip()
                    labels_raw.add(raw_label)

                    label_int = _safe_int(raw_label)
                    if label_int is None:
                        continue

                    subtype = _extract_subtype_from_image_path(str(row.get(image_col, "")).strip())
                    if not subtype:
                        continue

                    bucket = label_to_name_counts.setdefault(label_int, {})
                    bucket[subtype] = int(bucket.get(subtype, 0)) + 1

        int_labels = [_safe_int(v) for v in labels_raw if v != ""]
        if int_labels and all(v is not None for v in int_labels):
            unique_ints = sorted(set(v for v in int_labels if v is not None))
            max_id = max(unique_ints) if unique_ints else -1
            target_size = num_classes if num_classes > 0 else (max_id + 1)
            names_by_id = [str(i) for i in range(max(0, target_size))]
            for class_id in unique_ints:
                if class_id < 0 or class_id >= len(names_by_id):
                    continue
                name_counts = label_to_name_counts.get(class_id, {})
                if name_counts:
                    best_name = sorted(name_counts.items(), key=lambda x: x[1], reverse=True)[0][0]
                    names_by_id[class_id] = best_name
            names = names_by_id
        else:
            names = sorted(v for v in labels_raw if v != "")

    if num_classes > 0:
        if len(names) < num_classes:
            names.extend(str(i) for i in range(len(names), num_classes))
        elif len(names) > num_classes:
            names = names[:num_classes]

    return names


def save_label_names_files(class_names: list[str], out_json: Path, out_csv: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    payload = [{"class_id": i, "label_name": name} for i, name in enumerate(class_names)]
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["class_id", "label_name"])
        writer.writeheader()
        writer.writerows(payload)


def _save_dict_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with out_path.open("w", newline="", encoding="utf-8") as f:
            f.write("")
        return
    keys = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _map_point(x: float, y: float, left: int, top: int, width: int, height: int) -> tuple[int, int]:
    px = left + int(max(0.0, min(1.0, x)) * width)
    py = top + height - int(max(0.0, min(1.0, y)) * height)
    return px, py


def save_roc_image(roc_items: list[dict[str, object]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    w, h = 1100, 850
    left, top = 90, 60
    pw, ph = 700, 700
    image = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    palette = [
        (220, 20, 60), (30, 144, 255), (46, 139, 87), (255, 140, 0),
        (138, 43, 226), (70, 130, 180), (160, 82, 45), (199, 21, 133),
        (0, 139, 139), (178, 34, 34),
    ]

    draw.rectangle([left, top, left + pw, top + ph], outline=(50, 50, 50), width=2)
    draw.line([left, top + ph, left + pw, top], fill=(180, 180, 180), width=1)
    draw.text((left, top - 24), "ROC (one-vs-rest)", fill="black", font=font)
    draw.text((left + pw // 2 - 20, top + ph + 20), "FPR", fill="black", font=font)
    draw.text((20, top + ph // 2), "TPR", fill="black", font=font)

    for i in range(6):
        val = i / 5
        x, y = _map_point(val, 0.0, left, top, pw, ph)
        draw.line([x, top, x, top + ph], fill=(235, 235, 235), width=1)
        draw.text((x - 8, top + ph + 4), f"{val:.1f}", fill="black", font=font)
        x0, y0 = _map_point(0.0, val, left, top, pw, ph)
        draw.line([left, y0, left + pw, y0], fill=(235, 235, 235), width=1)
        draw.text((left - 30, y0 - 6), f"{val:.1f}", fill="black", font=font)

    legend_x = left + pw + 20
    legend_y = top
    for idx, item in enumerate(roc_items):
        color = palette[idx % len(palette)]
        fpr = item["fpr"]
        tpr = item["tpr"]
        points: list[tuple[int, int]] = []
        for x, y in zip(fpr, tpr):
            points.append(_map_point(float(x), float(y), left, top, pw, ph))
        if len(points) >= 2:
            draw.line(points, fill=color, width=2)

        label = f"{item['class_id']}:{item['label_name']} AUC={float(item['auc']):.4f}"
        draw.rectangle([legend_x, legend_y + idx * 18 + 4, legend_x + 12, legend_y + idx * 18 + 14], fill=color)
        draw.text((legend_x + 18, legend_y + idx * 18), label, fill="black", font=font)

    image.save(out_path)


def _read_history_csv(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed: dict[str, float] = {}
            for k, v in row.items():
                try:
                    parsed[k] = float(v) if v not in (None, "") else float("nan")
                except Exception:
                    parsed[k] = float("nan")
            rows.append(parsed)
    return rows


def _find_history_file(config: dict, exp_name: str) -> Path | None:
    output_cfg = config.get("output", {}) or {}
    log_dir = Path(str(output_cfg.get("log_dir", "outputs/logs")))
    candidates = [
        log_dir / f"{exp_name}_history.csv",
        log_dir / f"{exp_name}_smoke_history.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _plot_series_panel(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    title: str,
    xvals: list[float],
    series: list[tuple[str, list[float], tuple[int, int, int]]],
    font,
) -> None:
    x0, y0, x1, y1 = rect
    w = x1 - x0
    h = y1 - y0
    draw.rectangle([x0, y0, x1, y1], outline=(60, 60, 60), width=1)
    draw.text((x0 + 6, y0 - 18), title, fill="black", font=font)

    valid_vals: list[float] = []
    for _, vals, _ in series:
        for v in vals:
            if not math.isnan(v):
                valid_vals.append(v)
    if not valid_vals or not xvals:
        draw.text((x0 + 8, y0 + 8), "No data", fill="black", font=font)
        return

    ymin = min(valid_vals)
    ymax = max(valid_vals)
    if abs(ymax - ymin) < 1e-12:
        ymax = ymin + 1.0

    xmin = min(xvals)
    xmax = max(xvals)
    if abs(xmax - xmin) < 1e-12:
        xmax = xmin + 1.0

    for i in range(5):
        frac = i / 4
        y = y0 + h - int(frac * h)
        draw.line([x0, y, x1, y], fill=(235, 235, 235), width=1)

    for i, (name, vals, color) in enumerate(series):
        points: list[tuple[int, int]] = []
        for xv, yv in zip(xvals, vals):
            if math.isnan(yv):
                continue
            px = x0 + int((xv - xmin) / (xmax - xmin) * w)
            py = y0 + h - int((yv - ymin) / (ymax - ymin) * h)
            points.append((px, py))
        if len(points) >= 2:
            draw.line(points, fill=color, width=2)
        draw.rectangle([x0 + 8, y0 + 8 + i * 16, x0 + 18, y0 + 18 + i * 16], fill=color)
        draw.text((x0 + 24, y0 + 6 + i * 16), name, fill="black", font=font)


def save_training_curves_image(config: dict, exp_name: str, out_path: Path) -> Path | None:
    history_path = _find_history_file(config, exp_name)
    if history_path is None:
        return None
    rows = _read_history_csv(history_path)
    if not rows:
        return None

    epochs = [r.get("epoch", float(i + 1)) for i, r in enumerate(rows)]
    train_loss = [r.get("train_loss", float("nan")) for r in rows]
    val_loss = [r.get("val_loss", float("nan")) for r in rows]
    train_acc = [r.get("train_acc", float("nan")) for r in rows]
    val_acc = [r.get("val_acc", float("nan")) for r in rows]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (1200, 700), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((20, 10), f"Training/Validation Curves - {exp_name}", fill="black", font=font)

    _plot_series_panel(
        draw,
        rect=(60, 60, 580, 650),
        title="Loss",
        xvals=epochs,
        series=[
            ("train_loss", train_loss, (220, 20, 60)),
            ("val_loss", val_loss, (30, 144, 255)),
        ],
        font=font,
    )
    _plot_series_panel(
        draw,
        rect=(640, 60, 1160, 650),
        title="Accuracy",
        xvals=epochs,
        series=[
            ("train_acc", train_acc, (34, 139, 34)),
            ("val_acc", val_acc, (255, 140, 0)),
        ],
        font=font,
    )

    image.save(out_path)
    return history_path


def main():
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

    print("Evaluation entry is working.")
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
            "checkpoint": args.checkpoint,
        }
    )

    print("\nDevice info:")
    pprint(info)

    print("\nBuilding dataloaders...")
    loaders = build_dataloaders(config)
    split_name = "test" if "test" in loaders else "val"
    if split_name not in loaders:
        raise RuntimeError("No test/val dataloader was built. Need at least one evaluation split.")
    if split_name == "val":
        print("[Eval] Test split not found. Falling back to val split.")

    print("Loader splits:", list(loaders.keys()))
    print(f"Evaluation split: {split_name}")

    print("\nBuilding model + loss...")
    model = build_model(config).to(device)
    loss_fn = build_loss_fn(config)

    if args.checkpoint:
        print(f"[Eval] Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print("[Eval] Checkpoint loaded.")
    else:
        print("[Eval] No checkpoint provided; evaluating random-initialized model.")

    metrics, y_true, y_pred, y_prob = evaluate_one_split(
        model=model,
        dataloader=loaders[split_name],
        loss_fn=loss_fn,
        device=device,
    )

    num_classes_cfg = int(config.get("data", {}).get("num_classes", 0))
    inferred_classes = int(max(y_true.max().item(), y_pred.max().item()) + 1)
    num_classes = num_classes_cfg if num_classes_cfg > 0 else inferred_classes
    cm = build_confusion_matrix(y_true=y_true, y_pred=y_pred, num_classes=num_classes)
    class_names = infer_label_names_from_config_and_csv(config=config, num_classes=num_classes)
    summary_metrics, per_class_metrics, roc_items = compute_classification_metrics(
        cm=cm,
        y_true=y_true,
        y_prob=y_prob,
        class_names=class_names,
    )
    metrics = {**metrics, **summary_metrics}

    exp_name = str(config.get("experiment_name", "experiment"))
    output_cfg = config.get("output", {}) or {}
    pred_dir = Path(str(output_cfg.get("predictions_dir", "outputs/predictions")))
    fig_dir = Path(str(output_cfg.get("figure_dir", "outputs/figures")))

    metrics_json_path = pred_dir / f"{exp_name}_{split_name}_metrics.json"
    metrics_csv_path = pred_dir / f"{exp_name}_{split_name}_metrics.csv"
    cm_csv_path = pred_dir / f"{exp_name}_{split_name}_confusion_matrix.csv"
    cm_img_path = fig_dir / f"{exp_name}_{split_name}_confusion_matrix.png"
    labels_json_path = pred_dir / f"{exp_name}_{split_name}_label_names.json"
    labels_csv_path = pred_dir / f"{exp_name}_{split_name}_label_names.csv"
    per_class_csv_path = pred_dir / f"{exp_name}_{split_name}_per_class_metrics.csv"
    per_class_json_path = pred_dir / f"{exp_name}_{split_name}_per_class_metrics.json"
    roc_json_path = pred_dir / f"{exp_name}_{split_name}_roc_curves.json"
    roc_img_path = fig_dir / f"{exp_name}_{split_name}_roc.png"
    curves_img_path = fig_dir / f"{exp_name}_training_validation_curves.png"

    save_metrics_files(metrics=metrics, out_json=metrics_json_path, out_csv=metrics_csv_path)
    save_confusion_matrix_csv(cm=cm, out_path=cm_csv_path, class_names=class_names)
    save_confusion_matrix_image(cm=cm, out_path=cm_img_path, class_names=class_names)
    save_label_names_files(class_names=class_names, out_json=labels_json_path, out_csv=labels_csv_path)
    _save_dict_csv(rows=per_class_metrics, out_path=per_class_csv_path)
    with per_class_json_path.open("w", encoding="utf-8") as f:
        json.dump(per_class_metrics, f, indent=2)
    with roc_json_path.open("w", encoding="utf-8") as f:
        json.dump(roc_items, f, indent=2)
    save_roc_image(roc_items=roc_items, out_path=roc_img_path)
    history_source = save_training_curves_image(config=config, exp_name=exp_name, out_path=curves_img_path)

    print("\n[Eval] Metrics:")
    pprint(metrics)
    print(f"[Eval] Saved metrics JSON: {metrics_json_path}")
    print(f"[Eval] Saved metrics CSV : {metrics_csv_path}")
    print(f"[Eval] Saved CM CSV      : {cm_csv_path}")
    print(f"[Eval] Saved CM image    : {cm_img_path}")
    print(f"[Eval] Saved labels JSON : {labels_json_path}")
    print(f"[Eval] Saved labels CSV  : {labels_csv_path}")
    print(f"[Eval] Saved per-class CSV: {per_class_csv_path}")
    print(f"[Eval] Saved per-class JSON: {per_class_json_path}")
    print(f"[Eval] Saved ROC JSON    : {roc_json_path}")
    print(f"[Eval] Saved ROC image   : {roc_img_path}")
    if history_source is not None:
        print(f"[Eval] Saved train/val curves: {curves_img_path} (from {history_source})")
    else:
        print("[Eval] Training/validation history CSV not found; skipped curve image.")


if __name__ == "__main__":
    main()
