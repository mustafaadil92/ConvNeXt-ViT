from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch
from PIL import Image

from src.explainability import (
    extract_convnext_gradcam,
    extract_hybrid_attention_map,
    extract_hybrid_gradcam,
    extract_vit_attention_map,
    load_config,
    load_model_from_config,
    load_split_dataframe,
    load_tensor_from_pil,
    normalize_map,
    overlay_heatmap,
    render_panel,
    resolve_sample,
    resize_map,
)
from src.utils import describe_device, get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Generate ConvNeXt / ViT / hybrid explanation panels")
    parser.add_argument("--convnext-config", type=str, required=True)
    parser.add_argument("--convnext-checkpoint", type=str, required=True)
    parser.add_argument("--vit-config", type=str, required=True)
    parser.add_argument("--vit-checkpoint", type=str, required=True)
    parser.add_argument("--hybrid-config", type=str, required=True)
    parser.add_argument("--hybrid-checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--image-path", type=str, default="")
    parser.add_argument("--label", type=int, default=None)
    parser.add_argument("--target-class", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--output", type=str, default="outputs/figures/explanations/explanation_panel.png")
    parser.add_argument("--batch-export", action="store_true")
    parser.add_argument("--samples-per-class", type=int, default=None)
    parser.add_argument("--correctness-mode", type=str, default="all", choices=["all", "hybrid"])
    parser.add_argument("--output-dir", type=str, default="outputs/figures/explanations/batch")
    return parser.parse_args()


def _slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_")


def _predict_class(model: torch.nn.Module, image_tensor: torch.Tensor) -> int:
    with torch.no_grad():
        logits = model(image_tensor)
    return int(logits.argmax(dim=1).item())


def _load_models(args, device: torch.device):
    convnext_config, convnext_model = load_model_from_config(
        config_path=args.convnext_config,
        checkpoint_path=args.convnext_checkpoint,
        device=device,
    )
    vit_config, vit_model = load_model_from_config(
        config_path=args.vit_config,
        checkpoint_path=args.vit_checkpoint,
        device=device,
    )
    hybrid_config, hybrid_model = load_model_from_config(
        config_path=args.hybrid_config,
        checkpoint_path=args.hybrid_checkpoint,
        device=device,
    )
    return (
        (convnext_config, convnext_model),
        (vit_config, vit_model),
        (hybrid_config, hybrid_model),
    )


def _build_panel(
    image_path: str | Path,
    sample_label: int | None,
    target_class: int | None,
    alpha: float,
    output_path: str | Path,
    convnext_bundle,
    vit_bundle,
    hybrid_bundle,
    device: torch.device,
) -> Path:
    convnext_config, convnext_model = convnext_bundle
    vit_config, vit_model = vit_bundle
    hybrid_config, hybrid_model = hybrid_bundle

    original_image = Image.open(image_path).convert("RGB")
    convnext_tensor = load_tensor_from_pil(convnext_config, original_image, device)
    vit_tensor = load_tensor_from_pil(vit_config, original_image, device)
    hybrid_tensor = load_tensor_from_pil(hybrid_config, original_image, device)

    panel_hw = (hybrid_tensor.shape[-2], hybrid_tensor.shape[-1])
    panel_image = original_image.resize((panel_hw[1], panel_hw[0]))

    class_for_cam = sample_label if target_class is None else target_class

    convnext_cam, convnext_pred = extract_convnext_gradcam(
        model=convnext_model,
        image_tensor=convnext_tensor,
        target_class=class_for_cam,
    )
    vit_attn, vit_pred = extract_vit_attention_map(vit_model, vit_tensor)
    hybrid_cam, hybrid_pred = extract_hybrid_gradcam(
        model=hybrid_model,
        image_tensor=hybrid_tensor,
        target_class=class_for_cam,
    )
    hybrid_attn, _ = extract_hybrid_attention_map(hybrid_model, hybrid_tensor)

    convnext_map = normalize_map(resize_map(convnext_cam, panel_hw))
    vit_map = normalize_map(resize_map(vit_attn, panel_hw))
    hybrid_cam_map = normalize_map(resize_map(hybrid_cam, panel_hw))
    hybrid_attn_map = normalize_map(resize_map(hybrid_attn, panel_hw))

    avg_fusion = normalize_map(alpha * hybrid_cam_map + (1.0 - alpha) * hybrid_attn_map)
    agreement_map = normalize_map(hybrid_cam_map * hybrid_attn_map)

    panel_items = [
        ("Original", panel_image),
        ("ConvNeXt Grad-CAM", overlay_heatmap(panel_image, convnext_map)),
        ("ViT Attention", overlay_heatmap(panel_image, vit_map)),
        ("Hybrid Grad-CAM", overlay_heatmap(panel_image, hybrid_cam_map)),
        ("Hybrid Attention", overlay_heatmap(panel_image, hybrid_attn_map)),
        ("Hybrid Average Fusion", overlay_heatmap(panel_image, avg_fusion)),
        ("Hybrid Agreement Map", overlay_heatmap(panel_image, agreement_map)),
    ]
    footer_lines = [
        f"sample={image_path} | label={sample_label} | target_class={class_for_cam}",
        f"predictions: convnext={convnext_pred}, vit={vit_pred}, hybrid={hybrid_pred} | alpha={alpha:.2f}",
    ]
    return render_panel(panel_image, panel_items, footer_lines, output_path)


def _run_single(args, bundles, device: torch.device):
    _, _, hybrid_bundle = bundles
    hybrid_config, _ = hybrid_bundle
    sample_path, sample_label = resolve_sample(
        config=hybrid_config,
        split=args.split,
        sample_index=args.sample_index,
        image_path=args.image_path or None,
        label=args.label,
    )
    out_path = _build_panel(
        image_path=sample_path,
        sample_label=sample_label,
        target_class=args.target_class,
        alpha=args.alpha,
        output_path=args.output,
        convnext_bundle=bundles[0],
        vit_bundle=bundles[1],
        hybrid_bundle=bundles[2],
        device=device,
    )
    print(f"Saved explanation panel: {out_path}")


def _run_batch(args, bundles, device: torch.device):
    convnext_bundle, vit_bundle, hybrid_bundle = bundles
    hybrid_config, _ = hybrid_bundle
    data_cfg = hybrid_config.get("data", {}) or {}
    image_col = str(data_cfg.get("image_col", "image_path"))
    label_col = str(data_cfg.get("label_col", "label"))

    df = load_split_dataframe(hybrid_config, args.split)
    quota = args.samples_per_class
    if quota is None:
        quota = 50 if int(data_cfg.get("num_classes", 0)) == 2 else 20

    grouped: dict[int, list[tuple[int, Path]]] = {}
    for idx, row in df.iterrows():
        image_path = Path(str(row[image_col]).strip())
        sample_label = int(row[label_col])
        original_image = Image.open(image_path).convert("RGB")

        convnext_pred = _predict_class(convnext_bundle[1], load_tensor_from_pil(convnext_bundle[0], original_image, device))
        vit_pred = _predict_class(vit_bundle[1], load_tensor_from_pil(vit_bundle[0], original_image, device))
        hybrid_pred = _predict_class(hybrid_bundle[1], load_tensor_from_pil(hybrid_bundle[0], original_image, device))

        is_correct = hybrid_pred == sample_label
        if args.correctness_mode == "all":
            is_correct = is_correct and convnext_pred == sample_label and vit_pred == sample_label

        if not is_correct:
            continue

        bucket = grouped.setdefault(sample_label, [])
        if len(bucket) < quota:
            bucket.append((idx, image_path))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    insufficient: list[str] = []
    for class_id in sorted(grouped.keys()):
        if len(grouped[class_id]) < quota:
            insufficient.append(f"class {class_id}: found {len(grouped[class_id])}, needed {quota}")

    expected_classes = sorted({int(v) for v in df[label_col].tolist()})
    missing_classes = [class_id for class_id in expected_classes if class_id not in grouped]
    for class_id in missing_classes:
        insufficient.append(f"class {class_id}: found 0, needed {quota}")

    for class_id in expected_classes:
        class_dir = output_dir / f"class_{class_id}"
        class_dir.mkdir(parents=True, exist_ok=True)
        for rank, (sample_index, image_path) in enumerate(grouped.get(class_id, []), start=1):
            out_name = f"{rank:03d}_idx{sample_index}_{_slugify(image_path.stem)}.png"
            _build_panel(
                image_path=image_path,
                sample_label=class_id,
                target_class=class_id,
                alpha=args.alpha,
                output_path=class_dir / out_name,
                convnext_bundle=convnext_bundle,
                vit_bundle=vit_bundle,
                hybrid_bundle=hybrid_bundle,
                device=device,
            )

    print(f"Saved batch explanation panels under: {output_dir}")
    if insufficient:
        print("Quota warning:")
        for line in insufficient:
            print(f"  - {line}")


def main():
    args = parse_args()
    device = get_device(args.device)
    print(describe_device(device))
    bundles = _load_models(args, device)

    if args.batch_export:
        _run_batch(args, bundles, device)
    else:
        _run_single(args, bundles, device)


if __name__ == "__main__":
    main()
