from __future__ import annotations

import csv
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import pandas as pd

from src.datasets.transforms import build_eval_transforms
from src.models import build_model
from src.utils import load_yaml_config


def load_config(path: str | Path) -> dict:
    return load_yaml_config(str(path))


def load_model_from_config(
    config_path: str | Path,
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[dict, torch.nn.Module]:
    config = load_config(config_path)
    model = build_model(config).to(device)

    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()
    return config, model


def resolve_sample(
    config: dict,
    split: str,
    sample_index: int,
    image_path: str | None = None,
    label: int | None = None,
) -> tuple[Path, int | None]:
    if image_path:
        return Path(image_path), label

    data_cfg = config.get("data", {}) or {}
    split_key = f"{split}_csv"
    csv_path = Path(str(data_cfg.get(split_key, "")).strip())
    if not csv_path.exists():
        raise FileNotFoundError(f"Split CSV not found for '{split}': {csv_path}")

    image_col = str(data_cfg.get("image_col", "image_path"))
    label_col = str(data_cfg.get("label_col", "label"))

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise RuntimeError(f"CSV has no rows: {csv_path}")
    if sample_index < 0 or sample_index >= len(rows):
        raise IndexError(f"sample_index={sample_index} is out of range for {csv_path}")

    row = rows[sample_index]
    path = Path(str(row[image_col]).strip())
    label_val = row.get(label_col, None)
    return path, int(label_val) if label_val not in (None, "") else None


def load_input_tensor(config: dict, image_path: str | Path, device: torch.device) -> tuple[Image.Image, torch.Tensor]:
    image = Image.open(image_path).convert("RGB")
    tensor = build_eval_transforms(config)(image).unsqueeze(0).to(device)
    return image, tensor


def load_tensor_from_pil(config: dict, image: Image.Image, device: torch.device) -> torch.Tensor:
    return build_eval_transforms(config)(image).unsqueeze(0).to(device)


def normalize_map(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = x.detach().to(torch.float32)
    x = x - x.min()
    denom = x.max().clamp_min(eps)
    return x / denom


def resize_map(x: torch.Tensor, size_hw: tuple[int, int]) -> torch.Tensor:
    if x.ndim == 2:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.ndim == 3:
        x = x.unsqueeze(1)
    out = F.interpolate(x, size=size_hw, mode="bilinear", align_corners=False)
    return out.squeeze(0).squeeze(0)


@contextmanager
def capture_attention(module: torch.nn.Module):
    original_forward = module.forward
    captured: dict[str, torch.Tensor | None] = {"weights": None}

    def wrapped_forward(*args, **kwargs):
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = False
        output = original_forward(*args, **kwargs)
        if isinstance(output, tuple) and len(output) > 1:
            captured["weights"] = output[1].detach()
        return output

    module.forward = wrapped_forward
    try:
        yield captured
    finally:
        module.forward = original_forward


def extract_vit_attention_map(model: torch.nn.Module, image_tensor: torch.Tensor) -> tuple[torch.Tensor, int]:
    attn_module = model.encoder.layers[-1].self_attn
    with capture_attention(attn_module) as captured:
        logits = model(image_tensor)

    pred_class = int(logits.argmax(dim=1).item())
    weights = captured["weights"]
    if weights is None:
        raise RuntimeError("Failed to capture ViT attention weights.")

    attn = weights[0].mean(dim=0)  # (tokens, tokens)
    cls_to_patch = attn[0, 1:]
    grid_h = int(model.patch_embed.grid_h)
    grid_w = int(model.patch_embed.grid_w)
    return normalize_map(cls_to_patch.view(grid_h, grid_w)), pred_class


def extract_hybrid_attention_map(model: torch.nn.Module, image_tensor: torch.Tensor) -> tuple[torch.Tensor, int]:
    attn_module = model.blocks[-1].mha
    with capture_attention(attn_module) as captured:
        logits = model(image_tensor)

    pred_class = int(logits.argmax(dim=1).item())
    weights = captured["weights"]
    if weights is None:
        raise RuntimeError("Failed to capture hybrid attention weights.")

    attn = weights[0].mean(dim=0)  # (tokens, tokens)
    cls_to_patch = attn[0, 1:]
    grid_h = int(model.num_patches_h)
    grid_w = int(model.num_patches_w)
    return normalize_map(cls_to_patch.view(grid_h, grid_w)), pred_class


def compute_gradcam(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    target_module: torch.nn.Module,
    target_class: int | None = None,
) -> tuple[torch.Tensor, int]:
    activations: dict[str, torch.Tensor] = {}
    gradients: dict[str, torch.Tensor] = {}

    def forward_hook(_module, _inp, output):
        activations["value"] = output

    def backward_hook(_module, _grad_input, grad_output):
        gradients["value"] = grad_output[0]

    handle_fwd = target_module.register_forward_hook(forward_hook)
    handle_bwd = target_module.register_full_backward_hook(backward_hook)

    try:
        model.zero_grad(set_to_none=True)
        logits = model(image_tensor)
        pred_class = int(logits.argmax(dim=1).item())
        class_idx = pred_class if target_class is None else int(target_class)
        score = logits[:, class_idx].sum()
        score.backward()
    finally:
        handle_fwd.remove()
        handle_bwd.remove()

    if "value" not in activations or "value" not in gradients:
        raise RuntimeError("Failed to capture Grad-CAM activations/gradients.")

    acts = activations["value"]
    grads = gradients["value"]
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * acts).sum(dim=1))
    cam = normalize_map(cam[0])
    return cam, pred_class


def extract_convnext_gradcam(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    target_class: int | None = None,
) -> tuple[torch.Tensor, int]:
    return compute_gradcam(
        model=model,
        image_tensor=image_tensor,
        target_module=model.features[7],
        target_class=target_class,
    )


def extract_hybrid_gradcam(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    target_class: int | None = None,
) -> tuple[torch.Tensor, int]:
    return compute_gradcam(
        model=model,
        image_tensor=image_tensor,
        target_module=model.blocks[-1].dwconv,
        target_class=target_class,
    )


def compute_scorecam(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    target_module: torch.nn.Module,
    target_class: int | None = None,
    max_channels: int = 32,
) -> tuple[torch.Tensor, int]:
    activations: dict[str, torch.Tensor] = {}

    def forward_hook(_module, _inp, output):
        activations["value"] = output.detach()

    handle = target_module.register_forward_hook(forward_hook)
    try:
        with torch.no_grad():
            logits = model(image_tensor)
    finally:
        handle.remove()

    if "value" not in activations:
        raise RuntimeError("Failed to capture Score-CAM activations.")

    pred_class = int(logits.argmax(dim=1).item())
    class_idx = pred_class if target_class is None else int(target_class)
    base_score = float(torch.softmax(logits, dim=1)[0, class_idx].item())

    acts = activations["value"][0]
    channel_energy = acts.mean(dim=(1, 2))
    topk = min(int(max_channels), int(acts.shape[0]))
    top_indices = torch.topk(channel_energy, k=topk, largest=True).indices

    score_map = torch.zeros(acts.shape[1:], dtype=torch.float32, device=image_tensor.device)
    weights: list[float] = []
    maps: list[torch.Tensor] = []

    with torch.no_grad():
        for channel_idx in top_indices.tolist():
            act_map = acts[channel_idx]
            act_map = normalize_map(act_map)
            up_map = resize_map(act_map, (image_tensor.shape[-2], image_tensor.shape[-1]))
            masked_input = image_tensor * up_map.unsqueeze(0).unsqueeze(0)
            masked_logits = model(masked_input)
            masked_score = float(torch.softmax(masked_logits, dim=1)[0, class_idx].item())
            weight = max(0.0, masked_score - base_score)
            weights.append(weight)
            maps.append(act_map)

    if not maps:
        return normalize_map(score_map), pred_class

    if max(weights) <= 0.0:
        weights = [1.0 for _ in weights]

    total_weight = sum(weights)
    if total_weight <= 0:
        total_weight = float(len(weights))

    for weight, act_map in zip(weights, maps):
        score_map = score_map + float(weight) * act_map

    score_map = torch.relu(score_map / total_weight)
    return normalize_map(score_map), pred_class


def extract_convnext_scorecam(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    target_class: int | None = None,
    max_channels: int = 32,
) -> tuple[torch.Tensor, int]:
    return compute_scorecam(
        model=model,
        image_tensor=image_tensor,
        target_module=model.features[7],
        target_class=target_class,
        max_channels=max_channels,
    )


def extract_hybrid_scorecam(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    target_class: int | None = None,
    max_channels: int = 32,
) -> tuple[torch.Tensor, int]:
    return compute_scorecam(
        model=model,
        image_tensor=image_tensor,
        target_module=model.blocks[-1].dwconv,
        target_class=target_class,
        max_channels=max_channels,
    )


def compute_eigencam(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    target_module: torch.nn.Module,
) -> tuple[torch.Tensor, int]:
    activations: dict[str, torch.Tensor] = {}

    def forward_hook(_module, _inp, output):
        activations["value"] = output.detach()

    handle = target_module.register_forward_hook(forward_hook)
    try:
        with torch.no_grad():
            logits = model(image_tensor)
    finally:
        handle.remove()

    if "value" not in activations:
        raise RuntimeError("Failed to capture Eigen-CAM activations.")

    pred_class = int(logits.argmax(dim=1).item())
    acts = activations["value"][0].to(torch.float32)  # (C, H, W)
    c, h, w = acts.shape
    flat = acts.view(c, h * w).transpose(0, 1)  # (HW, C)
    flat = flat - flat.mean(dim=0, keepdim=True)

    # Principal spatial component over channels.
    _, _, v = torch.linalg.svd(flat, full_matrices=False)
    principal = v[0]  # (C,)
    cam = torch.matmul(flat, principal).view(h, w)
    cam = torch.relu(cam)
    return normalize_map(cam), pred_class


def extract_convnext_eigencam(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    return compute_eigencam(
        model=model,
        image_tensor=image_tensor,
        target_module=model.features[7],
    )


def extract_hybrid_eigencam(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    return compute_eigencam(
        model=model,
        image_tensor=image_tensor,
        target_module=model.blocks[-1].dwconv,
    )


def _heatmap_rgb(map_2d: torch.Tensor) -> torch.Tensor:
    x = normalize_map(map_2d)
    r = x
    g = torch.clamp(1.0 - (2.0 * (x - 0.5)).abs(), 0.0, 1.0)
    b = 1.0 - x
    return torch.stack([r, g, b], dim=0)


def overlay_heatmap(base_image: Image.Image, map_2d: torch.Tensor, alpha: float = 0.45) -> Image.Image:
    target_w = int(map_2d.shape[-1])
    target_h = int(map_2d.shape[-2])
    base_image = base_image.resize((target_w, target_h))
    base = torch.tensor(list(base_image.getdata()), dtype=torch.float32)
    base = base.view(target_h, target_w, 3).permute(2, 0, 1) / 255.0
    heat = _heatmap_rgb(map_2d).to(torch.float32)
    blended = ((1.0 - alpha) * base + alpha * heat).clamp(0.0, 1.0)
    arr = (blended.permute(1, 2, 0).cpu().numpy() * 255.0).astype("uint8")
    return Image.fromarray(arr)


def render_panel(
    original_image: Image.Image,
    panel_items: list[tuple[str, Image.Image]],
    footer_lines: list[str],
    out_path: str | Path,
) -> Path:
    tile_w, tile_h = original_image.size
    title_h = 28
    gap = 18
    cols = len(panel_items)
    rows = 1
    footer_h = 56
    canvas_w = gap + cols * (tile_w + gap)
    canvas_h = gap + rows * (title_h + tile_h + gap) + footer_h

    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for idx, (title, image) in enumerate(panel_items):
        row = idx // cols
        col = idx % cols
        x = gap + col * (tile_w + gap)
        y = gap + row * (title_h + tile_h + gap)
        draw.text((x, y), title, fill="black", font=font)
        canvas.paste(image.resize((tile_w, tile_h)), (x, y + title_h))
        draw.rectangle([x, y + title_h, x + tile_w, y + title_h + tile_h], outline=(180, 180, 180), width=1)

    footer_y = gap + rows * (title_h + tile_h + gap)
    for i, line in enumerate(footer_lines):
        draw.text((gap, footer_y + 4 + i * 16), line, fill="black", font=font)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return out_path


def load_split_dataframe(config: dict, split: str) -> pd.DataFrame:
    data_cfg = config.get("data", {}) or {}
    split_key = f"{split}_csv"
    csv_path = Path(str(data_cfg.get(split_key, "")).strip())
    if not csv_path.exists():
        raise FileNotFoundError(f"Split CSV not found for '{split}': {csv_path}")
    return pd.read_csv(csv_path)
