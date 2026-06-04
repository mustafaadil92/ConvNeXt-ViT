from __future__ import annotations

from torch import nn
from torchvision.models import convnext_base, convnext_small, convnext_tiny


def build_convnext_model(config: dict) -> nn.Module:
    """
    Build a pure ConvNeXt classifier from config.

    Supported names in model.name:
    - convnext
    - convnext_tiny
    - convnext_small
    - convnext_base
    """
    data_cfg = config.get("data", {}) or {}
    model_cfg = config.get("model", {}) or {}

    model_name = str(model_cfg.get("name", "convnext")).lower()
    variant = str(model_cfg.get("variant", model_name)).lower()
    num_classes = int(data_cfg.get("num_classes", 2))
    use_pretrained = bool(model_cfg.get("pretrained", False))

    if variant in {"convnext", "convnext_tiny"}:
        weights = "DEFAULT" if use_pretrained else None
        model = convnext_tiny(weights=weights)
    elif variant == "convnext_small":
        weights = "DEFAULT" if use_pretrained else None
        model = convnext_small(weights=weights)
    elif variant == "convnext_base":
        weights = "DEFAULT" if use_pretrained else None
        model = convnext_base(weights=weights)
    else:
        raise ValueError(
            f"Unsupported convnext variant: {variant}. "
            f"Expected one of: convnext_tiny, convnext_small, convnext_base."
        )

    in_features = int(model.classifier[2].in_features)
    model.classifier[2] = nn.Linear(in_features, num_classes)
    return model
