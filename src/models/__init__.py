from .blocks import build_block_specs
from .heads import ClassificationHeadSpec, build_classification_head_spec
from .convnext import build_convnext_model
from .convnext_vit import ConvNeXtViTModelSpec, build_model as build_model_hybrid_or_vit


def build_model(config: dict):
    model_cfg = config.get("model", {}) or {}
    model_name = str(model_cfg.get("name", "vit_convnext_hybrid")).lower()

    if model_name in {"convnext", "convnext_tiny", "convnext_small", "convnext_base"}:
        return build_convnext_model(config)

    return build_model_hybrid_or_vit(config)


__all__ = [
    "build_block_specs",
    "ClassificationHeadSpec",
    "build_classification_head_spec",
    "ConvNeXtViTModelSpec",
    "build_model",
]
