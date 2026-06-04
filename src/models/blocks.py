from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class ConvBlockSpec:
    name: str = "convnext_block"
    channels: int = 96
    depth: int = 1


@dataclass
class TransformerBlockSpec:
    name: str = "vit_block"
    embed_dim: int = 192
    num_heads: int = 3
    mlp_ratio: float = 4.0
    depth: int = 1


@dataclass
class FusionBlockSpec:
    name: str = "fusion_block"
    method: str = "concat"  # concat | add | gated
    out_dim: int | None = None


def build_block_specs(config: dict) -> dict:
    """
    Build placeholder block specifications from config.
    Later these map to real torch modules.
    """
    model_cfg = config.get("model", {}) or {}

    conv_spec = ConvBlockSpec(
        channels=int(model_cfg.get("conv_channels", 96)),
        depth=int(model_cfg.get("conv_depth", 1)),
    )

    vit_spec = TransformerBlockSpec(
        embed_dim=int(model_cfg.get("embed_dim", 192)),
        num_heads=int(model_cfg.get("num_heads", 3)),
        mlp_ratio=float(model_cfg.get("mlp_ratio", 4.0)),
        depth=int(model_cfg.get("vit_depth", 1)),
    )

    fusion_spec = FusionBlockSpec(
        method=str(model_cfg.get("fusion_method", "concat")).lower(),
        out_dim=model_cfg.get("fusion_out_dim", None),
    )

    return {
        "conv": asdict(conv_spec),
        "vit": asdict(vit_spec),
        "fusion": asdict(fusion_spec),
    }