from __future__ import annotations

from torch import nn

from .hybrid_vit_convnext import ViTConvNeXtHybrid
from .vit import create_vit_s


class SimpleCNNBaseline(nn.Module):
    """
    Small CNN baseline for pipeline/debug validation.
    """

    def __init__(self, num_classes: int = 3, dropout: float = 0.1) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 224 -> 112

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 112 -> 56

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 56 -> 28

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=float(dropout)),
            nn.Linear(256, int(num_classes)),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class ConvNeXtViTModelSpec:
    """
    Lightweight config summary wrapper for reporting/debugging.
    """

    def __init__(self, config: dict) -> None:
        self.config = config

        data_cfg = config.get("data", {}) or {}
        model_cfg = config.get("model", {}) or {}

        self.name = str(model_cfg.get("name", "vit_convnext_hybrid"))

        self.input_size = data_cfg.get("input_size", 224)
        if isinstance(self.input_size, int):
            self.image_size = (int(self.input_size), int(self.input_size))
        elif isinstance(self.input_size, (list, tuple)) and len(self.input_size) == 2:
            self.image_size = (int(self.input_size[0]), int(self.input_size[1]))
        else:
            raise ValueError("data.input_size must be int or [H, W]")

        self.num_classes = int(data_cfg.get("num_classes", 2))
        self.dropout = float(model_cfg.get("dropout", 0.1))
        self.patch_size = int(model_cfg.get("patch_size", 20))
        self.embed_dim = int(model_cfg.get("embed_dim", 128))
        self.num_heads = int(model_cfg.get("num_heads", 4))
        self.num_layers = int(model_cfg.get("num_layers", 8))

    def summary(self) -> dict:
        return {
            "name": self.name,
            "image_size": self.image_size,
            "num_classes": self.num_classes,
            "dropout": self.dropout,
            "patch_size": self.patch_size,
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
        }


def _parse_image_size(data_cfg: dict) -> tuple[int, int]:
    input_size = data_cfg.get("input_size", 224)

    if isinstance(input_size, int):
        return int(input_size), int(input_size)

    if isinstance(input_size, (list, tuple)) and len(input_size) == 2:
        return int(input_size[0]), int(input_size[1])

    raise ValueError("data.input_size must be int or [H, W]")


def build_model(config: dict) -> nn.Module:
    """
    Build model from config.

    Supported model names:
    - simple_cnn_baseline
    - convnext_vit_hybrid
    - vit_convnext_hybrid
    - vit-s
    - vit_s
    - vit_small
    """
    data_cfg = config.get("data", {}) or {}
    model_cfg = config.get("model", {}) or {}

    model_name = str(model_cfg.get("name", "vit_convnext_hybrid")).lower()
    num_classes = int(data_cfg.get("num_classes", 2))
    dropout = float(model_cfg.get("dropout", 0.1))

    if model_name == "simple_cnn_baseline":
        return SimpleCNNBaseline(num_classes=num_classes, dropout=dropout)

    if model_name in {"convnext_vit_hybrid", "vit_convnext_hybrid"}:
        image_h, image_w = _parse_image_size(data_cfg)

        patch_size = int(model_cfg.get("patch_size", 20))
        embed_dim = int(model_cfg.get("embed_dim", 128))
        num_heads = int(model_cfg.get("num_heads", 4))
        num_layers = int(model_cfg.get("num_layers", 8))
        in_channels = int(model_cfg.get("in_channels", 3))

        return ViTConvNeXtHybrid(
            input_channels=in_channels,
            image_size=(image_h, image_w),
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
        )

    if model_name in {"vit-s", "vit_s", "vit_small"}:
        image_h, image_w = _parse_image_size(data_cfg)
        patch_size = int(model_cfg.get("patch_size", 16))
        in_channels = int(model_cfg.get("in_channels", 3))
        embed_dim = int(model_cfg.get("embed_dim", 384))
        num_heads = int(model_cfg.get("num_heads", 6))
        depth = int(model_cfg.get("depth", model_cfg.get("num_layers", 12)))
        mlp_ratio = float(model_cfg.get("mlp_ratio", 4.0))

        return create_vit_s(
            image_size=(image_h, image_w),
            patch_size=patch_size,
            in_channels=in_channels,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

    raise ValueError(f"Unsupported model name: {model_name}")
