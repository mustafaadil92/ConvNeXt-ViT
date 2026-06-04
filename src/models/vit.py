from __future__ import annotations

import torch
from torch import nn


class PatchEmbed(nn.Module):
    """
    Standard ViT patch embedding using a strided Conv2d projection.
    """

    def __init__(
        self,
        image_size: tuple[int, int] = (224, 224),
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 384,
    ) -> None:
        super().__init__()
        self.image_h = int(image_size[0])
        self.image_w = int(image_size[1])
        self.patch_size = int(patch_size)
        self.in_channels = int(in_channels)
        self.embed_dim = int(embed_dim)

        if self.image_h % self.patch_size != 0 or self.image_w % self.patch_size != 0:
            raise ValueError("Image size must be divisible by patch_size for ViT.")

        self.grid_h = self.image_h // self.patch_size
        self.grid_w = self.image_w // self.patch_size
        self.num_patches = self.grid_h * self.grid_w

        self.proj = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x


class ViTSmall(nn.Module):
    """
    Standard ViT-S/16 style classifier.
    """

    def __init__(
        self,
        image_size: tuple[int, int] = (224, 224),
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.image_h = int(image_size[0])
        self.image_w = int(image_size[1])
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.embed_dim = int(embed_dim)

        self.patch_embed = PatchEmbed(
            image_size=(self.image_h, self.image_w),
            patch_size=int(patch_size),
            in_channels=self.in_channels,
            embed_dim=self.embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))
        self.pos_drop = nn.Dropout(float(dropout))

        mlp_dim = int(self.embed_dim * float(mlp_ratio))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=int(num_heads),
            dim_feedforward=mlp_dim,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=int(depth))
        self.norm = nn.LayerNorm(self.embed_dim)
        self.head = nn.Linear(self.embed_dim, self.num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")
        _, c, h, w = x.shape
        if c != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {c}")
        if h != self.image_h or w != self.image_w:
            raise ValueError(
                f"Expected image size ({self.image_h}, {self.image_w}), got ({h}, {w})"
            )

        x = self.patch_embed(x)
        bsz = x.shape[0]
        cls = self.cls_token.expand(bsz, -1, -1)
        x = torch.cat([cls, x], dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)
        x = self.encoder(x)
        x = self.norm(x)

        cls_out = x[:, 0]
        logits = self.head(cls_out)
        return logits


def create_vit_s(
    image_size: tuple[int, int] = (224, 224),
    patch_size: int = 16,
    in_channels: int = 3,
    num_classes: int = 1000,
    embed_dim: int = 384,
    depth: int = 12,
    num_heads: int = 6,
    mlp_ratio: float = 4.0,
    dropout: float = 0.0,
) -> ViTSmall:
    return ViTSmall(
        image_size=image_size,
        patch_size=patch_size,
        in_channels=in_channels,
        num_classes=num_classes,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
    )
