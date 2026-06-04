from __future__ import annotations

import math

import torch
from torch import nn


class PatchExtractor(nn.Module):
    """
    Extract non-overlapping patches using nn.Unfold.
    Output shape: (B, N, patch_dim)
    """

    def __init__(self, patch_size: int) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        self.unfold = nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: (B, C, H, W)
        # unfold -> (B, C*P*P, N)
        patches = self.unfold(images)
        # -> (B, N, C*P*P)
        patches = patches.transpose(1, 2).contiguous()
        return patches


class PatchEncoder(nn.Module):
    """
    Linear projection + positional embedding + CLS token.
    """

    def __init__(self, num_patches: int, patch_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.num_patches = int(num_patches)
        self.patch_dim = int(patch_dim)
        self.embed_dim = int(embed_dim)

        self.projection = nn.Linear(self.patch_dim, self.embed_dim)

        # Match TF intent:
        # pos_embedding: (1, N+1, D), cls_token: (1, 1, D)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.num_patches + 1, self.embed_dim) * 0.02
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        # patch_tokens: (B, N, patch_dim)
        bsz = patch_tokens.size(0)

        x = self.projection(patch_tokens)  # (B, N, D)

        cls = self.cls_token.expand(bsz, -1, -1)  # (B, 1, D)
        x = torch.cat([cls, x], dim=1)  # (B, N+1, D)

        x = x + self.pos_embedding
        return x


class ChannelLayerNorm2d(nn.Module):
    """
    LayerNorm over channels for NCHW tensors (equivalent to applying LN on NHWC channels).
    TF code applies LayerNormalization on channels-last conv features.
    """

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, H, W, C) -> LN(C) -> back
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class HybridViTConvNeXtBlock(nn.Module):
    """
    One hybrid block:
      x -> LN -> MHSA -> Drop -> Add
      x -> LN -> (ConvNeXt-like local branch over patch tokens only) -> Add
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_patches_h: int,
        num_patches_w: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.num_patches_h = int(num_patches_h)
        self.num_patches_w = int(num_patches_w)
        self.dropout_p = float(dropout)

        # 1) Transformer attention branch
        self.ln1 = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.mha = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=self.dropout_p,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(self.dropout_p)

        # 2) ConvNeXt-like branch over patch tokens only
        self.ln2 = nn.LayerNorm(self.embed_dim, eps=1e-6)

        self.dwconv = nn.Conv2d(
            self.embed_dim,
            self.embed_dim,
            kernel_size=7,
            padding=3,
            groups=self.embed_dim,  # depthwise
        )
        self.ln_conv = ChannelLayerNorm2d(self.embed_dim, eps=1e-6)
        self.pw1 = nn.Conv2d(self.embed_dim, 4 * self.embed_dim, kernel_size=1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(4 * self.embed_dim, self.embed_dim, kernel_size=1)
        self.ffn_dropout = nn.Dropout(self.dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N+1, D)

        # ---- 1) MHSA branch ----
        shortcut = x
        x_norm1 = self.ln1(x)
        attn_out, _ = self.mha(x_norm1, x_norm1, x_norm1, need_weights=False)
        attn_out = self.attn_dropout(attn_out)
        x = shortcut + attn_out

        # ---- 2) ConvNeXt-like branch over patch tokens ----
        shortcut2 = x
        x_norm2 = self.ln2(x)  # (B, N+1, D)

        cls_token = x_norm2[:, :1, :]    # (B, 1, D)
        patches = x_norm2[:, 1:, :]      # (B, N, D)

        bsz, n_patches, dim = patches.shape
        expected = self.num_patches_h * self.num_patches_w
        if n_patches != expected:
            raise ValueError(
                f"Patch token count mismatch: got {n_patches}, expected {expected} "
                f"({self.num_patches_h}x{self.num_patches_w})"
            )
        if dim != self.embed_dim:
            raise ValueError(f"Embed dim mismatch: got {dim}, expected {self.embed_dim}")

        # (B, N, D) -> (B, H_p, W_p, D) -> (B, D, H_p, W_p)
        patches_2d = patches.view(bsz, self.num_patches_h, self.num_patches_w, self.embed_dim)
        patches_2d = patches_2d.permute(0, 3, 1, 2).contiguous()

        y = self.dwconv(patches_2d)
        y = self.ln_conv(y)
        y = self.pw1(y)
        y = self.act(y)
        y = self.pw2(y)
        y = self.ffn_dropout(y)

        # Residual within spatial tokens
        patches_2d_out = patches_2d + y

        # Back to sequence: (B, D, H_p, W_p) -> (B, H_p, W_p, D) -> (B, N, D)
        patches_out = patches_2d_out.permute(0, 2, 3, 1).contiguous()
        patches_out = patches_out.view(bsz, expected, self.embed_dim)

        # Re-attach CLS
        x_ffn = torch.cat([cls_token, patches_out], dim=1)  # (B, N+1, D)

        # Final residual (full token set)
        x = shortcut2 + x_ffn
        return x


class ViTConvNeXtHybrid(nn.Module):
    """
    PyTorch port of the user's TensorFlow hybrid:
    - patch extraction
    - patch encoding + CLS + positional embedding
    - stacked hybrid MHSA + ConvNeXt-token blocks
    - CLS + attention pooling classification head

    Returns raw logits (no softmax), suitable for CrossEntropyLoss.
    """

    def __init__(
        self,
        input_channels: int = 3,
        image_size: tuple[int, int] = (460, 460),
        patch_size: int = 20,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        num_classes: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.input_channels = int(input_channels)
        self.image_h = int(image_size[0])
        self.image_w = int(image_size[1])
        self.patch_size = int(patch_size)
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        self.num_classes = int(num_classes)
        self.dropout = float(dropout)

        if self.image_h % self.patch_size != 0:
            raise ValueError("Image height must be divisible by patch_size")
        if self.image_w % self.patch_size != 0:
            raise ValueError("Image width must be divisible by patch_size")

        self.num_patches_h = self.image_h // self.patch_size
        self.num_patches_w = self.image_w // self.patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w
        self.patch_dim = self.input_channels * self.patch_size * self.patch_size

        self.patch_extractor = PatchExtractor(self.patch_size)
        self.patch_encoder = PatchEncoder(
            num_patches=self.num_patches,
            patch_dim=self.patch_dim,
            embed_dim=self.embed_dim,
        )

        self.blocks = nn.ModuleList(
            [
                HybridViTConvNeXtBlock(
                    embed_dim=self.embed_dim,
                    num_heads=self.num_heads,
                    num_patches_h=self.num_patches_h,
                    num_patches_w=self.num_patches_w,
                    dropout=self.dropout,
                )
                for _ in range(self.num_layers)
            ]
        )

        # Classification head
        self.attn_score = nn.Linear(self.embed_dim, 1)

        self.head_norm = nn.LayerNorm(2 * self.embed_dim)
        self.head_dense_1 = nn.Linear(2 * self.embed_dim, 256)
        self.head_act_1 = nn.GELU()
        self.head_dropout_1 = nn.Dropout(0.2)

        self.head_dense_2 = nn.Linear(256, 128)
        self.head_act_2 = nn.GELU()
        self.head_dropout_2 = nn.Dropout(0.2)

        self.classifier = nn.Linear(128, self.num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")

        _, c, h, w = x.shape
        if c != self.input_channels:
            raise ValueError(f"Expected {self.input_channels} channels, got {c}")
        if h != self.image_h or w != self.image_w:
            raise ValueError(
                f"Expected image size ({self.image_h}, {self.image_w}), got ({h}, {w})"
            )

        x = self.patch_extractor(x)  # (B, N, patch_dim)
        x = self.patch_encoder(x)    # (B, N+1, D)

        for blk in self.blocks:
            x = blk(x)

        return x  # token sequence (B, N+1, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)  # (B, N+1, D)

        # CLS token
        cls = x[:, 0, :]  # (B, D)

        # Attention pooling over all tokens
        scores = self.attn_score(x)              # (B, N+1, 1)
        weights = torch.softmax(scores, dim=1)   # (B, N+1, 1)
        attn_pool = (x * weights).sum(dim=1)     # (B, D)

        # Combine CLS + pooled representation
        combined = torch.cat([cls, attn_pool], dim=1)  # (B, 2D)

        # MLP head
        h = self.head_norm(combined)
        h = self.head_dense_1(h)
        h = self.head_act_1(h)
        h = self.head_dropout_1(h)

        h = self.head_dense_2(h)
        h = self.head_act_2(h)
        h = self.head_dropout_2(h)

        logits = self.classifier(h)  # (B, num_classes)
        return logits


def create_vit_convnext_hybrid(
    input_shape: tuple[int, int, int] = (460, 460, 3),  # H, W, C (TF-style for compatibility)
    patch_size: int = 20,
    embed_dim: int = 128,
    num_heads: int = 4,
    num_layers: int = 8,
    num_classes: int = 8,
    dropout: float = 0.1,
) -> ViTConvNeXtHybrid:
    """
    Compatibility factory matching your TensorFlow function signature style.
    """
    h, w, c = input_shape
    return ViTConvNeXtHybrid(
        input_channels=int(c),
        image_size=(int(h), int(w)),
        patch_size=int(patch_size),
        embed_dim=int(embed_dim),
        num_heads=int(num_heads),
        num_layers=int(num_layers),
        num_classes=int(num_classes),
        dropout=float(dropout),
    )