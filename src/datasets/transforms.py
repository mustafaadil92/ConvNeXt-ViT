from __future__ import annotations

from torchvision import transforms


def _get_input_size(config: dict) -> tuple[int, int]:
    data_cfg = config.get("data", {}) or {}
    input_size = data_cfg.get("input_size", 224)

    if isinstance(input_size, int):
        return int(input_size), int(input_size)

    if isinstance(input_size, (list, tuple)) and len(input_size) == 2:
        return int(input_size[0]), int(input_size[1])

    raise ValueError("data.input_size must be int or [H, W]")


def build_train_transforms(config: dict):
    """
    Training transforms: resize only (+ tensor conversion for model input).
    """
    input_h, input_w = _get_input_size(config)

    return transforms.Compose(
        [
            transforms.Resize((input_h, input_w)),
            transforms.ToTensor(),
        ]
    )


def build_eval_transforms(config: dict):
    """
    Validation/Test transforms: resize only (+ tensor conversion for model input).
    """
    input_h, input_w = _get_input_size(config)

    return transforms.Compose(
        [
            transforms.Resize((input_h, input_w)),
            transforms.ToTensor(),
        ]
    )
