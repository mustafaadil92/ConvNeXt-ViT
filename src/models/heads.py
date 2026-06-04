from __future__ import annotations


class ClassificationHeadSpec:
    """
    Placeholder specification for a classification head.

    Later this will become a real torch.nn.Module head implementation.
    """

    def __init__(
        self,
        in_features: int | None = None,
        num_classes: int = 2,
        dropout: float = 0.0,
    ) -> None:
        self.in_features = in_features
        self.num_classes = int(num_classes)
        self.dropout = float(dropout)

    def to_dict(self) -> dict:
        return {
            "in_features": self.in_features,
            "num_classes": self.num_classes,
            "dropout": self.dropout,
        }


def build_classification_head_spec(config: dict) -> ClassificationHeadSpec:
    """
    Build a placeholder head spec from config.
    """
    data_cfg = config.get("data", {}) or {}
    model_cfg = config.get("model", {}) or {}

    return ClassificationHeadSpec(
        in_features=model_cfg.get("head_in_features", None),
        num_classes=int(data_cfg.get("num_classes", 2)),
        dropout=float(model_cfg.get("dropout", 0.0)),
    )