from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class MetricTracker:
    """
    Simple running average tracker for scalar metrics (e.g., loss).
    """
    name: str
    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * int(n)
        self.count += int(n)

    @property
    def avg(self) -> float:
        return self.total / self.count if self.count > 0 else 0.0

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["avg"] = self.avg
        return d


def accuracy_from_counts(correct: int, total: int) -> float:
    """
    Compute accuracy from integer counts.
    """
    if total <= 0:
        return 0.0
    return float(correct) / float(total)


def summarize_epoch_metrics(metrics: dict[str, float]) -> dict[str, float]:
    """
    Placeholder hook for epoch metric formatting/post-processing.
    For now, returns a shallow normalized float dict.
    """
    out: dict[str, float] = {}
    for k, v in metrics.items():
        out[str(k)] = float(v)
    return out