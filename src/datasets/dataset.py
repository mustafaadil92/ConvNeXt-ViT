from __future__ import annotations

from pathlib import Path
from typing import Callable, Any

import pandas as pd
from PIL import Image


class ImageClassificationDataset:
    """
    CSV-based image classification dataset.

    Expected CSV columns:
    - image_path
    - label

    Notes:
    - `transform` can be a torchvision transform pipeline (or compatible callable)
    - Labels are returned as int
    """

    def __init__(
        self,
        csv_path: str | Path,
        image_col: str = "image_path",
        label_col: str = "label",
        transform: Callable | None = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.image_col = image_col
        self.label_col = label_col
        self.transform = transform

        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

        self.df = pd.read_csv(self.csv_path)

        required_cols = {self.image_col, self.label_col}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")

        self.records = self.df[[self.image_col, self.label_col]].to_dict("records")

    def __len__(self) -> int:
        return len(self.records)

    def _load_image(self, image_path: str | Path) -> Image.Image:
        p = Path(image_path)
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {p}")
        # Convert to RGB to avoid grayscale/RGBA inconsistency
        return Image.open(p).convert("RGB")

    def __getitem__(self, index: int) -> tuple[Any, int]:
        row = self.records[index]

        image_path = row[self.image_col]
        label = int(row[self.label_col])

        image = self._load_image(image_path)

        if self.transform is not None:
            image = self.transform(image)

        return image, label