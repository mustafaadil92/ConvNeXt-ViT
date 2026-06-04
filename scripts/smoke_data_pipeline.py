
from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

from src.datasets import build_datasets, build_dataloaders


def make_dummy_image(path: Path, size: int = 224) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    img = Image.new("RGB", (size, size), color=(30, 30, 30))
    draw = ImageDraw.Draw(img)
    draw.rectangle((20, 20, size - 20, size - 20), outline=(255, 0, 0), width=4)
    draw.text((40, 40), "dummy", fill=(255, 255, 0))
    img.save(path)


def main():
    project_root = Path(__file__).resolve().parents[1]
    dummy_dir = project_root / "data" / "processed" / "dummy"
    csv_path = project_root / "data" / "folds" / "dummy_train.csv"
    img_path = dummy_dir / "img0.png"

    make_dummy_image(img_path, size=256)

    df = pd.DataFrame(
        [
            {"image_path": str(img_path), "label": 0},
            {"image_path": str(img_path), "label": 1},
            {"image_path": str(img_path), "label": 2},
        ]
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    config = {
        "data": {
            "train_csv": str(csv_path),
            "val_csv": str(csv_path),
            "test_csv": "",
            "image_col": "image_path",
            "label_col": "label",
            "num_classes": 3,
            "input_size": 224,
            "num_workers": 0,   # safer for Windows smoke test
            "pin_memory": True,
        },
        "train": {
            "batch_size": 2,
        },
    }

    datasets = build_datasets(config)
    loaders = build_dataloaders(config)

    print("Datasets:", {k: len(v) for k, v in datasets.items()})
    print("Loaders:", list(loaders.keys()))

    batch = next(iter(loaders["train"]))
    images, labels = batch

    print("Batch images shape:", tuple(images.shape))
    print("Batch labels shape:", tuple(labels.shape))
    print("Batch labels:", labels.tolist())
    print("Data pipeline smoke test OK")


if __name__ == "__main__":
    main()