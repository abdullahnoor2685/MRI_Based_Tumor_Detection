import os
import shutil
import random
from pathlib import Path


def split_dataset(
    source_dir: str,
    output_dir: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> None:
    random.seed(seed)
    source = Path(source_dir)
    output = Path(output_dir)

    for split in ("train", "val", "test"):
        (output / split).mkdir(parents=True, exist_ok=True)

    for class_dir in sorted(source.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        images = sorted(class_dir.glob("*"))
        images = [p for p in images if p.suffix.lower()
                  in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}]
        random.shuffle(images)

        n = len(images)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        splits = {
            "train": images[:n_train],
            "val": images[n_train: n_train + n_val],
            "test": images[n_train + n_val:],
        }

        for split_name, split_images in splits.items():
            dest = output / split_name / class_name
            dest.mkdir(parents=True, exist_ok=True)
            for img in split_images:
                shutil.copy2(img, dest / img.name)

    print(f"Dataset split complete → {output}")