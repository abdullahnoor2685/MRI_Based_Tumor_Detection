import os
from pathlib import Path
from typing import Optional, Tuple
import torch
from torch.utils.data import Dataset
from PIL import Image, UnidentifiedImageError


class MRIDataset(Dataset):
    def __init__(self, root_dir: str, transform=None, class_to_idx: Optional[dict] = None) -> None:
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []

        if class_to_idx is not None:
            self.class_to_idx = class_to_idx
        else:
            classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
            self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

        for class_name, label in self.class_to_idx.items():
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in valid_extensions:
                    try:
                        with Image.open(img_path) as img:
                            img.verify()
                        self.samples.append((img_path, label))
                    except Exception:
                        print(f"Skipping corrupted image: {img_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224))
        if self.transform:
            image = self.transform(image)
        return image, label

    def get_class_counts(self) -> dict:
        counts = {cls: 0 for cls in self.class_to_idx}
        for _, label in self.samples:
            counts[self.idx_to_class[label]] += 1
        return counts
from torch.utils.data import DataLoader
from torchvision import transforms

def get_dataloaders(cfg):
    # Standard transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Path to your actual data structure (data/raw/train and data/raw/val)
    base_path = Path(__file__).parent.parent.parent 
    train_path = base_path / "data" / "raw" / "train"
    val_path = base_path / "data" / "raw" / "val"

    # Check if paths exist to avoid the WinError 3
    if not train_path.exists():
        raise FileNotFoundError(f"Missing training folder at: {train_path}")

    train_dataset = MRIDataset(root_dir=str(train_path), transform=transform)
    val_dataset = MRIDataset(root_dir=str(val_path), transform=transform)

    # Use batch size from config
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    return train_loader, val_loader, None