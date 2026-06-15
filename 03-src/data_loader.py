import os
from pathlib import Path

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import albumentations as A
import cv2
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT_DIR = PROJECT_ROOT / "01-data" / "splits"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
LABEL_MAP = {"real": 0, "fake": 1}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


class RealFakeDataset(Dataset):
    """CSV-backed dataset for real-vs-AI image classification."""

    def __init__(self, csv_path: str | Path, transform=None):
        self.csv_path = resolve_path(csv_path)
        self.df = pd.read_csv(self.csv_path)
        self.transform = transform
        self.label_map = LABEL_MAP

        required = {"path", "label"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"{self.csv_path} is missing columns: {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = resolve_path(row["path"])
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.label_map[str(row["label"]).lower()]

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        return image, torch.tensor(label, dtype=torch.long)


def get_transforms(img_size: int = 224):
    train_transform = A.Compose(
        [
            A.SmallestMaxSize(max_size=256),
            A.RandomCrop(width=img_size, height=img_size),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
            A.GaussNoise(p=0.3),
            A.ImageCompression(quality_range=(70, 95), compression_type="jpeg", p=0.3),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )

    eval_transform = A.Compose(
        [
            A.SmallestMaxSize(max_size=256),
            A.CenterCrop(width=img_size, height=img_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )

    return train_transform, eval_transform


def get_dataloaders(
    train_csv: str | Path = DEFAULT_SPLIT_DIR / "train.csv",
    val_csv: str | Path = DEFAULT_SPLIT_DIR / "val.csv",
    test_csv: str | Path = DEFAULT_SPLIT_DIR / "test.csv",
    batch_size: int = 32,
    num_workers: int = 4,
    img_size: int = 224,
):
    train_t, eval_t = get_transforms(img_size=img_size)

    train_ds = RealFakeDataset(train_csv, transform=train_t)
    val_ds = RealFakeDataset(val_csv, transform=eval_t)
    test_ds = RealFakeDataset(test_csv, transform=eval_t)
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


def validate_split_csv(csv_path: str | Path) -> dict[str, object]:
    df = pd.read_csv(resolve_path(csv_path))
    missing_paths = [p for p in df["path"].tolist() if not resolve_path(p).exists()]
    labels = df["label"].value_counts().to_dict()
    return {
        "rows": int(len(df)),
        "labels": labels,
        "missing_paths": len(missing_paths),
        "first_missing_path": missing_paths[0] if missing_paths else None,
    }
