import os
import torch
import cv2
import pandas as pd
import numpy as np
# Disable albumentations remote version check to avoid network timeout warnings
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# --- CẤU HÌNH CHUNG ---
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224

class RealFakeDataset(Dataset):
    """
    Custom Dataset cho bài toán Real vs AI Images.
    Đọc ảnh bằng OpenCV để tối ưu tốc độ cho Albumentations.
    """
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        # Đảm bảo label là số nguyên
        self.label_map = {'real': 0, 'fake': 1}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['path']
        
        # Đọc ảnh bằng OpenCV (BGR) -> Chuyển sang RGB
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Lấy label từ cột 'label'
        label_name = row['label']
        label = self.label_map[label_name]

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, torch.tensor(label, dtype=torch.long)

def get_transforms():
    """
    Định nghĩa bộ Transform cho Training và Validation.
    """
    train_transform = A.Compose([
        # 1. Resize & Crop: Triệt tiêu Resolution Bias
        A.SmallestMaxSize(max_size=256),
        A.RandomCrop(width=IMG_SIZE, height=IMG_SIZE),

        # 2. Augmentation: Tăng tính tổng quát (Generalization)
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),

        # 3. Phân tích nhiễu & Nén: Chống Overfitting vào artifacts định dạng
        # Use default GaussNoise (no unsupported args) and ImageCompression with correct params
        A.GaussNoise(p=0.3),
        A.ImageCompression(quality_range=(70, 95), compression_type="jpeg", p=0.3),

        # 4. Chuẩn hóa: Đưa pixel về phân bố chuẩn
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

    val_test_transform = A.Compose([
        A.SmallestMaxSize(max_size=256),
        A.CenterCrop(width=IMG_SIZE, height=IMG_SIZE),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

    return train_transform, val_test_transform

def get_dataloaders(train_csv, val_csv, test_csv, batch_size=32, num_workers=4):
    """
    Hàm tiện ích để khởi tạo nhanh 3 DataLoaders.
    """
    train_t, val_t = get_transforms()

    train_ds = RealFakeDataset(train_csv, transform=train_t)
    val_ds = RealFakeDataset(val_csv, transform=val_t)
    test_ds = RealFakeDataset(test_csv, transform=val_t)

    # Use pinned memory only when CUDA is available to avoid PyTorch warning on CPU-only machines
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader