"""Cleaning pipeline for Real vs AI image classification."""
import sys
from pathlib import Path
import csv, json, hashlib
import numpy as np
from collections import defaultdict
from itertools import combinations
from PIL import Image, ImageFile
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from multiprocessing import Pool, cpu_count
import os

# ============ CONFIG ============
# Resolve all data paths from repository root so the script works from any CWD.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_REAL_DIR = PROJECT_ROOT / "Data/raw/Real Image"
RAW_FAKE_DIR = PROJECT_ROOT / "Data/raw/AI Image (SDXL)"
CLEAN_REAL_DIR = PROJECT_ROOT / "Data/cleaned/real"
CLEAN_FAKE_DIR = PROJECT_ROOT / "Data/cleaned/fake"
REPORT_DIR = PROJECT_ROOT / "Data/reports"
SPLIT_DIR = PROJECT_ROOT / "Data/splits"
VALID_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
ImageFile.LOAD_TRUNCATED_IMAGES = False

# ============ UTILITIES ============
def iter_image_files(directory):
    for p in sorted(directory.iterdir()):
        if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTS:
            yield p

def parse_group_id_real(path):
    return path.stem.strip()

def parse_group_id_fake(path):
    stem = path.stem.strip()
    return stem.split("-", 1)[1].strip() if "-" in stem else stem

def standardize_image(img_path, out_path):
    try:
        img = Image.open(img_path)
        if img.mode in ("I", "I;16", "F"):
            arr = np.array(img, dtype=np.float32)
            arr = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255)
            img = Image.fromarray(arr.astype(np.uint8))
        if img.mode != "RGB":
            img = img.convert("RGB")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path, format="PNG")
        return True
    except Exception as e:
        return False

def _process_image_wrapper(args):
    """Multiprocessing wrapper: (src, dst, label) -> (label, dst_name, ok)."""
    src_path, dst_path, label = args
    ok = standardize_image(str(src_path), str(dst_path))
    return (label, dst_path.stem, ok)

# ============ G1: STANDARDIZATION ============
def g1_standardization():
    print("\n╔" + "═"*68 + "╗")
    print("║" + "GIAI DOAN 1: CHUAN HOA FORMAT & CHANNEL".center(68) + "║")
    print("╚" + "═"*68 + "╝\n")
    
    CLEAN_REAL_DIR.mkdir(parents=True, exist_ok=True)
    CLEAN_FAKE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Prepare tasks
    tasks = []
    print("[G1] Preparing real images...")
    for src_path in iter_image_files(RAW_REAL_DIR):
        gid = parse_group_id_real(src_path)
        dst = CLEAN_REAL_DIR / f"{gid}.png"
        tasks.append((src_path, dst, "real"))
    
    print(f"[G1] Preparing AI images...")
    for src_path in iter_image_files(RAW_FAKE_DIR):
        gid = parse_group_id_fake(src_path)
        dst = CLEAN_FAKE_DIR / f"{gid}.png"
        tasks.append((src_path, dst, "fake"))
    
    # Process in parallel
    num_workers = max(4, cpu_count() - 1)
    print(f"[G1] Processing {len(tasks)} images with {num_workers} workers...")
    
    results = []
    with Pool(num_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(_process_image_wrapper, tasks), 1):
            results.append(result)
            if i % 500 == 0:
                print(f"  {i}/{len(tasks)} done")
    
    # Count results
    real_ok = sum(1 for l, _, ok in results if l == "real" and ok)
    real_fail = sum(1 for l, _, ok in results if l == "real" and not ok)
    fake_ok = sum(1 for l, _, ok in results if l == "fake" and ok)
    fake_fail = sum(1 for l, _, ok in results if l == "fake" and not ok)
    
    print(f"\n[G1] REAL: {real_ok} OK, {real_fail} FAIL")
    print(f"[G1] AI  : {fake_ok} OK, {fake_fail} FAIL")
    
    real_gids = {p.stem for p in CLEAN_REAL_DIR.glob("*.png")}
    fake_gids = {p.stem for p in CLEAN_FAKE_DIR.glob("*.png")}
    print(f"[G1] Summary: {len(real_gids & fake_gids)} paired | {len(real_gids - fake_gids)} real-only | {len(fake_gids - real_gids)} fake-only")
    
    return {"real_total": real_ok + real_fail, "fake_total": fake_ok + fake_fail, "real_ok": real_ok, "fake_ok": fake_ok}

# ============ G2: INTEGRITY CHECK ============
def g2_integrity():
    print("\n╔" + "═"*68 + "╗")
    print("║" + "GIAI DOAN 2: INTEGRITY CHECK".center(68) + "║")
    print("╚" + "═"*68 + "╝\n")
    
    real_dir, fake_dir = CLEAN_REAL_DIR, CLEAN_FAKE_DIR
    real_gids = {p.stem for p in real_dir.glob("*.png")}
    fake_gids = {p.stem for p in fake_dir.glob("*.png")}
    all_gids = sorted(real_gids | fake_gids)
    
    invalid = []
    for i, gid in enumerate(all_gids, 1):
        if i % 500 == 0: print(f"  Checked {i}/{len(all_gids)}")
        rp, fp = real_dir / f"{gid}.png", fake_dir / f"{gid}.png"
        
        if not (rp.exists() and fp.exists()):
            invalid.append(gid)
            continue
        
        try:
            for p in [rp, fp]:
                with Image.open(p) as img:
                    img.verify()
                with Image.open(p) as img:
                    img.load()
                    if img.size[0] < 224 or img.size[1] < 224:
                        invalid.append(gid)
                        break
                    if np.array(img).std() < 5.0:
                        invalid.append(gid)
                        break
        except:
            invalid.append(gid)
    
    print(f"\n[G2] Removing {len(invalid)} invalid groups...")
    for gid in invalid:
        (real_dir / f"{gid}.png").unlink(missing_ok=True)
        (fake_dir / f"{gid}.png").unlink(missing_ok=True)
    
    print(f"[G2] Done: {len(invalid)} groups removed")
    return len(invalid)


# ============ G3: DE-DUPLICATION ============
def get_md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def find_exact_duplicates(paths: list[Path]) -> dict:
    map_h = defaultdict(list)
    for p in paths:
        map_h[get_md5(str(p))].append(str(p))
    return {k: v for k, v in map_h.items() if len(v) > 1}

def get_phash(path: str, hash_size: int = 8, highfreq_factor: int = 4):
    img_size = hash_size * highfreq_factor
    img = Image.open(path).convert("L").resize((img_size, img_size))
    pixels = np.asarray(img, dtype=np.float32)
    freq = np.fft.fft2(pixels)
    low_freq = np.abs(freq[:hash_size, :hash_size])
    med = np.median(low_freq[1:, :]) if low_freq.size > 1 else np.median(low_freq)
    bits = (low_freq > med).flatten().astype(np.uint8)
    return bits

def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))

def compute_phash_map(paths: list[Path]) -> dict:
    return {str(p): get_phash(str(p)) for p in paths}

def ssim_global(img1: np.ndarray, img2: np.ndarray) -> float:
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    mu1 = img1.mean(); mu2 = img2.mean()
    sigma1 = ((img1 - mu1) ** 2).mean()
    sigma2 = ((img2 - mu2) ** 2).mean()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()
    num = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
    den = (mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 + sigma2 + c2)
    return float(num / (den + 1e-12))

def confirm_with_ssim(path1: str, path2: str, threshold: float = 0.95):
    a = np.array(Image.open(path1).convert("L").resize((256, 256)))
    b = np.array(Image.open(path2).convert("L").resize((256, 256)))
    score = ssim_global(a, b)
    return score >= threshold, float(score)

def g3_deduplicate(cleaned_dir: Path = PROJECT_ROOT / "Data/cleaned", hamming_threshold: int = 10, ssim_threshold: float = 0.95):
    print("\n╔" + "═"*68 + "╗")
    print("║" + "GIAI DOAN 3: DE-DUPLICATION".center(68) + "║")
    print("╚" + "═"*68 + "╝\n")

    real_paths = sorted((cleaned_dir / "real").glob("*.png"))
    fake_paths = sorted((cleaned_dir / "fake").glob("*.png"))

    # Exact duplicates
    exact_real = find_exact_duplicates(real_paths)
    exact_fake = find_exact_duplicates(fake_paths)

    # Cross exact
    cross_exact = []
    real_map = defaultdict(list)
    fake_map = defaultdict(list)
    for p in real_paths:
        real_map[get_md5(str(p))].append(str(p))
    for p in fake_paths:
        fake_map[get_md5(str(p))].append(str(p))
    for h in set(real_map) & set(fake_map):
        for r in real_map[h]:
            for f in fake_map[h]:
                cross_exact.append((r, f))

    # pHash maps
    real_ph = compute_phash_map(real_paths)
    fake_ph = compute_phash_map(fake_paths)

    # Cross near-duplicates (by hamming)
    near_cross = []
    for r, hr in real_ph.items():
        for f, hf in fake_ph.items():
            d = hamming_distance(hr, hf)
            if d <= hamming_threshold:
                near_cross.append((r, f, int(d)))

    # Confirm near_cross with SSIM and prepare deletions (keep first)
    to_delete = []
    confirmed_pairs = []
    for r, f, d in near_cross:
        ok, score = confirm_with_ssim(r, f, threshold=ssim_threshold)
        if ok:
            confirmed_pairs.append({"real": r, "fake": f, "hamming": d, "ssim": score})
            # delete fake (policy: keep real representative)
            to_delete.append(f)

    # Also handle exact duplicates: delete duplicates keeping first
    for h, items in exact_real.items():
        keep = items[0]
        for rem in items[1:]:
            to_delete.append(rem)
    for h, items in exact_fake.items():
        keep = items[0]
        for rem in items[1:]:
            to_delete.append(rem)
    for r, f in cross_exact:
        # prefer keeping real, delete fake
        to_delete.append(f)

    # Remove duplicates (unique)
    to_delete = sorted(set(to_delete))
    deleted = []
    for p in to_delete:
        try:
            Path(p).unlink()
            deleted.append(p)
        except Exception:
            pass

    # Save report
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "duplicates_report.json"
    payload = {
        "exact_real_count": len(exact_real),
        "exact_fake_count": len(exact_fake),
        "cross_exact_count": len(cross_exact),
        "near_cross_candidates": len(near_cross),
        "near_cross_confirmed": len(confirmed_pairs),
        "deleted_count": len(deleted),
        "deleted": deleted,
        "confirmed_pairs": confirmed_pairs,
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[G3] Duplicate report saved: {out}")
    print(f"[G3] Deleted files: {len(deleted)}")
    return payload

# ============ G4: RESOLUTION ANALYSIS ============
def g4_resolution():
    print("\n╔" + "═"*68 + "╗")
    print("║" + "GIAI DOAN 4: RESOLUTION ANALYSIS".center(68) + "║")
    print("╚" + "═"*68 + "╝\n")
    
    records = []
    for label in ["real", "fake"]:
        for p in (CLEAN_REAL_DIR if label == "real" else CLEAN_FAKE_DIR).glob("*.png"):
            with Image.open(p) as img:
                w, h = img.size
            records.append({"path": str(p), "label": label, "width": w, "height": h, "aspect_ratio": round(w/h, 3), "group_id": p.stem})
    
    df = pd.DataFrame(records)
    
    _, axes = plt.subplots(1, 3, figsize=(18, 5))
    for label, color in [("real", "blue"), ("fake", "red")]:
        sub = df[df.label == label]
        axes[0].scatter(sub.width, sub.height, alpha=0.3, c=color, label=label, s=10)
    axes[0].set_xlabel("Width"); axes[0].set_ylabel("Height"); axes[0].set_title("Resolution")
    
    for label, color in [("real", "blue"), ("fake", "red")]:
        sub = df[df.label == label]
        axes[1].hist(sub.aspect_ratio, bins=50, alpha=0.5, color=color, label=label)
    axes[1].set_xlabel("Aspect Ratio"); axes[1].set_title("Aspect Ratio Distribution")
    
    df.boxplot(column="aspect_ratio", by="label", ax=axes[2])
    axes[2].set_title("Boxplot"); axes[2].set_xlabel("Label")
    
    plt.tight_layout()
    out = REPORT_DIR / "resolution_analysis.png"
    plt.savefig(out, dpi=150)
    plt.close()
    
    print(df.groupby("label")[["width", "height", "aspect_ratio"]].describe())
    print(f"\n[G4] Plot saved: {out}")
    return df

# ============ G6: RUNTIME TRANSFORMS + DATALOADER CHECK ============
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def _try_import_torch():
    try:
        import torch
        return torch
    except Exception:
        return None

def _try_import_albumentations():
    try:
        import albumentations as A
        try:
            from albumentations.pytorch import ToTensorV2
            uses_to_tensorv2 = True
        except Exception:
            ToTensorV2 = None
            uses_to_tensorv2 = False
        return A, ToTensorV2, uses_to_tensorv2
    except Exception:
        return None, None, False

def check_dataloader_cropping_custom(sample_csv=str(SPLIT_DIR / "train.csv")):
    torch = _try_import_torch()
    if torch is None:
        print("[G6] Skipping DataLoader cropping check: torch not available or failed to initialize.")
        return False

    class Compose:
        def __init__(self, transforms):
            self.transforms = transforms
        def __call__(self, img):
            for t in self.transforms:
                img = t(img)
            return img

    class ResizeShortSide:
        def __init__(self, size): self.size = size
        def __call__(self, img: Image.Image):
            w, h = img.size
            if w <= h:
                new_w = self.size; new_h = int(h * self.size / w)
            else:
                new_h = self.size; new_w = int(w * self.size / h)
            return img.resize((new_w, new_h), Image.BILINEAR)

    class RandomCrop:
        def __init__(self, size): self.size = size
        def __call__(self, img: Image.Image):
            w, h = img.size
            if w < self.size or h < self.size:
                return img.resize((self.size, self.size), Image.BILINEAR)
            left = np.random.randint(0, w - self.size + 1)
            top = np.random.randint(0, h - self.size + 1)
            return img.crop((left, top, left + self.size, top + self.size))

    class CenterCrop:
        def __init__(self, size): self.size = size
        def __call__(self, img: Image.Image):
            w, h = img.size
            if w < self.size or h < self.size:
                return img.resize((self.size, self.size), Image.BILINEAR)
            left = (w - self.size) // 2; top = (h - self.size) // 2
            return img.crop((left, top, left + self.size, top + self.size))

    class ToTensorNormalize:
        def __init__(self, mean, std):
            self.mean = torch.tensor(mean).view(3,1,1)
            self.std = torch.tensor(std).view(3,1,1)
        def __call__(self, img: Image.Image):
            arr = np.asarray(img, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(arr).permute(2,0,1)
            return (tensor - self.mean) / self.std

    class RealFakeDataset(torch.utils.data.Dataset):
        def __init__(self, csv_path, transform=None):
            self.df = pd.read_csv(csv_path)
            self.transform = transform
            self.label_map = {"real": 0, "fake": 1}
        def __len__(self):
            return len(self.df)
        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            img = Image.open(row["path"]).convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
            else:
                img = torch.from_numpy(np.asarray(img, dtype=np.float32)/255.0).permute(2,0,1)
            return img, torch.tensor(self.label_map[row["label"]], dtype=torch.long)

    if not Path(sample_csv).exists():
        print(f"[G6] Split CSV not found: {sample_csv}")
        return False
    train_transform = Compose([ResizeShortSide(256), RandomCrop(224), ToTensorNormalize(IMAGENET_MEAN, IMAGENET_STD)])
    val_transform = Compose([ResizeShortSide(256), CenterCrop(224), ToTensorNormalize(IMAGENET_MEAN, IMAGENET_STD)])

    ds = RealFakeDataset(sample_csv, transform=train_transform)
    loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True, num_workers=0)
    batch_imgs, batch_labels = next(iter(loader))
    ok = isinstance(batch_imgs, torch.Tensor) and batch_imgs.shape[1] == 3 and batch_imgs.shape[2] == 224 and batch_imgs.shape[3] == 224
    print(f"[G6] DataLoader batch shape: {batch_imgs.shape} -> cropped= {ok}")
    return ok

def check_crop_pil(sample_csv=str(SPLIT_DIR / "train.csv")):
    """PIL-only resize+center-crop check on a sample from `train.csv`."""
    p = Path(sample_csv)
    if not p.exists():
        print(f"[G6-PIL] Split CSV not found: {sample_csv}")
        return False
    with open(p, newline='', encoding='utf-8') as f:
        _ = f.readline()
        line = f.readline().strip()
        if not line:
            print("[G6-PIL] train.csv is empty")
            return False
        img_path = Path(line.split(',')[0])
    if not img_path.exists():
        print(f"[G6-PIL] Sample image not found: {img_path}")
        return False

    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    if w <= h:
        new_w = 256; new_h = int(h * 256 / w)
    else:
        new_h = 256; new_w = int(w * 256 / h)
    img_rs = img.resize((new_w, new_h), Image.BILINEAR)
    w2, h2 = img_rs.size
    if w2 < 224 or h2 < 224:
        img_c = img_rs.resize((224, 224), Image.BILINEAR)
    else:
        left = (w2 - 224)//2; top = (h2 - 224)//2
        img_c = img_rs.crop((left, top, left+224, top+224))
    ok = img_c.size == (224, 224)
    print(f"[G6-PIL] After resize+crop: {img_c.size} -> cropped={ok}")
    return ok


# --- G6: runtime transforms (albumentations) + torch Dataset/DataLoader check ---
def build_alb_transforms():
    """Return (train_transform, val_transform, uses_to_tensorv2)."""
    A, ToTensorV2, uses_to_tensorv2 = _try_import_albumentations()
    if A is None:
        return (None, None, False)

    train_transform = A.Compose([
        A.SmallestMaxSize(max_size=256),
        A.CenterCrop(224, 224),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, p=0.5),
        A.RandomBrightnessContrast(0.1, 0.1, p=0.3),
        A.GaussNoise(var_limit=(0.001, 0.005), p=0.3),
        A.ImageCompression(quality_lower=70, quality_upper=95, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2() if uses_to_tensorv2 else A.pytorch.transforms.ToTensorV2() if hasattr(A, 'pytorch') else A.Normalize(),
    ])

    val_test_transform = A.Compose([
        A.SmallestMaxSize(max_size=256),
        A.CenterCrop(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2() if uses_to_tensorv2 else A.pytorch.transforms.ToTensorV2() if hasattr(A, 'pytorch') else A.Normalize(),
    ])

    return (train_transform, val_test_transform, uses_to_tensorv2)


def check_dataloader_cropping(batch_size=8):
    """Quick verify: albumentations -> DataLoader yields [B,3,224,224]."""
    torch = _try_import_torch()
    if torch is None:
        print("[G6-ALB] Skipping albumentations check: torch not available or failed to initialize.")
        return False
    if not SPLIT_DIR.exists():
        print("[G6-ALB] Splits directory missing")
        return False
    train_csv = SPLIT_DIR / 'train.csv'
    if not train_csv.exists():
        print(f"[G6-ALB] train.csv not found: {train_csv}")
        return False
    df = pd.read_csv(train_csv)
    train_t, val_t, uses_to_tensorv2 = build_alb_transforms()
    if train_t is None:
        print("[G6-ALB] albumentations not available")
        return False

    class AlbDataset(torch.utils.data.Dataset):
        def __init__(self, df, transform=None):
            self.df = df.reset_index(drop=True)
            self.transform = transform

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            img_path = row['path'] if 'path' in row else row[0]
            img = Image.open(img_path).convert('RGB')
            img_np = np.array(img)
            if self.transform is not None:
                out = self.transform(image=img_np)
                if isinstance(out, dict) and 'image' in out:
                    img_t = out['image']
                else:
                    img_t = out
            else:
                img_t = np.transpose(img_np / 255.0, (2,0,1)).astype('float32')
                img_t = torch.from_numpy(img_t)

            # label handling: expect 'label' column (0/1) or infer from path
            if 'label' in self.df.columns:
                label = int(row['label'])
            else:
                label = 1 if 'fake' in str(img_path).lower() or 'ai' in str(img_path).lower() else 0

            return img_t, label

    ds = AlbDataset(df, transform=train_t)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    batch = next(iter(dl))
    imgs, labels = batch
    try:
        ok = imgs.shape[1:] == (3,224,224)
    except Exception:
        ok = False
    print(f"[G6-ALB] Batch imgs.shape={getattr(imgs,'shape',None)}, ok={ok}")
    return ok

# ============ G5: SPLIT ============
def g5_split(df):
    print("\n╔" + "═"*68 + "╗")
    print("║" + "GIAI DOAN 5: GROUP-AWARE SPLIT".center(68) + "║")
    print("╚" + "═"*68 + "╝\n")
    
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, temp_idx = next(gss1.split(df, groups=df["group_id"]))
    train_df, temp_df = df.iloc[train_idx].copy(), df.iloc[temp_idx].copy()
    
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    val_idx, test_idx = next(gss2.split(temp_df, groups=temp_df["group_id"]))
    val_df, test_df = temp_df.iloc[val_idx].copy(), temp_df.iloc[test_idx].copy()
    
    for d, name in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
        d["split"] = name
    
    train_gids = set(train_df["group_id"])
    val_gids = set(val_df["group_id"])
    test_gids = set(test_df["group_id"])
    assert len(train_gids & val_gids) == 0, "LEAK"
    assert len(train_gids & test_gids) == 0, "LEAK"
    assert len(val_gids & test_gids) == 0, "LEAK"
    print("[G5] ✓ No data leakage")
    
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    for d, name in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
        d.to_csv(SPLIT_DIR / f"{name}.csv", index=False)
    
    print(f"\n[G5] Train: {len(train_df)} ({train_df.label.value_counts().to_dict()})")
    print(f"[G5] Val  : {len(val_df)} ({val_df.label.value_counts().to_dict()})")
    print(f"[G5] Test : {len(test_df)} ({test_df.label.value_counts().to_dict()})")
    print(f"[G5] CSV saved to: {SPLIT_DIR}")

# ============ MAIN ============
if __name__ == "__main__":
    try:
        g1_stats = g1_standardization()
        removed = g2_integrity()
        # G3: de-duplication
        g3_report = g3_deduplicate()
        df = g4_resolution()
        g5_split(df)
        # G6: quick DataLoader cropping check (torch optional)
        if os.environ.get("SKIP_TORCH") == "1":
            check_ok = check_crop_pil()
        elif _try_import_torch() is not None:
            check_ok = check_dataloader_cropping_custom()
        else:
            check_ok = check_crop_pil()
        print(f"[G6] Cropping check passed: {check_ok}")
        
        print("\n╔" + "═"*68 + "╗")
        print("║" + "✓ CLEANING PIPELINE COMPLETE".center(68) + "║")
        print("╚" + "═"*68 + "╝\n")
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
