import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from sklearn.model_selection import GroupShuffleSplit


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_REAL_DIR = PROJECT_ROOT / "01-data" / "raw" / "Real Image"
RAW_FAKE_DIR = PROJECT_ROOT / "01-data" / "raw" / "AI Image (SDXL)"
CLEAN_DIR = PROJECT_ROOT / "01-data" / "cleaned"
CLEAN_REAL_DIR = CLEAN_DIR / "real"
CLEAN_FAKE_DIR = CLEAN_DIR / "fake"
REPORT_DIR = PROJECT_ROOT / "01-data" / "reports"
SPLIT_DIR = PROJECT_ROOT / "01-data" / "splits"
VALID_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
ImageFile.LOAD_TRUNCATED_IMAGES = False


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare DS107 real-vs-AI image dataset")
    parser.add_argument("--raw-real-dir", default=str(RAW_REAL_DIR))
    parser.add_argument("--raw-fake-dir", default=str(RAW_FAKE_DIR))
    parser.add_argument("--clean-dir", default=str(CLEAN_DIR))
    parser.add_argument("--report-dir", default=str(REPORT_DIR))
    parser.add_argument("--split-dir", default=str(SPLIT_DIR))
    parser.add_argument("--min-size", type=int, default=224)
    parser.add_argument("--blank-std-threshold", type=float, default=5.0)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-dedup", action="store_true")
    parser.add_argument("--reset-cleaned", action="store_true", help="Remove 01-data/cleaned before rebuilding it")
    parser.add_argument("--validate-only", action="store_true", help="Only validate existing split CSV files")
    return parser.parse_args()


def iter_image_files(directory: Path):
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTS:
            yield path


def real_group_id(path: Path) -> str:
    return path.stem.strip()


def fake_group_id(path: Path) -> str:
    stem = path.stem.strip()
    return stem.split("-", 1)[1].strip() if "-" in stem else stem


def standardize_image(src: Path, dst: Path) -> bool:
    try:
        with Image.open(src) as img:
            if img.mode in ("I", "I;16", "F"):
                arr = np.asarray(img, dtype=np.float32)
                arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
                img = Image.fromarray(arr.astype(np.uint8))
            if img.mode != "RGB":
                img = img.convert("RGB")
            dst.parent.mkdir(parents=True, exist_ok=True)
            img.save(dst, format="PNG")
        return True
    except Exception:
        return False


def standardize_dataset(raw_real_dir: Path, raw_fake_dir: Path, clean_real_dir: Path, clean_fake_dir: Path):
    clean_real_dir.mkdir(parents=True, exist_ok=True)
    clean_fake_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for label, src_dir, out_dir, gid_fn in [
        ("real", raw_real_dir, clean_real_dir, real_group_id),
        ("fake", raw_fake_dir, clean_fake_dir, fake_group_id),
    ]:
        ok_count = 0
        fail_count = 0
        for src in iter_image_files(src_dir):
            gid = gid_fn(src)
            dst = out_dir / f"{gid}.png"
            ok = standardize_image(src, dst)
            ok_count += int(ok)
            fail_count += int(not ok)
            rows.append({"label": label, "source": str(src), "target": str(dst), "group_id": gid, "ok": ok})
        print(f"[standardize] {label}: ok={ok_count} failed={fail_count}")

    return pd.DataFrame(rows)


def remove_invalid_pairs(clean_real_dir: Path, clean_fake_dir: Path, min_size: int, blank_std_threshold: float):
    real_ids = {p.stem for p in clean_real_dir.glob("*.png")}
    fake_ids = {p.stem for p in clean_fake_dir.glob("*.png")}
    invalid = set(real_ids ^ fake_ids)

    for gid in sorted(real_ids & fake_ids):
        for path in (clean_real_dir / f"{gid}.png", clean_fake_dir / f"{gid}.png"):
            if not image_is_valid(path, min_size, blank_std_threshold):
                invalid.add(gid)
                break

    for gid in invalid:
        (clean_real_dir / f"{gid}.png").unlink(missing_ok=True)
        (clean_fake_dir / f"{gid}.png").unlink(missing_ok=True)

    print(f"[integrity] removed invalid/unpaired groups={len(invalid)}")
    return sorted(invalid)


def image_is_valid(path: Path, min_size: int, blank_std_threshold: float) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            img = img.convert("RGB")
            w, h = img.size
            if w < min_size or h < min_size:
                return False
            if np.asarray(img).std() < blank_std_threshold:
                return False
        return True
    except Exception:
        return False


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def deduplicate(clean_real_dir: Path, clean_fake_dir: Path, report_dir: Path):
    paths = sorted(clean_real_dir.glob("*.png")) + sorted(clean_fake_dir.glob("*.png"))
    hash_map: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        hash_map[md5(path)].append(path)

    deleted = []
    duplicate_groups = []
    for digest, items in hash_map.items():
        if len(items) <= 1:
            continue
        items = sorted(items, key=lambda p: (p.parent.name != "real", str(p)))
        keep = items[0]
        removed = items[1:]
        duplicate_groups.append({"md5": digest, "keep": str(keep), "remove": [str(p) for p in removed]})
        for path in removed:
            path.unlink(missing_ok=True)
            deleted.append(str(path))

    payload = {"exact_duplicate_groups": duplicate_groups, "deleted_count": len(deleted), "deleted": deleted}
    report_dir.mkdir(parents=True, exist_ok=True)
    with (report_dir / "duplicates_report.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"[dedup] exact duplicates removed={len(deleted)}")
    return payload


def build_manifest(clean_real_dir: Path, clean_fake_dir: Path):
    records = []
    for label, directory in [("real", clean_real_dir), ("fake", clean_fake_dir)]:
        for path in sorted(directory.glob("*.png")):
            with Image.open(path) as img:
                w, h = img.size
            records.append(
                {
                    "path": str(path),
                    "label": label,
                    "width": w,
                    "height": h,
                    "aspect_ratio": round(w / h, 4),
                    "group_id": path.stem,
                }
            )
    return pd.DataFrame(records)


def save_resolution_report(df: pd.DataFrame, report_dir: Path):
    report_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(report_dir / "manifest.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for label, color in [("real", "tab:blue"), ("fake", "tab:red")]:
        sub = df[df["label"] == label]
        axes[0].scatter(sub["width"], sub["height"], alpha=0.3, c=color, label=label, s=10)
        axes[1].hist(sub["aspect_ratio"], bins=50, alpha=0.5, color=color, label=label)
    axes[0].set_title("Resolution")
    axes[0].set_xlabel("width")
    axes[0].set_ylabel("height")
    axes[0].legend()
    axes[1].set_title("Aspect ratio")
    axes[1].set_xlabel("width / height")
    df.boxplot(column="aspect_ratio", by="label", ax=axes[2])
    axes[2].set_title("Aspect ratio by label")
    axes[2].set_xlabel("label")
    fig.suptitle("")
    plt.tight_layout()
    plt.savefig(report_dir / "resolution_analysis.png", dpi=150)
    plt.close(fig)

    summary = df.groupby("label")[["width", "height", "aspect_ratio"]].describe()
    summary.to_csv(report_dir / "resolution_summary.csv")
    print(f"[report] saved manifest and resolution reports to {report_dir}")


def split_manifest(df: pd.DataFrame, split_dir: Path, val_size: float, test_size: float, seed: int):
    if val_size + test_size <= 0 or val_size + test_size >= 1:
        raise ValueError("val_size + test_size must be between 0 and 1")

    temp_size = val_size + test_size
    gss1 = GroupShuffleSplit(n_splits=1, test_size=temp_size, random_state=seed)
    train_idx, temp_idx = next(gss1.split(df, groups=df["group_id"]))
    train_df = df.iloc[train_idx].copy()
    temp_df = df.iloc[temp_idx].copy()

    relative_test_size = test_size / temp_size
    gss2 = GroupShuffleSplit(n_splits=1, test_size=relative_test_size, random_state=seed)
    val_idx, test_idx = next(gss2.split(temp_df, groups=temp_df["group_id"]))
    val_df = temp_df.iloc[val_idx].copy()
    test_df = temp_df.iloc[test_idx].copy()

    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        part["split"] = name

    split_dir.mkdir(parents=True, exist_ok=True)
    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        part.to_csv(split_dir / f"{name}.csv", index=False)
        print(f"[split] {name}: rows={len(part)} labels={part['label'].value_counts().to_dict()}")

    validate_splits(split_dir)


def validate_splits(split_dir: Path):
    frames = {}
    for name in ["train", "val", "test"]:
        path = split_dir / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        frames[name] = df
        missing = [p for p in df["path"].tolist() if not Path(p).exists()]
        if missing:
            raise FileNotFoundError(f"{name}.csv has missing image paths, first: {missing[0]}")
        print(f"[validate] {name}: rows={len(df)} labels={df['label'].value_counts().to_dict()}")

    groups = {name: set(df["group_id"]) for name, df in frames.items()}
    overlaps = {
        "train_val": groups["train"] & groups["val"],
        "train_test": groups["train"] & groups["test"],
        "val_test": groups["val"] & groups["test"],
    }
    bad = {k: len(v) for k, v in overlaps.items() if v}
    if bad:
        raise ValueError(f"Group leakage detected: {bad}")

    print("[validate] no group leakage across train/val/test")


def main():
    args = parse_args()
    raw_real_dir = Path(args.raw_real_dir)
    raw_fake_dir = Path(args.raw_fake_dir)
    clean_dir = Path(args.clean_dir)
    clean_real_dir = clean_dir / "real"
    clean_fake_dir = clean_dir / "fake"
    report_dir = Path(args.report_dir)
    split_dir = Path(args.split_dir)

    if args.validate_only:
        validate_splits(split_dir)
        return

    if args.reset_cleaned and clean_dir.exists():
        shutil.rmtree(clean_dir)

    standardize_log = standardize_dataset(raw_real_dir, raw_fake_dir, clean_real_dir, clean_fake_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    standardize_log.to_csv(report_dir / "standardization_log.csv", index=False)

    invalid_groups = remove_invalid_pairs(
        clean_real_dir,
        clean_fake_dir,
        min_size=args.min_size,
        blank_std_threshold=args.blank_std_threshold,
    )
    with (report_dir / "integrity_report.json").open("w", encoding="utf-8") as f:
        json.dump({"removed_groups": invalid_groups, "removed_count": len(invalid_groups)}, f, indent=2)

    if not args.no_dedup:
        deduplicate(clean_real_dir, clean_fake_dir, report_dir)
        remove_invalid_pairs(clean_real_dir, clean_fake_dir, args.min_size, args.blank_std_threshold)

    manifest = build_manifest(clean_real_dir, clean_fake_dir)
    save_resolution_report(manifest, report_dir)
    split_manifest(manifest, split_dir, args.val_size, args.test_size, args.seed)
    print("[done] data preparation complete")


if __name__ == "__main__":
    main()
