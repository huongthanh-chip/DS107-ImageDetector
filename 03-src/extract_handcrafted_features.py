import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from data_loader import DEFAULT_SPLIT_DIR, LABEL_MAP, resolve_path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT_ROOT / "01-data" / "features" / "handcrafted_features.csv"


def parse_args():
    parser = argparse.ArgumentParser(description="Extract handcrafted image features for Real/Fake analysis")
    parser.add_argument("--split-dir", default=str(DEFAULT_SPLIT_DIR))
    parser.add_argument("--out-csv", default=str(DEFAULT_OUT))
    parser.add_argument("--resize", type=int, default=256, help="Resize longest side for feature extraction speed")
    return parser.parse_args()


def load_split_df(split_dir: str | Path) -> pd.DataFrame:
    split_dir = Path(split_dir)
    frames = []
    for split in ["train", "val", "test"]:
        df = pd.read_csv(split_dir / f"{split}.csv")
        df["split"] = split
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def resize_longest_side(image: np.ndarray, max_side: int) -> np.ndarray:
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return image
    scale = max_side / longest
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def safe_entropy(gray: np.ndarray) -> float:
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    probs = hist / max(hist.sum(), 1.0)
    probs = probs[probs > 0]
    return float(-(probs * np.log2(probs)).sum())


def high_frequency_ratio(gray: np.ndarray) -> tuple[float, float, float]:
    gray_f = gray.astype(np.float32) / 255.0
    spectrum = np.fft.fftshift(np.fft.fft2(gray_f))
    power = np.abs(spectrum) ** 2
    h, w = power.shape
    cy, cx = h // 2, w // 2
    radius = max(4, min(h, w) // 8)
    yy, xx = np.ogrid[:h, :w]
    low_mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2
    low = float(power[low_mask].mean())
    high = float(power[~low_mask].mean())
    return low, high, high / (low + 1e-8)


def extract_features(path: str | Path) -> dict[str, float]:
    path = resolve_path(path)
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(path)

    file_size = float(path.stat().st_size)
    h0, w0 = image_bgr.shape[:2]
    image_bgr = resize_longest_side(image_bgr, max_side=256)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    rgb = image_rgb.astype(np.float32)
    means = rgb.reshape(-1, 3).mean(axis=0)
    stds = rgb.reshape(-1, 3).std(axis=0)
    rg = np.abs(rgb[:, :, 0] - rgb[:, :, 1])
    yb = np.abs(0.5 * (rgb[:, :, 0] + rgb[:, :, 1]) - rgb[:, :, 2])
    colorfulness = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    edges = cv2.Canny(gray, 100, 200)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    residual = gray.astype(np.float32) - blur.astype(np.float32)
    low_freq, high_freq, hf_ratio = high_frequency_ratio(gray)

    return {
        "orig_width": float(w0),
        "orig_height": float(h0),
        "orig_aspect_ratio": float(w0 / max(h0, 1)),
        "file_size": file_size,
        "file_size_per_pixel": file_size / max(w0 * h0, 1),
        "mean_r": float(means[0]),
        "mean_g": float(means[1]),
        "mean_b": float(means[2]),
        "std_r": float(stds[0]),
        "std_g": float(stds[1]),
        "std_b": float(stds[2]),
        "brightness_mean": float(gray.mean()),
        "brightness_std": float(gray.std()),
        "saturation_mean": float(hsv[:, :, 1].mean()),
        "saturation_std": float(hsv[:, :, 1].std()),
        "colorfulness": colorfulness,
        "entropy": safe_entropy(gray),
        "laplacian_var": float(lap.var()),
        "edge_density": float((edges > 0).mean()),
        "noise_estimate": float(residual.std()),
        "low_freq_energy": low_freq,
        "high_freq_energy": high_freq,
        "high_low_freq_ratio": hf_ratio,
    }


def main():
    args = parse_args()
    df = load_split_df(args.split_dir)
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="handcrafted"):
        features = extract_features(row["path"])
        rows.append(
            {
                "path": row["path"],
                "label": row["label"],
                "label_num": LABEL_MAP[str(row["label"]).lower()],
                "split": row["split"],
                "group_id": row["group_id"],
                **features,
            }
        )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved handcrafted features: {out_csv}")


if __name__ == "__main__":
    main()
