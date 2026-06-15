import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABEL_NAMES = {0: "real", 1: "fake"}


def _annotate(img, text, color=(255, 0, 0)):
    h, w = img.shape[:2]
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 28), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
    cv2.putText(img, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return img


def visualize_errors(csv_path=PROJECT_ROOT / "04-reports" / "predictions.csv", num_per_type=6, save_dir=None, show=True):
    df = pd.read_csv(csv_path)

    if "true_label_num" in df.columns and "pred_label" in df.columns:
        df["true_num"] = df["true_label_num"].astype(int)
        df["pred_num"] = df["pred_label"].astype(int)
    elif "label" in df.columns and "pred_label" in df.columns:
        inv_map = {"real": 0, "fake": 1}
        df["true_num"] = df["label"].map(inv_map)
        df["pred_num"] = df["pred_label"].astype(int)
    else:
        raise ValueError("predictions CSV must contain either true_label_num/pred_label or label/pred_label")

    real_as_fake = df[(df["true_num"] == 0) & (df["pred_num"] == 1)].head(num_per_type)
    fake_as_real = df[(df["true_num"] == 1) & (df["pred_num"] == 0)].head(num_per_type)

    cols = max(1, len(real_as_fake), len(fake_as_real))
    fig, axes = plt.subplots(2, cols, figsize=(4 * cols, 8))
    if axes.ndim == 1:
        axes = axes.reshape(2, -1)

    for row_idx, rows in enumerate([real_as_fake, fake_as_real]):
        for i in range(cols):
            ax = axes[row_idx, i]
            if i < len(rows):
                row = rows.iloc[i]
                p = Path(row["path"])
                if p.exists():
                    img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
                    prob = row.get("pred_prob_fake", None)
                    tname = LABEL_NAMES[int(row["true_num"])]
                    pname = LABEL_NAMES[int(row["pred_num"])]
                    text = f"true: {tname} -> pred: {pname}" + (f" ({prob:.3f})" if pd.notna(prob) else "")
                    ax.imshow(_annotate(img, text))
                else:
                    ax.text(0.5, 0.5, "missing", ha="center")
            ax.axis("off")

    plt.suptitle("Error Analysis: Real->Fake (top) | Fake->Real (bottom)")
    plt.tight_layout()

    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        mis = df[df["true_num"] != df["pred_num"]]
        for _, row in mis.iterrows():
            p = Path(row["path"])
            if not p.exists():
                continue
            img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
            prob = row.get("pred_prob_fake", None)
            tname = LABEL_NAMES.get(int(row["true_num"]), str(row["true_num"]))
            pname = LABEL_NAMES.get(int(row["pred_num"]), str(row["pred_num"]))
            text = f"true:{tname} pred:{pname}" + (f" ({prob:.3f})" if pd.notna(prob) else "")
            img = _annotate(img, text)
            out_path = save_path / f"{p.stem}_true{tname}_pred{pname}{p.suffix}"
            cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    if show:
        plt.show()


def _parse_args_and_run():
    parser = argparse.ArgumentParser(description="Visualize misclassified images from predictions CSV")
    parser.add_argument("--csv", type=str, default=str(PROJECT_ROOT / "04-reports" / "predictions.csv"))
    parser.add_argument("--num", type=int, default=6, help="Number per error type to display")
    parser.add_argument("--save-dir", type=str, default=None, help="Directory to save annotated misclassified images")
    parser.add_argument("--no-show", action="store_true", help="Do not open matplotlib viewer")
    args = parser.parse_args()

    visualize_errors(csv_path=args.csv, num_per_type=args.num, save_dir=args.save_dir, show=not args.no_show)


if __name__ == "__main__":
    _parse_args_and_run()
