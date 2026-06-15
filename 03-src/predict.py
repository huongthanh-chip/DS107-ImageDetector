import argparse
import os
import shutil
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from data_loader import DEFAULT_SPLIT_DIR, INV_LABEL_MAP, get_dataloaders
from model import DEFAULT_MODEL_NAME, MODEL_REGISTRY, build_model, load_weights, model_output_name


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint and save per-image predictions")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--test-csv", default=str(DEFAULT_SPLIT_DIR / "test.csv"))
    parser.add_argument("--train-csv", default=str(DEFAULT_SPLIT_DIR / "train.csv"))
    parser.add_argument("--val-csv", default=str(DEFAULT_SPLIT_DIR / "val.csv"))
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, choices=sorted(MODEL_REGISTRY))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out-csv", default=str(PROJECT_ROOT / "04-reports" / "predictions.csv"))
    parser.add_argument("--metrics-csv", default=str(PROJECT_ROOT / "04-reports" / "metrics.csv"))
    parser.add_argument("--save-misclassified-dir", default=str(PROJECT_ROOT / "04-reports" / "misclassified"))
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_path = args.model_path or str(PROJECT_ROOT / "04-reports" / "runs" / model_output_name(args.model_name) / "best.pt")

    _, _, test_loader = get_dataloaders(
        args.train_csv,
        args.val_csv,
        args.test_csv,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
    )
    dataset = test_loader.dataset

    model = build_model(args.model_name, pretrained=False, num_classes=2).to(device)
    metadata = load_weights(model, model_path, device)
    model.eval()

    all_preds: list[int] = []
    all_probs: list[float] = []
    all_labels: list[int] = []
    softmax = torch.nn.Softmax(dim=1)

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="predict"):
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            probs = softmax(outputs)
            all_probs.extend(probs[:, 1].cpu().tolist())
            all_preds.extend(outputs.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    df = dataset.df.copy().reset_index(drop=True)
    df["true_label_num"] = all_labels
    df["true_label_name"] = [INV_LABEL_MAP[x] for x in all_labels]
    df["pred_label"] = all_preds
    df["pred_label_name"] = [INV_LABEL_MAP[x] for x in all_preds]
    df["pred_prob_fake"] = all_probs

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    report = classification_report(
        all_labels,
        all_preds,
        target_names=["real", "fake"],
        output_dict=True,
        zero_division=0,
    )
    metrics = pd.DataFrame(report).T
    metrics_csv = Path(args.metrics_csv)
    metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(metrics_csv)

    save_misclassified(df, args.save_misclassified_dir)

    print(f"Loaded checkpoint: {model_path}")
    if metadata:
        print(f"Checkpoint metadata: {metadata}")
    print(f"Saved predictions: {out_csv}")
    print(f"Saved metrics: {metrics_csv}")
    print("Confusion matrix:")
    print(confusion_matrix(all_labels, all_preds))
    print(classification_report(all_labels, all_preds, target_names=["real", "fake"], zero_division=0))


def save_misclassified(df: pd.DataFrame, save_dir: str | None) -> None:
    if not save_dir:
        return

    out_dir = Path(save_dir)
    if out_dir.exists():
        for old_file in out_dir.iterdir():
            if old_file.is_file():
                old_file.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)

    mis_df = df[df["true_label_num"].astype(int) != df["pred_label"].astype(int)]
    for _, row in mis_df.iterrows():
        src = Path(row["path"])
        if not src.exists():
            continue
        dest_name = (
            f"true{int(row['true_label_num'])}_pred{int(row['pred_label'])}_"
            f"p{float(row['pred_prob_fake']):.3f}_{os.path.basename(src)}"
        )
        try:
            shutil.copy(src, out_dir / dest_name)
        except OSError:
            continue

    print(f"Saved {len(mis_df)} misclassified images: {out_dir}")


if __name__ == "__main__":
    main()
