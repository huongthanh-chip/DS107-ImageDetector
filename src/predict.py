import os
import argparse
import torch
import pandas as pd
import numpy as np
import shutil
from pathlib import Path
from timm import create_model
from data_loader import get_dataloaders
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


def parse_args():
    parser = argparse.ArgumentParser(description="Predict on test set and save results")
    parser.add_argument('--model-path', type=str, default='best_swin_tiny.pth', help='Path to model state_dict')
    parser.add_argument('--test-csv', type=str, default='data/splits/test.csv', help='Test CSV path')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--device', type=str, default=None, help='Device to run on (cuda or cpu)')
    parser.add_argument('--out-csv', type=str, default='reports/predictions.csv', help='Output CSV for predictions')
    parser.add_argument('--save-misclassified-dir', type=str, default='reports/misclassified',
                        help='Directory to save misclassified images (optional)')
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare dataloader (we only need the test loader)
    _, _, test_loader = get_dataloaders('data/splits/train.csv', 'data/splits/val.csv', args.test_csv,
                                        batch_size=args.batch_size)
    dataset = test_loader.dataset

    # Build model and load weights
    model = create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=2)
    model = model.to(device)

    state = torch.load(args.model_path, map_location=device)
    # state may be a state_dict or a full checkpoint
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    model.load_state_dict(state)
    model.eval()

    all_preds = []
    all_probs = []
    all_labels = []

    softmax = torch.nn.Softmax(dim=1)
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = softmax(outputs)
            # probability of class 1 (fake)
            prob_fake = probs[:, 1].cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().numpy()

            all_probs.extend(prob_fake.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    # Prepare dataframe: use dataset.df to get paths and original labels
    df = dataset.df.copy()
    # Map textual labels to numeric using dataset.label_map if available
    if hasattr(dataset, 'label_map'):
        df['true_label_num'] = df['label'].map(dataset.label_map)
    else:
        df['true_label_num'] = all_labels

    df = df.reset_index(drop=True)
    df['pred_label'] = all_preds
    df['pred_prob_fake'] = all_probs

    # Ensure output directory exists
    out_dir = os.path.dirname(args.out_csv)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    df.to_csv(args.out_csv, index=False)

    # Save misclassified images if requested and if true labels are available
    save_dir = args.save_misclassified_dir
    if save_dir and 'true_label_num' in df.columns:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        mis_df = df[df['true_label_num'].astype(int) != df['pred_label'].astype(int)]
        for _, row in mis_df.iterrows():
            src = row.get('path')
            if not src or not os.path.exists(src):
                continue
            base = os.path.basename(src)
            dest_name = f"true{int(row['true_label_num'])}_pred{int(row['pred_label'])}_{base}"
            dest_path = os.path.join(save_dir, dest_name)
            try:
                shutil.copy(src, dest_path)
            except Exception:
                # fallback: ignore failures copying individual files
                continue
        print(f"Saved {len(mis_df)} misclassified images to: {save_dir}")

    # Compute metrics
    y_true = df['true_label_num'].astype(int).values
    y_pred = df['pred_label'].astype(int).values

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print(f"Saved predictions to: {args.out_csv}")
    print(f"Samples: {len(df)} | Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['real','fake'], zero_division=0))


if __name__ == '__main__':
    main()
