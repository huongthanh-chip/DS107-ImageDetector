import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm

from data_loader import DEFAULT_SPLIT_DIR, LABEL_MAP, resolve_path
from extract_handcrafted_features import DEFAULT_OUT as DEFAULT_HANDCRAFTED_CSV
from model import DEFAULT_MODEL_NAME, build_model, load_weights, model_output_name


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURE_DIR = PROJECT_ROOT / "01-data" / "features"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "04-reports" / "hybrid"


def parse_args():
    parser = argparse.ArgumentParser(description="Train handcrafted + CNN embedding hybrid classifiers")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--checkpoint", default=None, help="Trained CNN checkpoint used as embedding extractor")
    parser.add_argument("--handcrafted-csv", default=str(DEFAULT_HANDCRAFTED_CSV))
    parser.add_argument("--split-dir", default=str(DEFAULT_SPLIT_DIR))
    parser.add_argument("--feature-dir", default=str(DEFAULT_FEATURE_DIR))
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--mode", choices=["handcrafted", "hybrid"], default="hybrid")
    return parser.parse_args()


class ImagePathDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(resolve_path(row["path"])).convert("RGB")
        return self.transform(image), idx


def load_all_splits(split_dir: str | Path) -> pd.DataFrame:
    split_dir = Path(split_dir)
    frames = []
    for split in ["train", "val", "test"]:
        df = pd.read_csv(split_dir / f"{split}.csv")
        df["split"] = split
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["label_num"] = out["label"].str.lower().map(LABEL_MAP)
    return out


def build_feature_extractor(model_name: str, checkpoint: str | Path, device: str):
    model = build_model(model_name=model_name, pretrained=False, num_classes=2)
    load_weights(model, checkpoint, device)
    model.reset_classifier(0)
    model = model.to(device)
    model.eval()
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config, is_training=False)
    return model, transform


@torch.no_grad()
def extract_embeddings(df: pd.DataFrame, model_name: str, checkpoint: str | Path, device: str, batch_size: int, num_workers: int) -> np.ndarray:
    model, transform = build_feature_extractor(model_name, checkpoint, device)
    dataset = ImagePathDataset(df, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    embeddings = np.zeros((len(df), model.num_features), dtype=np.float32)

    for images, indices in tqdm(loader, desc=f"embeddings:{model_output_name(model_name)}"):
        images = images.to(device)
        feats = model(images).detach().cpu().numpy()
        embeddings[indices.numpy()] = feats
    return embeddings


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    ignore = {"path", "label", "label_num", "split", "group_id"}
    return [c for c in df.columns if c not in ignore and pd.api.types.is_numeric_dtype(df[c])]


def evaluate_classifier(name: str, clf, x_train, y_train, x_test, y_test, out_dir: Path):
    clf.fit(x_train, y_train)
    preds = clf.predict(x_test)
    report = classification_report(y_test, preds, target_names=["real", "fake"], output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, preds)
    pd.DataFrame(report).T.to_csv(out_dir / f"{name}_metrics.csv")
    pd.DataFrame(cm, index=["true_real", "true_fake"], columns=["pred_real", "pred_fake"]).to_csv(out_dir / f"{name}_confusion_matrix.csv")
    print(f"\n{name}")
    print(classification_report(y_test, preds, target_names=["real", "fake"], zero_division=0))
    return report


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hand_df = pd.read_csv(args.handcrafted_csv)
    split_df = load_all_splits(args.split_dir)[["path", "label", "label_num", "split", "group_id"]]
    df = split_df.merge(hand_df.drop(columns=["label", "label_num", "split", "group_id"], errors="ignore"), on="path", how="left")
    if df.isna().any().any():
        missing = df.columns[df.isna().any()].tolist()
        raise ValueError(f"Missing handcrafted features in columns: {missing}")

    feature_cols = get_feature_columns(df)
    x_hand = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["label_num"].to_numpy(dtype=np.int64)
    train_mask = df["split"].eq("train").to_numpy()
    test_mask = df["split"].eq("test").to_numpy()

    features = {"feature_columns": feature_cols, "mode": args.mode}

    if args.mode == "hybrid":
        checkpoint = args.checkpoint or PROJECT_ROOT / "04-reports" / "runs" / model_output_name(args.model_name) / "best.pt"
        if not Path(checkpoint).exists():
            raise FileNotFoundError(f"Hybrid mode needs a trained checkpoint: {checkpoint}")
        emb_path = Path(args.feature_dir) / f"cnn_embeddings_{model_output_name(args.model_name)}.npy"
        if emb_path.exists():
            embeddings = np.load(emb_path)
        else:
            embeddings = extract_embeddings(df, args.model_name, checkpoint, device, args.batch_size, args.num_workers)
            emb_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(emb_path, embeddings)
        x = np.concatenate([x_hand, embeddings], axis=1)
        features["checkpoint"] = str(checkpoint)
        features["embedding_dim"] = int(embeddings.shape[1])
    else:
        x = x_hand

    with (out_dir / "feature_config.json").open("w", encoding="utf-8") as f:
        json.dump(features, f, indent=2)

    classifiers = {
        "logreg": Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ]
        ),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced", n_jobs=1),
    }

    for name, clf in classifiers.items():
        evaluate_classifier(name, clf, x[train_mask], y[train_mask], x[test_mask], y[test_mask], out_dir)


if __name__ == "__main__":
    main()
