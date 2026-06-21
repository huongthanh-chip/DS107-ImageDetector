import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

from data_loader import DEFAULT_SPLIT_DIR, get_dataloaders
from model import DEFAULT_MODEL_NAME, MODEL_REGISTRY, build_model, model_output_name, save_checkpoint


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Train a real-vs-AI image classifier")
    parser.add_argument("--train-csv", default=str(DEFAULT_SPLIT_DIR / "train.csv"))
    parser.add_argument("--val-csv", default=str(DEFAULT_SPLIT_DIR / "val.csv"))
    parser.add_argument("--test-csv", default=str(DEFAULT_SPLIT_DIR / "test.csv"))
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, choices=sorted(MODEL_REGISTRY))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="cuda, cpu, or leave empty for auto")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--amp", action="store_true", help="Use mixed precision on CUDA")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument(
        "--class-balanced-alpha",
        action="store_true",
        help="Use inverse-frequency class weights as focal alpha",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SparseCategoricalFocalLoss(nn.Module):
    """Focal Loss for integer class labels."""

    def __init__(self, gamma: float = 2.0, alpha: list[float] | torch.Tensor | None = None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(logits, dim=1)
        log_p_t = log_p.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)
        p_t = log_p_t.exp()
        loss = -((1 - p_t) ** self.gamma) * log_p_t

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device).gather(dim=0, index=targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        raise ValueError(f"Unsupported reduction: {self.reduction}")


def compute_class_balanced_alpha(train_csv: str | Path) -> list[float]:
    df = pd.read_csv(train_csv)
    counts = df["label"].str.lower().value_counts()
    real_count = float(counts.get("real", 0))
    fake_count = float(counts.get("fake", 0))
    if real_count == 0 or fake_count == 0:
        raise ValueError(f"Cannot compute alpha from class counts: {counts.to_dict()}")

    total = real_count + fake_count
    real_weight = total / (2.0 * real_count)
    fake_weight = total / (2.0 * fake_count)
    return [real_weight, fake_weight]


def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = 0.0
    batches = 0
    y_true: list[int] = []
    y_pred: list[int] = []

    pbar = tqdm(loader, desc="train", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        preds = outputs.argmax(dim=1)
        total_loss += loss.item()
        batches += 1
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())
        pbar.set_postfix(loss=f"{total_loss / max(1, batches):.4f}")

    return compute_metrics(y_true, y_pred, total_loss / max(1, len(loader)))


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    y_true: list[int] = []
    y_pred: list[int] = []

    for images, labels in tqdm(loader, desc="eval", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        preds = outputs.argmax(dim=1)

        total_loss += loss.item()
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

    return compute_metrics(y_true, y_pred, total_loss / max(1, len(loader)))


def compute_metrics(y_true, y_pred, loss):
    return {
        "loss": float(loss),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "04-reports" / "runs" / model_output_name(args.model_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, _ = get_dataloaders(
        args.train_csv,
        args.val_csv,
        args.test_csv,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
    )

    model = build_model(
        model_name=args.model_name,
        pretrained=not args.no_pretrained,
        num_classes=2,
    ).to(device)

    alpha = compute_class_balanced_alpha(args.train_csv) if args.class_balanced_alpha else None
    criterion = SparseCategoricalFocalLoss(gamma=args.focal_gamma, alpha=alpha)
    print(f"Using SparseCategoricalFocalLoss(gamma={args.focal_gamma}, alpha={alpha})")
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.2,
        patience=3,
        min_lr=1e-7,
    )
    use_cuda = str(device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and use_cuda) if use_cuda else None
    if scaler is not None and not scaler.is_enabled():
        scaler = None

    best_f1 = -1.0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler=scaler)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_metrics["loss"])

        row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train f1={train_metrics['f1']:.4f} acc={train_metrics['accuracy']:.4f} | "
            f"val f1={val_metrics['f1']:.4f} acc={val_metrics['accuracy']:.4f}"
        )

        save_checkpoint(
            output_dir / "latest.pt",
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            model_name=model_output_name(args.model_name),
            img_size=args.img_size,
            val_metrics=val_metrics,
        )
        improved = val_metrics["f1"] > best_f1 + args.early_stopping_min_delta
        if improved:
            best_f1 = val_metrics["f1"]
            epochs_without_improvement = 0
            save_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                model_name=model_output_name(args.model_name),
                img_size=args.img_size,
                val_metrics=val_metrics,
            )
            print(f"Saved best checkpoint: {output_dir / 'best.pt'}")
        else:
            epochs_without_improvement += 1
            print(
                f"val_f1 did not improve from {best_f1:.4f} "
                f"({epochs_without_improvement}/{args.early_stopping_patience})"
            )
            if epochs_without_improvement >= args.early_stopping_patience:
                print("Early stopping triggered")
                break


if __name__ == "__main__":
    main()
