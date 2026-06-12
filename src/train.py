import torch
import torch.nn as nn
import torch.optim as optim
from timm import create_model
from tqdm import tqdm
from data_loader import get_dataloaders # Module G6 của bạn
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# --- CẤU HÌNH (HYPERPARAMETERS) ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
LR = 1e-4             # Transformers thường nhạy cảm với LR, khởi đầu nhỏ
EPOCHS = 5
NUM_CLASSES = 2       # Real (0) và Fake (1)

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix(loss=running_loss/len(loader), acc=100.*correct/total)
    
    return running_loss/len(loader), 100.*correct/total

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(predicted.cpu().numpy().tolist())

    return running_loss/len(loader), 100.*correct/total, y_true, y_pred

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Load Data
    train_loader, val_loader, test_loader = get_dataloaders(
        'data/splits/train.csv', 'data/splits/val.csv', 'data/splits/test.csv',
        batch_size=BATCH_SIZE
    )

    # 2. Khởi tạo Swin Tiny
    # 'swin_tiny_patch4_window7_224' là bản chuẩn cho ảnh 224x224
    model = create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=NUM_CLASSES)
    model = model.to(DEVICE)

    # 3. Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    # Với Transformer, dùng AdamW + Weight Decay là "chuẩn bài"
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
    
    # 4. Scheduler (Giảm LR theo thời gian để hội tụ tốt hơn)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 5. Vòng lặp Training
    best_acc = 0
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_acc, y_true, y_pred = validate(model, val_loader, criterion, DEVICE)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        scheduler.step()
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"Recall: {rec:.4f} | Precision: {prec:.4f} | F1-Score: {f1:.4f}")
        # Lưu model tốt nhất
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_swin_tiny.pth")
            print("=> Saved Best Model")