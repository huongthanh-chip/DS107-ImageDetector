from data_loader import get_dataloaders
import multiprocessing


def main():
    # Khởi tạo các DataLoader từ file CSV đã split
    train_loader, val_loader, test_loader = get_dataloaders(
        train_csv='data/splits/train.csv',
        val_csv='data/splits/val.csv',
        test_csv='data/splits/test.csv',
        batch_size=32,
        num_workers=4
    )

    # Thử lấy một batch để kiểm tra
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}") # Kỳ vọng: [32, 3, 224, 224]
    print(f"Labels: {labels}")            # Kỳ vọng: Tensor chứa 0 và 1


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()