# DS107 Image Detector

Dự án xây dựng pipeline phân loại ảnh thật và ảnh do AI tạo sinh trên bộ dữ liệu ảnh Việt Nam. Hệ thống xử lý dữ liệu theo cặp `real/fake`, chia tập theo `group_id` để tránh rò rỉ dữ liệu, huấn luyện các mô hình CNN pretrained và lưu kết quả đánh giá để phân tích.

## Mục Tiêu

- Phân loại ảnh thành 2 lớp: `real` và `fake`.
- Chuẩn hóa dữ liệu ảnh về `PNG`, `RGB`.
- Chia tập `train/val/test` không leakage theo `group_id`.
- Huấn luyện và so sánh `MobileNetV3`, `EfficientNet-B0`, `DenseNet121`.
- Lưu metric, prediction, checkpoint và ảnh phân loại sai.
- Thử nghiệm thêm handcrafted features và mô hình hybrid.

## Cấu Trúc Thư Mục

```text
DS107-ImageDetector/
|-- 01-data/
|   |-- raw/
|   |   |-- Real Image/
|   |   `-- AI Image (SDXL)/
|   |-- cleaned/
|   |   |-- real/
|   |   `-- fake/
|   |-- splits/
|   |   |-- train.csv
|   |   |-- val.csv
|   |   `-- test.csv
|   |-- reports/
|   `-- features/
|-- 02-notebooks/
|-- 03-src/
|-- 04-reports/
|   |-- runs/
|   |-- hybrid/
|   `-- misclassified*/
|-- 05-docs/
|-- .gitignore
|-- .gitattributes
`-- README.md
```

## Giới Thiệu Thư Mục

- `01-data/`: chứa dữ liệu của dự án.
- `01-data/raw/`: ảnh gốc ban đầu, gồm ảnh thật trong `Real Image/` và ảnh AI trong `AI Image (SDXL)/`.
- `01-data/cleaned/`: ảnh đã được chuẩn hóa và làm sạch, chia thành `real/` và `fake/`.
- `01-data/splits/`: file CSV chia tập `train.csv`, `val.csv`, `test.csv`.
- `01-data/reports/`: báo cáo trong bước chuẩn bị dữ liệu như duplicate report, integrity report, manifest và biểu đồ resolution.
- `01-data/features/`: handcrafted features và CNN embeddings dùng cho mô hình hybrid.
- `02-notebooks/`: notebook thử nghiệm huấn luyện các mô hình MobileNetV3, EfficientNet-B0 và DenseNet121.
- `03-src/`: mã nguồn chính cho chuẩn bị dữ liệu, train, predict, trích xuất feature và chạy pipeline.
- `04-reports/`: kết quả huấn luyện và đánh giá, gồm checkpoint, metric, prediction, ảnh phân loại sai và kết quả hybrid.
- `04-reports/runs/`: checkpoint và lịch sử train theo từng model.
- `04-reports/hybrid/`: metric và confusion matrix của mô hình hybrid/baseline.
- `05-docs/`: tài liệu mô tả quy trình cleaning và ghi chú liên quan.

## Luồng Xử Lý

```text
01-data/raw/
-> 03-src/prepare_data.py
-> 01-data/cleaned/
-> 01-data/splits/
-> 03-src/train.py
-> 04-reports/runs/
-> 03-src/predict.py
-> 04-reports/
```

Với hướng hybrid:

```text
01-data/splits/
-> 03-src/extract_handcrafted_features.py
-> 01-data/features/
-> 03-src/train_hybrid_features.py
-> 04-reports/hybrid/
```

## Cài Đặt

```powershell
pip install torch torchvision timm albumentations opencv-python pandas scikit-learn matplotlib pillow tqdm
```

Nếu dùng GPU, cần cài bản `torch` phù hợp với CUDA trên máy.

## Chạy Pipeline

Kiểm tra split hiện có:

```powershell
python 03-src/run_pipeline.py --validate-only
```

Train và evaluate 3 model mặc định:

```powershell
python 03-src/run_pipeline.py --skip-prepare --epochs 50 --batch-size 32 --num-workers 4
```

Build lại dữ liệu cleaned từ raw rồi train:

```powershell
python 03-src/run_pipeline.py --reset-cleaned --epochs 50 --batch-size 32
```

## Chạy Từng Bước

Chuẩn bị dữ liệu:

```powershell
python 03-src/prepare_data.py --reset-cleaned
```

Train một model:

```powershell
python 03-src/train.py --model-name mobilenetv3 --epochs 50 --batch-size 32
python 03-src/train.py --model-name efficientnet_b0 --epochs 50 --batch-size 32
python 03-src/train.py --model-name densenet121 --epochs 50 --batch-size 32
```

Đánh giá checkpoint:

```powershell
python 03-src/predict.py --model-name mobilenetv3 --model-path 04-reports/runs/mobilenetv3/best.pt
```

## Hybrid Features

Trích xuất handcrafted features:

```powershell
python 03-src/extract_handcrafted_features.py
```

Train baseline từ handcrafted features:

```powershell
python 03-src/train_hybrid_features.py --mode handcrafted
```

Train hybrid bằng handcrafted features và CNN embeddings:

```powershell
python 03-src/train_hybrid_features.py --mode hybrid --model-name densenet121 --checkpoint 04-reports/runs/densenet121/best.pt
```

## Đầu Ra Chính

- `01-data/splits/train.csv`, `val.csv`, `test.csv`
- `01-data/reports/duplicates_report.json`
- `01-data/reports/resolution_analysis.png`
- `01-data/features/handcrafted_features.csv`
- `04-reports/runs/<model>/best.pt`
- `04-reports/runs/<model>/history.csv`
- `04-reports/predictions_<model>.csv`
- `04-reports/metrics_<model>.csv`
- `04-reports/misclassified_<model>/`
- `04-reports/hybrid/*_metrics.csv`

## Quy Ước Label

| Label | Class ID | Ý nghĩa |
| --- | ---: | --- |
| `real` | `0` | Ảnh thật |
| `fake` | `1` | Ảnh AI-generated |

## Ghi Chú

- `01-data/raw/`, `01-data/cleaned/`, `04-reports/runs/` và `04-reports/misclassified*/` không commit lên Git vì chứa dữ liệu hoặc output lớn.
- Project dùng `GroupShuffleSplit` theo `group_id` để ảnh cùng nguồn không xuất hiện ở nhiều split khác nhau.
