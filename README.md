### DS107 Image Detector

Đề tài xây dựng pipeline phát hiện ảnh thật và ảnh do AI tạo sinh, tập trung vào bộ dữ liệu ảnh Việt Nam. Dự án xử lý dữ liệu theo cặp ảnh thật/ảnh AI, chia tập không rò rỉ theo `group_id`, huấn luyện các mô hình CNN pretrained và đánh giá kết quả bằng các file metric/prediction có thể kiểm tra lại.

## Mục Tiêu Đề Tài

- Phân loại ảnh thành 2 lớp:
  - `real`: ảnh thật.
  - `fake`: ảnh AI-generated.
- Chuẩn hóa dữ liệu ảnh đầu vào về cùng định dạng `PNG`, `RGB`.
- Ghép cặp ảnh thật và ảnh AI theo `group_id` để tránh data leakage.
- Huấn luyện và so sánh 3 backbone CNN: `MobileNetV3`, `EfficientNet-B0`, `DenseNet121`.
- Trích xuất thêm handcrafted features và CNN embeddings để thử nghiệm mô hình hybrid như Logistic Regression và Random Forest.
- Lưu đầy đủ báo cáo, checkpoint, prediction, metric và ảnh phân loại sai để phục vụ phân tích.

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

## Giải Thích Chi Tiết Đường Dẫn

### `01-data/`

Thư mục chứa toàn bộ dữ liệu đầu vào, dữ liệu sau xử lý, file chia tập và feature trung gian.

| Đường dẫn | Vai trò |
| --- | --- |
| `01-data/raw/` | Chứa dữ liệu ảnh gốc trước khi cleaning. Thư mục này bị ignore trong Git vì dung lượng lớn. |
| `01-data/raw/Real Image/` | Chứa ảnh thật ban đầu. Label được suy ra là `real`. |
| `01-data/raw/AI Image (SDXL)/` | Chứa ảnh AI-generated ban đầu. Label được suy ra là `fake`. |
| `01-data/cleaned/` | Chứa ảnh đã chuẩn hóa sang `PNG`, `RGB`, đã kiểm tra lỗi, kích thước, ảnh rỗng và cặp real/fake. Thư mục này cũng bị ignore trong Git. |
| `01-data/cleaned/real/` | Ảnh thật sau cleaning, lưu theo tên `<group_id>.png`. |
| `01-data/cleaned/fake/` | Ảnh AI sau cleaning, lưu theo tên `<group_id>.png`. |
| `01-data/splits/train.csv` | Danh sách ảnh dùng để train. Có các cột như `path`, `label`, `width`, `height`, `aspect_ratio`, `group_id`, `split`. |
| `01-data/splits/val.csv` | Danh sách ảnh dùng để validation trong quá trình huấn luyện. |
| `01-data/splits/test.csv` | Danh sách ảnh dùng để đánh giá cuối cùng. |
| `01-data/reports/` | Báo cáo sinh ra trong bước chuẩn bị dữ liệu như manifest, duplicate report, integrity report và biểu đồ độ phân giải. |
| `01-data/reports/duplicates_report.json` | Ghi lại các nhóm ảnh trùng exact duplicate theo MD5 và số file đã xóa. |
| `01-data/reports/resolution_analysis.png` | Biểu đồ phân tích width, height và aspect ratio của ảnh thật/ảnh AI. |
| `01-data/features/` | Chứa feature trung gian dùng cho mô hình hybrid. |
| `01-data/features/handcrafted_features.csv` | Các feature thủ công như kích thước, màu sắc, entropy, edge density, noise estimate, tần số cao/thấp. |
| `01-data/features/cnn_embeddings_densenet121.npy` | Embedding CNN được trích từ checkpoint DenseNet121 để ghép với handcrafted features. |

### `02-notebooks/`

Thư mục chứa notebook thử nghiệm và huấn luyện theo từng backbone.

| Đường dẫn | Vai trò |
| --- | --- |
| `02-notebooks/mobilenetv3-ds107.ipynb` | Notebook thử nghiệm mô hình MobileNetV3. |
| `02-notebooks/efficientnet-b0-ds107.ipynb` | Notebook thử nghiệm mô hình EfficientNet-B0. |
| `02-notebooks/densenet121-ds107.ipynb` | Notebook thử nghiệm mô hình DenseNet121. |

### `03-src/`

Thư mục mã nguồn chính của pipeline. Các script trong đây có thể chạy độc lập hoặc chạy thông qua `run_pipeline.py`.

| Đường dẫn | Vai trò |
| --- | --- |
| `03-src/prepare_data.py` | Chuẩn hóa ảnh, kiểm tra integrity, xóa cặp lỗi, loại duplicate, sinh manifest, phân tích resolution và tạo `train/val/test` split theo `group_id`. |
| `03-src/data_loader.py` | Định nghĩa `RealFakeDataset`, transform train/eval bằng Albumentations, DataLoader và mapping label `real -> 0`, `fake -> 1`. |
| `03-src/model.py` | Registry và hàm build model cho `mobilenetv3`, `efficientnet_b0`, `densenet121`; hỗ trợ lưu/tải checkpoint. |
| `03-src/train.py` | Huấn luyện classifier bằng PyTorch, dùng `SparseCategoricalFocalLoss`, Adam, ReduceLROnPlateau, early stopping và lưu `latest.pt`, `best.pt`, `history.csv`. |
| `03-src/predict.py` | Load checkpoint, dự đoán trên test set, lưu prediction CSV, metric CSV, confusion matrix và ảnh dự đoán sai. |
| `03-src/run_pipeline.py` | Script điều phối toàn bộ pipeline: prepare data, train nhiều model, evaluate nhiều model. |
| `03-src/extract_handcrafted_features.py` | Trích xuất handcrafted features từ toàn bộ split và lưu vào `01-data/features/handcrafted_features.csv`. |
| `03-src/train_hybrid_features.py` | Train mô hình hybrid bằng handcrafted features hoặc handcrafted features + CNN embeddings. Kết quả lưu ở `04-reports/hybrid/`. |
| `03-src/error.py` | Hỗ trợ visualize ảnh bị phân loại sai từ file predictions. |

### `04-reports/`

Thư mục chứa kết quả huấn luyện, đánh giá và phân tích lỗi.

| Đường dẫn | Vai trò |
| --- | --- |
| `04-reports/runs/` | Chứa checkpoint và log huấn luyện theo từng model. Thư mục này bị ignore trong Git vì file checkpoint lớn. |
| `04-reports/runs/mobilenetv3/` | Kết quả train của MobileNetV3, thường có `best.pt`, `latest.pt`, `history.csv`. |
| `04-reports/runs/efficientnet_b0/` | Kết quả train của EfficientNet-B0. |
| `04-reports/runs/densenet121/` | Kết quả train của DenseNet121. |
| `04-reports/predictions_mobilenetv3.csv` | Prediction theo từng ảnh của MobileNetV3, gồm nhãn thật, nhãn dự đoán và xác suất fake. |
| `04-reports/predictions_efficientnet_b0.csv` | Prediction theo từng ảnh của EfficientNet-B0. |
| `04-reports/predictions_densenet121.csv` | Prediction theo từng ảnh của DenseNet121. |
| `04-reports/metrics_mobilenetv3.csv` | Precision, recall, F1-score, support và accuracy của MobileNetV3. |
| `04-reports/metrics_efficientnet_b0.csv` | Metric đánh giá của EfficientNet-B0. |
| `04-reports/metrics_densenet121.csv` | Metric đánh giá của DenseNet121. |
| `04-reports/misclassified/` | Ảnh phân loại sai khi chạy predict theo output mặc định. |
| `04-reports/misclassified_mobilenetv3/` | Ảnh phân loại sai của MobileNetV3. |
| `04-reports/misclassified_efficientnet_b0/` | Ảnh phân loại sai của EfficientNet-B0. |
| `04-reports/misclassified_densenet121/` | Ảnh phân loại sai của DenseNet121. |
| `04-reports/hybrid/` | Kết quả mô hình hybrid/baseline từ handcrafted features và CNN embeddings. |
| `04-reports/hybrid/logreg_metrics.csv` | Metric của Logistic Regression. |
| `04-reports/hybrid/logreg_confusion_matrix.csv` | Confusion matrix của Logistic Regression. |
| `04-reports/hybrid/random_forest_metrics.csv` | Metric của Random Forest. |
| `04-reports/hybrid/random_forest_confusion_matrix.csv` | Confusion matrix của Random Forest. |
| `04-reports/hybrid/feature_config.json` | Ghi lại danh sách feature, mode chạy và checkpoint embedding nếu dùng hybrid. |
| `04-reports/resolution_analysis.png` | Bản sao/phiên bản báo cáo resolution dùng cho phần báo cáo tổng hợp. |
| `04-reports/duplicates_report.json` | Bản sao/phiên bản báo cáo duplicate dùng cho phần báo cáo tổng hợp. |

### `05-docs/`

Thư mục tài liệu mô tả quy trình và ghi chú kỹ thuật.

| Đường dẫn | Vai trò |
| --- | --- |
| `05-docs/cleaning.md` | Đặc tả quy trình cleaning dữ liệu: chuẩn hóa format, kiểm tra integrity, deduplication, phân tích resolution và group-aware split. |
| `05-docs/legacy/` | Lưu tài liệu hoặc file cũ nếu cần đối chiếu trong quá trình phát triển. |

### File cấu hình ở thư mục gốc

| Đường dẫn | Vai trò |
| --- | --- |
| `.gitignore` | Bỏ qua dữ liệu lớn, checkpoint, ảnh phân loại sai, cache Python và cấu hình local. |
| `.gitattributes` | Cấu hình Git LFS cho file model lớn như `best_swin_tiny.pth`. |
| `README.md` | Tài liệu giới thiệu dự án, cấu trúc thư mục và hướng dẫn chạy. |

## Luồng Xử Lý Chính

```text
01-data/raw/Real Image + 01-data/raw/AI Image (SDXL)
-> 03-src/prepare_data.py
-> 01-data/cleaned/{real,fake}
-> 01-data/splits/{train,val,test}.csv
-> 03-src/train.py
-> 04-reports/runs/<model>/
-> 03-src/predict.py
-> 04-reports/predictions_<model>.csv + metrics_<model>.csv + misclassified_<model>/
```

Với nhánh hybrid:

```text
01-data/splits/*.csv
-> 03-src/extract_handcrafted_features.py
-> 01-data/features/handcrafted_features.csv
-> 03-src/train_hybrid_features.py
-> 04-reports/hybrid/
```

## Cài Đặt Phụ Thuộc

```powershell
pip install torch torchvision timm albumentations opencv-python pandas scikit-learn matplotlib pillow tqdm
```

Nếu dùng GPU, cần cài bản `torch` phù hợp với CUDA trên máy.

## Kiểm Tra Split Hiện Có

Lệnh này không sửa dữ liệu, chỉ kiểm tra đường dẫn ảnh và kiểm tra leakage theo `group_id`.

```powershell
python 03-src/run_pipeline.py --validate-only
```

## Chạy Pipeline Đầy Đủ

Mặc định pipeline train và evaluate cả 3 model: `mobilenetv3`, `efficientnet_b0`, `densenet121`.

```powershell
python 03-src/run_pipeline.py --skip-prepare --epochs 50 --batch-size 32 --num-workers 4
```

Nếu cần build lại `01-data/cleaned/` từ dữ liệu raw:

```powershell
python 03-src/run_pipeline.py --reset-cleaned --epochs 50 --batch-size 32
```

Chỉ chạy một số model:

```powershell
python 03-src/run_pipeline.py --skip-prepare --models mobilenetv3 densenet121 --epochs 50 --batch-size 32
```

## Chạy Từng Bước

Chuẩn bị hoặc kiểm tra dữ liệu:

```powershell
python 03-src/prepare_data.py --validate-only
python 03-src/prepare_data.py --reset-cleaned
```

Huấn luyện từng model:

```powershell
python 03-src/train.py --model-name mobilenetv3 --epochs 50 --batch-size 32
python 03-src/train.py --model-name efficientnet_b0 --epochs 50 --batch-size 32
python 03-src/train.py --model-name densenet121 --epochs 50 --batch-size 32
```

Đánh giá checkpoint:

```powershell
python 03-src/predict.py --model-name mobilenetv3 --model-path 04-reports/runs/mobilenetv3/best.pt
```

`03-src/train.py` dùng `SparseCategoricalFocalLoss` mặc định với `gamma=2.0`. Nếu muốn thêm trọng số lớp theo tần suất trong train split:

```powershell
python 03-src/train.py --model-name mobilenetv3 --class-balanced-alpha
```

## Feature Engineering Và Hybrid Model

Trích handcrafted image features:

```powershell
python 03-src/extract_handcrafted_features.py
```

Train baseline chỉ dùng handcrafted features:

```powershell
python 03-src/train_hybrid_features.py --mode handcrafted
```

Train hybrid classifier bằng handcrafted features và CNN embeddings từ checkpoint đã train:

```powershell
python 03-src/train_hybrid_features.py --mode hybrid --model-name densenet121 --checkpoint 04-reports/runs/densenet121/best.pt
```

## Đầu Ra Chính

- `01-data/reports/duplicates_report.json`
- `01-data/reports/integrity_report.json`
- `01-data/reports/manifest.csv`
- `01-data/reports/resolution_analysis.png`
- `01-data/splits/train.csv`
- `01-data/splits/val.csv`
- `01-data/splits/test.csv`
- `04-reports/runs/<model>/best.pt`
- `04-reports/runs/<model>/latest.pt`
- `04-reports/runs/<model>/history.csv`
- `04-reports/predictions_<model>.csv`
- `04-reports/metrics_<model>.csv`
- `04-reports/misclassified_<model>/`
- `01-data/features/handcrafted_features.csv`
- `01-data/features/cnn_embeddings_<model>.npy`
- `04-reports/hybrid/*_metrics.csv`
- `04-reports/hybrid/*_confusion_matrix.csv`

## Quy Ước Label

| Label | Class ID | Ý nghĩa |
| --- | ---: | --- |
| `real` | `0` | Ảnh thật |
| `fake` | `1` | Ảnh AI-generated |

## Ghi Chú Về Dữ Liệu Và Git

- `01-data/raw/` và `01-data/cleaned/` không được commit vì chứa dữ liệu ảnh lớn.
- `04-reports/runs/` và `04-reports/misclassified*/` không được commit vì là output sinh ra sau train/evaluate.
- Các file CSV nhỏ trong `01-data/splits/`, `01-data/features/`, `04-reports/` có thể dùng để tái lập đánh giá hoặc viết báo cáo.
- Khi chia tập, dự án dùng `GroupShuffleSplit` theo `group_id` để một ảnh thật và ảnh AI cùng nguồn không rơi vào nhiều split khác nhau.
