# DS107 Image Detector

Pipeline phan loai anh that va anh AI-generated tren bo du lieu Viet Nam.

## Cau Truc Repo

- `01-data/raw/`: du lieu anh goc.
- `01-data/cleaned/`: anh da chuan hoa RGB/PNG theo hai lop `real` va `fake`.
- `01-data/splits/`: split CSV theo `group_id` gom `train.csv`, `val.csv`, `test.csv`.
- `01-data/reports/`: bao cao EDA/cleaning cua du lieu.
- `02-notebooks/`: notebook train MobileNetV3, EfficientNet-B0, DenseNet121.
- `03-src/`: source code pipeline chinh.
- `04-reports/`: ket qua train/evaluate, predictions, metrics, misclassified samples.
- `05-docs/`: tai lieu mo ta cleaning va file legacy.

## Cai Dat Phu Thuoc

```powershell
pip install torch torchvision timm albumentations opencv-python pandas scikit-learn matplotlib pillow tqdm
```

## Recheck Split Hien Co

Lenh nay khong sua du lieu, chi kiem tra path va leakage theo `group_id`.

```powershell
python 03-src/run_pipeline.py --validate-only
```

## Chay Pipeline Day Du

Mac dinh lenh duoi train/evaluate ca 3 model: `mobilenetv3`, `efficientnet_b0`, `densenet121`.

```powershell
python 03-src/run_pipeline.py --skip-prepare --epochs 50 --batch-size 32 --num-workers 4
```

Neu muon build lai `01-data/cleaned` tu raw:

```powershell
python 03-src/run_pipeline.py --reset-cleaned --epochs 50 --batch-size 32
```

## Chay Tung Buoc

```powershell
python 03-src/prepare_data.py --validate-only
python 03-src/train.py --model-name mobilenetv3 --epochs 50 --batch-size 32
python 03-src/train.py --model-name efficientnet_b0 --epochs 50 --batch-size 32
python 03-src/train.py --model-name densenet121 --epochs 50 --batch-size 32
python 03-src/predict.py --model-name mobilenetv3 --model-path 04-reports/runs/mobilenetv3/best.pt
```

`03-src/train.py` su dung `SparseCategoricalFocalLoss` mac dinh voi `gamma=2.0`.
Neu muon them trong so lop theo tan suat train split:

```powershell
python 03-src/train.py --model-name mobilenetv3 --class-balanced-alpha
```

## Dau Ra Chinh

- `01-data/reports/duplicates_report.json`
- `01-data/reports/resolution_analysis.png`
- `04-reports/runs/<model>/best.pt`
- `04-reports/runs/<model>/history.csv`
- `04-reports/predictions_<model>.csv`
- `04-reports/metrics_<model>.csv`
- `04-reports/misclassified_<model>/`

## Label

- `real` -> class `0`
- `fake` -> class `1`
