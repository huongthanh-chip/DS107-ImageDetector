# Data Cleaning Specification (Real vs AI Vietnamese Images)

## 1. Muc tieu

Xay dung quy trinh cleaning de tao du lieu dau vao on dinh cho EDA va training model phan loai:
- real image
- AI-generated image (duoc tao tu real image)

Quy mo du lieu:
- Real: 2118 images
- AI: 2118 images
- Tong: >4200 images

## 2. Cau truc du lieu dau vao

Nguon raw hien tai:
- Data/raw/Real Image
- Data/raw/AI Image (SDXL)

Luu y quan trong:
- Dataset chua co nhan label truc tiep trong file.
- Nhan duoc suy ra tu thu muc nguon:
  - Thu muc Real Image -> label = real
  - Thu muc AI Image (SDXL) -> label = fake

## 3. Quy tac ghep cap real-fake theo ten file

Ten file AI co format:
- xxxxx-real_image_name.png

Quy tac trich group_id:
- Real image: group_id = stem cua file real
- AI image: group_id = phan sau dau '-' dau tien trong stem

Vi du:
- Real: IMG_0123.png -> group_id = IMG_0123
- AI: 9f3ab-IMG_0123.png -> group_id = IMG_0123

Sau chuan hoa, ca 2 nhanh real/fake deu duoc luu theo ten:
- Data/cleaned/real/<group_id>.png
- Data/cleaned/fake/<group_id>.png

Muc dich:
- Co the kiem tra integrity theo cap 1-1
- Tranh data leakage va de split theo group

## 4. Checklist thuc thi

### Giai doan 1: Chuan hoa Format & Channel
- Toan bo anh -> PNG, RGB, 8-bit
- Strip EXIF/metadata khi save
- Xu ly RGBA, Grayscale, Palette, 16-bit
- Chuan hoa ten file theo group_id

### Giai doan 2: Integrity Check
- PIL verify() + load() de phat hien corrupt/truncated
- Loc anh qua nho (< 224px)
- Loc anh don sac (std < 5.0)
- Neu mot ben loi -> xoa ca cap group
- Ghi log danh sach group/file bi xoa

### Giai doan 3: De-duplication
- MD5 exact duplicate:
  - trong real
  - trong fake
  - cross real <-> fake
- pHash near duplicate:
  - trong real
  - trong fake
  - cross real <-> fake
- Ve histogram Hamming distance de chon threshold
- SSIM xac nhan cap nghi ngo
- Xoa theo nguyen tac: giu 1 dai dien moi nhom trung

### Giai doan 4: Phan tich Resolution
- Ve scatter width-height (real vs fake)
- Ve histogram aspect ratio (real vs fake)
- Bao cao bat dong deu neu co
- Khong resize o giai doan nay

### Giai doan 5: Group-aware Split
- Tao dataframe co: path, label, group_id
- GroupShuffleSplit: 70% train / 15% val / 15% test
- verify_no_leakage(): 1 group khong xuat hien o 2 tap
- Luu Data/splits/train.csv, val.csv, test.csv

### Giai doan 6: Runtime Transform (trong DataLoader)
- Train: RandomCrop + Augmentation + Normalize
- Val/Test: CenterCrop + Normalize (deterministic)
- DataLoader: num_workers + pin_memory

## 5. Luong xu ly tong the

Data/raw/Real Image + Data/raw/AI Image (SDXL)
-> [G1] standardize + map group_id + save Data/cleaned/{real,fake}
-> [G2] integrity check theo cap group_id + remove invalid pair + log
-> [G3] de-dup (MD5/pHash/SSIM)
-> [G4] resolution/aspect analysis (khong resize)
-> [G5] group-aware split -> Data/splits/*.csv
-> [G6] runtime transform trong DataLoader -> training

## 6. Dau ra ky vong

- Data/cleaned/real/*.png
- Data/cleaned/fake/*.png
- Data/reports/* (integrity log, duplicate report, resolution plot)
- Data/splits/train.csv
- Data/splits/val.csv
- Data/splits/test.csv

## 7. Kiem tra nhanh truoc khi train

- So luong group_id real va fake la bang nhau hoac da duoc log sai lech
- Khong con file corrupt/truncated/blank/too-small
- Khong con duplicate nghiem trong (theo threshold da chon)
- train/val/test khong leakage theo group_id
- Duong dan trong CSV ton tai that tren dia
