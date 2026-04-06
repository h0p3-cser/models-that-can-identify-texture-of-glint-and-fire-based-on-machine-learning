# Flame/Glint Region Detector + Auto Crop (512x512)

This project focuses on one functional chain only:

Image input -> region detector -> one primary box -> expanded crop -> keep aspect ratio -> pad to 512x512 -> save crop + CSV log.

## 1) Dataset Layout (YOLO format)

```text
dataset/
  images/
    train/
    val/
  labels/
    train/
    val/
```

- One image corresponds to one `.txt` label file with the same stem.
- YOLO label line format:

```text
class_id x_center y_center width height
```

All coordinates are normalized to `[0, 1]`.

Class IDs:
- `0`: flame
- `1`: glint

## 2) Install

```bash
pip install -r requirements.txt
```

## 3) Train

```bash
python train_detector.py --data data.yaml --model yolov8n.pt --imgsz 640 --epochs 50 --batch 16
```

After training, check:
- `runs/detect/<name>/weights/best.pt`

## 4) Infer + Auto Crop

```bash
python infer_and_crop.py \
  --weights runs/detect/flame_glint_yolov8n/weights/best.pt \
  --source dataset/images/val \
  --output-dir crops \
  --log-csv crops/crop_log.csv \
  --save-by-class
```

Default strategy:
- Keep only one primary box per image.
- Selection rule: confidence first; if close (<= 0.05), choose larger area.
- Expand selected box by 25% on each side.
- Clip expanded box to image boundary.
- Crop from original color image.
- Resize with aspect ratio + black padding to 512x512.

## 5) Output

- Cropped samples: `crops/` (or `crops/<class_name>/` with `--save-by-class`)
- Log CSV: `crops/crop_log.csv`

CSV fields include image name/path, class, confidence, original box, expanded box, crop success flag, output file, and fail reason.
