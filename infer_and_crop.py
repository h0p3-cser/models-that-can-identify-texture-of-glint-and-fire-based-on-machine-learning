from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import cv2
from ultralytics import YOLO

from utils.image_ops import (
    DetectionCandidate,
    clip_box_to_image,
    crop_by_box,
    expand_box,
    resize_with_padding,
    select_primary_detection,
)
from utils.io_utils import ensure_dir, list_images, save_log_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run detection and crop one primary target region per image."
    )
    parser.add_argument("--weights", type=str, required=True, help="Path to trained YOLO weights (*.pt)")
    parser.add_argument("--source", type=str, required=True, help="Image file / directory / glob pattern")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan source directory")
    parser.add_argument("--output-dir", type=str, default="crops", help="Output crop directory")
    parser.add_argument("--log-csv", type=str, default="crops/crop_log.csv", help="CSV log path")

    parser.add_argument("--imgsz", type=int, default=640, help="Inference size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument("--device", type=str, default="auto", help="'auto', 'cpu', '0', '0,1'...")

    parser.add_argument(
        "--expand-ratio",
        type=float,
        default=0.25,
        help="Context expansion ratio for each side of selected box",
    )
    parser.add_argument(
        "--conf-tie-margin",
        type=float,
        default=0.05,
        help="If confidence difference <= margin, area is used as tiebreaker",
    )
    parser.add_argument("--target-size", type=int, default=512, help="Final sample size")
    parser.add_argument(
        "--save-by-class",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save crops into class subfolders (default: enabled)",
    )

    return parser.parse_args()


def resolve_device(device_arg: str):
    if device_arg != "auto":
        return device_arg

    try:
        import torch

        return 0 if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def model_class_names(model: YOLO) -> Dict[int, str]:
    names = getattr(model.model, "names", None)
    if names is None:
        return {0: "flame", 1: "glint"}

    if isinstance(names, list):
        return {i: str(n) for i, n in enumerate(names)}

    # ultralytics often stores names as dict[int,str]
    return {int(k): str(v) for k, v in names.items()}


def main() -> None:
    args = parse_args()

    weights_path = Path(args.weights).resolve()
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    image_paths = list_images(args.source, recursive=args.recursive)
    if not image_paths:
        raise FileNotFoundError(f"No images found in source: {args.source}")

    output_dir = ensure_dir(args.output_dir)
    log_csv = Path(args.log_csv)

    device = resolve_device(args.device)
    model = YOLO(str(weights_path))
    cls_names = model_class_names(model)

    print("=" * 80)
    print("[Infer+Crop Config]")
    print(f"weights      : {weights_path}")
    print(f"source count : {len(image_paths)}")
    print(f"output_dir   : {output_dir.resolve()}")
    print(f"log_csv      : {log_csv.resolve()}")
    print(f"imgsz        : {args.imgsz}")
    print(f"conf         : {args.conf}")
    print(f"expand_ratio : {args.expand_ratio}")
    print(f"target_size  : {args.target_size}")
    print(f"device       : {device}")
    print("=" * 80)

    rows: List[Dict] = []

    for idx, image_path in enumerate(image_paths, start=1):
        row = {
            "image_name": image_path.name,
            "image_path": str(image_path.resolve()),
            "detected_class_id": "",
            "detected_class_name": "",
            "confidence": "",
            "orig_x1": "",
            "orig_y1": "",
            "orig_x2": "",
            "orig_y2": "",
            "expand_x1": "",
            "expand_y1": "",
            "expand_x2": "",
            "expand_y2": "",
            "crop_success": False,
            "output_file": "",
            "fail_reason": "",
        }

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            row["fail_reason"] = "imread_failed"
            rows.append(row)
            continue

        h, w = image.shape[:2]

        try:
            result = model.predict(
                source=image,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=device,
                verbose=False,
            )[0]
        except Exception as exc:
            row["fail_reason"] = f"predict_error:{exc}"
            rows.append(row)
            continue

        if result.boxes is None or len(result.boxes) == 0:
            row["fail_reason"] = "no_detection"
            rows.append(row)
            continue

        xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)

        candidates: List[DetectionCandidate] = []
        for b, c, cls_id in zip(xyxy, confs, classes):
            clipped = clip_box_to_image(b, image_width=w, image_height=h)
            candidates.append(
                DetectionCandidate(
                    class_id=int(cls_id),
                    confidence=float(c),
                    box_xyxy=clipped,
                )
            )

        primary = select_primary_detection(candidates, conf_tie_margin=args.conf_tie_margin)
        if primary is None:
            row["fail_reason"] = "no_primary_box"
            rows.append(row)
            continue

        expanded = expand_box(
            primary.box_xyxy,
            image_width=w,
            image_height=h,
            expand_ratio=args.expand_ratio,
        )

        crop = crop_by_box(image, expanded)
        if crop.size == 0:
            row["fail_reason"] = "empty_crop"
            rows.append(row)
            continue

        try:
            sample_512 = resize_with_padding(crop, target_size=args.target_size, pad_color=(0, 0, 0))
        except Exception as exc:
            row["fail_reason"] = f"resize_pad_error:{exc}"
            rows.append(row)
            continue

        cls_id = primary.class_id
        cls_name = cls_names.get(cls_id, f"class_{cls_id}")

        if args.save_by_class:
            save_dir = ensure_dir(output_dir / cls_name)
        else:
            save_dir = output_dir

        out_name = f"{image_path.stem}_cls{cls_id}_{idx:06d}.jpg"
        out_path = save_dir / out_name
        ok = cv2.imwrite(str(out_path), sample_512)

        if not ok:
            row["fail_reason"] = "imwrite_failed"
            rows.append(row)
            continue

        x1, y1, x2, y2 = primary.box_xyxy
        ex1, ey1, ex2, ey2 = expanded
        row.update(
            {
                "detected_class_id": cls_id,
                "detected_class_name": cls_name,
                "confidence": round(primary.confidence, 6),
                "orig_x1": x1,
                "orig_y1": y1,
                "orig_x2": x2,
                "orig_y2": y2,
                "expand_x1": ex1,
                "expand_y1": ey1,
                "expand_x2": ex2,
                "expand_y2": ey2,
                "crop_success": True,
                "output_file": str(out_path.resolve()),
                "fail_reason": "",
            }
        )
        rows.append(row)

    csv_path = save_log_csv(rows, log_csv)
    success_count = sum(1 for r in rows if r["crop_success"])

    print("\n[Infer+Crop Done]")
    print(f"total   : {len(rows)}")
    print(f"success : {success_count}")
    print(f"failed  : {len(rows) - success_count}")
    print(f"csv     : {csv_path.resolve()}")


if __name__ == "__main__":
    main()
