from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.crop_utils import crop_and_resize_to_512, draw_preview, select_main_detection
from utils.io_utils import ensure_dir, list_images, read_image, save_image
from utils.log_utils import new_log_row, save_crop_log
from utils.yolo_infer import YoloInfer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process fire/glint datasets into centered square 512x512 samples based on YOLO outputs."
    )
    parser.add_argument("--model", type=str, required=True, help="Path to YOLO model weights (*.pt)")
    parser.add_argument("--fire_dir", type=str, default="测试集fire", help="Input fire folder")
    parser.add_argument("--glint_dir", type=str, default="测试集glint", help="Input glint folder")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output root directory")
    parser.add_argument("--log_path", type=str, default="logs/crop_log.csv", help="CSV log path")

    parser.add_argument("--img_size", type=int, default=512, help="Output image size")
    parser.add_argument(
        "--target_ratio",
        type=float,
        default=0.6,
        help="Target occupancy ratio in square crop (side = max(bw,bh)/target_ratio)",
    )
    parser.add_argument("--conf_thres", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--iou_thres", type=float, default=0.7, help="YOLO IoU threshold")
    parser.add_argument("--device", type=str, default="auto", help="auto/cpu/0/0,1...")

    parser.add_argument(
        "--select_strategy",
        choices=["score", "conf"],
        default="score",
        help="Main target selection strategy",
    )
    parser.add_argument(
        "--save_preview",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save preview image with bbox and square crop",
    )
    parser.add_argument("--recursive", action="store_true", help="Recursively scan input folders")
    parser.add_argument("--max_images", type=int, default=0, help="Limit images per folder, 0 = all")
    return parser.parse_args()


def process_one_folder(
    inferer: YoloInfer,
    source_dir: Path,
    source_name: str,
    output_dir: Path,
    preview_dir: Path | None,
    args: argparse.Namespace,
    rows: List[Dict],
) -> Tuple[int, int]:
    image_paths = list_images(source_dir, recursive=args.recursive)
    if args.max_images > 0:
        image_paths = image_paths[: args.max_images]

    success = 0
    failed = 0

    for image_path in tqdm(image_paths, desc=f"Processing {source_name}", unit="img"):
        row = new_log_row()
        row["source_folder"] = str(source_dir)
        row["source_image"] = image_path.name
        row["class_name"] = source_name

        try:
            image = read_image(image_path)
            if image is None:
                raise RuntimeError("imread_failed")

            detections = inferer.infer(image)
            if not detections:
                raise RuntimeError("no_detection")

            det_dicts = [
                {
                    "class_name": d.class_name,
                    "confidence": d.confidence,
                    "bbox_xyxy": d.bbox_xyxy,
                    "area": d.area,
                    "bbox_source": d.bbox_source,
                }
                for d in detections
            ]
            selected = select_main_detection(det_dicts, strategy=args.select_strategy)
            if selected is None:
                raise RuntimeError("main_detection_not_found")

            bbox = selected["bbox_xyxy"]
            crop_img, crop_meta = crop_and_resize_to_512(
                image=image,
                bbox_xyxy=bbox,
                img_size=args.img_size,
                target_ratio=args.target_ratio,
                pad_color=(0, 0, 0),
            )

            output_name = f"{image_path.stem}_512.jpg"
            output_path = output_dir / output_name
            if not save_image(crop_img, output_path):
                raise RuntimeError("save_output_failed")

            if preview_dir is not None:
                preview_img = draw_preview(
                    image=image,
                    bbox_xyxy=bbox,
                    square_xyxy=(
                        crop_meta.square_x1 - crop_meta.padded_left,
                        crop_meta.square_y1 - crop_meta.padded_top,
                        crop_meta.square_x2 - crop_meta.padded_left,
                        crop_meta.square_y2 - crop_meta.padded_top,
                    ),
                )
                preview_path = preview_dir / f"{source_name}_{image_path.stem}_preview.jpg"
                save_image(preview_img, preview_path)

            row.update(
                {
                    "detected_confidence": round(float(selected["confidence"]), 6),
                    "bbox_x1": round(float(bbox[0]), 2),
                    "bbox_y1": round(float(bbox[1]), 2),
                    "bbox_x2": round(float(bbox[2]), 2),
                    "bbox_y2": round(float(bbox[3]), 2),
                    "crop_side": crop_meta.crop_side,
                    "padded_left": crop_meta.padded_left,
                    "padded_top": crop_meta.padded_top,
                    "padded_right": crop_meta.padded_right,
                    "padded_bottom": crop_meta.padded_bottom,
                    "output_path": str(output_path.resolve()),
                    "status": "success",
                    "error_message": "",
                }
            )
            success += 1

        except Exception as exc:
            row.update(
                {
                    "status": "failed",
                    "error_message": str(exc),
                }
            )
            failed += 1

        rows.append(row)

    return success, failed


def main() -> None:
    args = parse_args()

    model_path = Path(args.model).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    fire_dir = Path(args.fire_dir).resolve()
    glint_dir = Path(args.glint_dir).resolve()

    if not fire_dir.exists():
        raise FileNotFoundError(f"fire_dir not found: {fire_dir}")
    if not glint_dir.exists():
        raise FileNotFoundError(f"glint_dir not found: {glint_dir}")

    output_root = ensure_dir(Path(args.output_dir))
    fire_output = ensure_dir(output_root / "fire_512")
    glint_output = ensure_dir(output_root / "glint_512")
    preview_output = ensure_dir(output_root / "previews") if args.save_preview else None

    inferer = YoloInfer(
        model_path=model_path,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        device=args.device,
    )

    print("=" * 90)
    print("[Process Config]")
    print(f"model          : {model_path}")
    print(f"fire_dir       : {fire_dir}")
    print(f"glint_dir      : {glint_dir}")
    print(f"output_root    : {output_root.resolve()}")
    print(f"log_path       : {Path(args.log_path).resolve()}")
    print(f"img_size       : {args.img_size}")
    print(f"target_ratio   : {args.target_ratio}")
    print(f"conf_thres     : {args.conf_thres}")
    print(f"iou_thres      : {args.iou_thres}")
    print(f"select_strategy: {args.select_strategy}")
    print(f"save_preview   : {args.save_preview}")
    print(f"device         : {inferer.device}")
    print("=" * 90)

    rows: List[Dict] = []

    fire_success, fire_failed = process_one_folder(
        inferer=inferer,
        source_dir=fire_dir,
        source_name="fire",
        output_dir=fire_output,
        preview_dir=preview_output,
        args=args,
        rows=rows,
    )

    glint_success, glint_failed = process_one_folder(
        inferer=inferer,
        source_dir=glint_dir,
        source_name="glint",
        output_dir=glint_output,
        preview_dir=preview_output,
        args=args,
        rows=rows,
    )

    log_path = save_crop_log(rows, args.log_path)

    total_success = fire_success + glint_success
    total_failed = fire_failed + glint_failed

    print("\n[Process Done]")
    print(f"fire  : success={fire_success}, failed={fire_failed}")
    print(f"glint : success={glint_success}, failed={glint_failed}")
    print(f"total : success={total_success}, failed={total_failed}")
    print(f"log   : {log_path.resolve()}")


if __name__ == "__main__":
    main()
