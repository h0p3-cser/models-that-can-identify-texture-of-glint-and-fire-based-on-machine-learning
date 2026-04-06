from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import yaml
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLOv8 detector for flame/glint region localization")
    parser.add_argument("--data", type=str, default="data.yaml", help="Path to YOLO data.yaml")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Pretrained model checkpoint")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers")
    parser.add_argument("--patience", type=int, default=20, help="Early-stop patience")
    parser.add_argument("--device", type=str, default="auto", help="'auto', 'cpu', '0', '0,1'...")
    parser.add_argument("--project", type=str, default="runs/detect", help="Output project directory")
    parser.add_argument("--name", type=str, default="flame_glint_yolov8n", help="Run name")
    parser.add_argument("--exist-ok", action="store_true", help="Allow overwrite run directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--hsv-h", type=float, default=0.015, help="HSV-H augmentation")
    parser.add_argument("--hsv-s", type=float, default=0.7, help="HSV-S augmentation")
    parser.add_argument("--hsv-v", type=float, default=0.4, help="HSV-V augmentation (brightness)")
    return parser.parse_args()


def resolve_device(device_arg: str):
    if device_arg != "auto":
        return device_arg

    try:
        import torch

        return 0 if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def parse_class_names(names_field) -> List[str]:
    if isinstance(names_field, list):
        return [str(x) for x in names_field]

    if isinstance(names_field, dict):
        # Support both int keys and str keys.
        pairs: List[Tuple[int, str]] = []
        for k, v in names_field.items():
            pairs.append((int(k), str(v)))
        pairs.sort(key=lambda x: x[0])
        return [v for _, v in pairs]

    raise ValueError("`names` in data.yaml must be a list or dict.")


def validate_data_yaml(data_yaml: Path) -> Dict:
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    with data_yaml.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    required_keys = ["train", "val", "names"]
    for key in required_keys:
        if key not in cfg:
            raise ValueError(f"Missing key `{key}` in {data_yaml}")

    names = parse_class_names(cfg["names"])
    if "nc" in cfg and int(cfg["nc"]) != len(names):
        raise ValueError(
            f"Mismatch between nc ({cfg['nc']}) and names length ({len(names)})."
        )

    cfg["_class_names"] = names
    return cfg


def main() -> None:
    args = parse_args()
    data_yaml = Path(args.data).resolve()

    data_cfg = validate_data_yaml(data_yaml)
    class_names = data_cfg["_class_names"]

    device = resolve_device(args.device)

    print("=" * 80)
    print("[Train Config]")
    print(f"data      : {data_yaml}")
    print(f"model     : {args.model}")
    print(f"classes   : {class_names}")
    print(f"imgsz     : {args.imgsz}")
    print(f"epochs    : {args.epochs}")
    print(f"batch     : {args.batch}")
    print(f"device    : {device}")
    print(f"project   : {args.project}")
    print(f"name      : {args.name}")
    print(f"hsv_h/s/v : {args.hsv_h} / {args.hsv_s} / {args.hsv_v}")
    print("=" * 80)

    model = YOLO(args.model)

    model.train(
        data=str(data_yaml),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        workers=args.workers,
        patience=args.patience,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        device=device,
        seed=args.seed,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
    )

    save_dir = Path(model.trainer.save_dir)
    best_pt = save_dir / "weights" / "best.pt"
    last_pt = save_dir / "weights" / "last.pt"

    print("\n[Train Done]")
    print(f"save_dir  : {save_dir}")
    print(f"best.pt   : {best_pt if best_pt.exists() else 'NOT_FOUND'}")
    print(f"last.pt   : {last_pt if last_pt.exists() else 'NOT_FOUND'}")


if __name__ == "__main__":
    main()
