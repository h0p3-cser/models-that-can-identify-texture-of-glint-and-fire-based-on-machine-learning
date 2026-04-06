from __future__ import annotations

import glob
from pathlib import Path
from typing import Dict, List, Sequence

import cv2
import numpy as np
import pandas as pd

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def list_images(source: Path | str, recursive: bool = False) -> List[Path]:
    """Collect image files from a file, directory, or glob pattern."""
    src = Path(source)

    if src.is_file() and src.suffix.lower() in IMAGE_SUFFIXES:
        return [src]

    if src.is_dir():
        pattern = "**/*" if recursive else "*"
        files = [p for p in src.glob(pattern) if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
        return sorted(files)

    # Treat as glob pattern (supports absolute patterns on Windows).
    matches = [Path(p) for p in glob.glob(str(source), recursive=recursive)]
    matches = [p for p in matches if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    return sorted(matches)


def read_image(image_path: Path | str) -> np.ndarray | None:
    """Read image robustly for unicode paths on Windows."""
    path = Path(image_path)
    if not path.exists():
        return None

    try:
        buffer = np.fromfile(str(path), dtype=np.uint8)
        if buffer.size == 0:
            return None
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        return image
    except Exception:
        return None


def save_image(image: np.ndarray, output_path: Path | str) -> bool:
    """Write image robustly for unicode paths on Windows."""
    path = Path(output_path)
    ensure_dir(path.parent)

    ext = path.suffix.lower()
    if ext not in IMAGE_SUFFIXES:
        ext = ".jpg"
        path = path.with_suffix(ext)

    try:
        ok, encoded = cv2.imencode(ext, image)
        if not ok:
            return False
        encoded.tofile(str(path))
        return True
    except Exception:
        return False


def save_log_csv(rows: Sequence[Dict], csv_path: Path | str) -> Path:
    path = Path(csv_path)
    ensure_dir(path.parent)

    df = pd.DataFrame(list(rows))
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path
