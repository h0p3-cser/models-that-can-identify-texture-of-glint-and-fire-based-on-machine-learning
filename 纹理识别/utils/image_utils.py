from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_rgb_image(image_path: Path | str) -> np.ndarray:
    """Load an image as RGB using a Unicode-safe path reader for Windows.

    We use cv2.imdecode(np.fromfile(...)) instead of cv2.imread so Chinese paths
    in the project root do not break image loading.
    """

    path = Path(image_path)
    raw_bytes = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    image_bgr = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Failed to read image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def build_valid_mask(image_rgb: np.ndarray) -> np.ndarray:
    """Create a mask that removes zero-padding regions from feature extraction.

    The dataset was padded with black pixels, so any pixel whose RGB sum is zero
    is treated as invalid padding and excluded from downstream features.
    """

    return np.sum(image_rgb, axis=2) > 0


def apply_mask(array: np.ndarray, mask: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    """Fill invalid pixels so visualization or intermediate transforms stay stable."""

    output = np.array(array, copy=True)
    output[~mask] = fill_value
    return output


def get_masked_values(array: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return only valid pixels so statistics ignore padded borders completely."""

    return array[mask]


def safe_ratio(numerator: float, denominator: float, eps: float = 1e-6) -> float:
    """Compute a stable ratio without exploding when the denominator is tiny."""

    return float(numerator / (denominator + eps))


def masked_mean(array: np.ndarray, mask: np.ndarray) -> float:
    """Compute the mean only on valid pixels, matching the requested formula."""

    values = get_masked_values(array.astype(np.float32), mask)
    if values.size == 0:
        return 0.0
    return float(values.mean())


def masked_std(array: np.ndarray, mask: np.ndarray) -> float:
    values = get_masked_values(array.astype(np.float32), mask)
    if values.size == 0:
        return 0.0
    return float(values.std())


def masked_max(array: np.ndarray, mask: np.ndarray) -> float:
    values = get_masked_values(array, mask)
    if values.size == 0:
        return 0.0
    return float(values.max())


def masked_percentile(array: np.ndarray, mask: np.ndarray, q: float) -> float:
    values = get_masked_values(array.astype(np.float32), mask)
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, q))


def masked_ratio(condition: np.ndarray, mask: np.ndarray) -> float:
    """Return the fraction of valid pixels that satisfy a condition."""

    valid = mask.astype(bool)
    denom = int(valid.sum())
    if denom == 0:
        return 0.0
    return float(np.logical_and(condition, valid).sum() / denom)


def erode_mask(mask: np.ndarray, kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
    """Shrink the valid region so neighborhood features do not touch padding.

    This is important for LBP, Sobel, Laplacian, and Canny because pixels close
    to the padding boundary can otherwise inherit fake edges from the black fill.
    """

    mask_uint8 = mask.astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    eroded = cv2.erode(mask_uint8, kernel, iterations=iterations)
    return eroded.astype(bool)


def iter_image_paths(folder: Path) -> Iterable[Path]:
    """Yield supported image files in a deterministic order."""

    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            yield path
