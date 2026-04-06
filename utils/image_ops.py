from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class DetectionCandidate:
    """Single detection candidate for primary-box selection."""

    class_id: int
    confidence: float
    box_xyxy: Tuple[int, int, int, int]

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.box_xyxy
        return max(0, x2 - x1) * max(0, y2 - y1)


def clip_box_to_image(
    box_xyxy: Sequence[float], image_width: int, image_height: int
) -> Tuple[int, int, int, int]:
    """Clip box coordinates to image boundary and cast to int."""
    x1, y1, x2, y2 = box_xyxy
    x1 = int(np.clip(np.floor(x1), 0, image_width - 1))
    y1 = int(np.clip(np.floor(y1), 0, image_height - 1))
    x2 = int(np.clip(np.ceil(x2), 1, image_width))
    y2 = int(np.clip(np.ceil(y2), 1, image_height))

    if x2 <= x1:
        x2 = min(image_width, x1 + 1)
    if y2 <= y1:
        y2 = min(image_height, y1 + 1)

    return x1, y1, x2, y2


def expand_box(
    box_xyxy: Sequence[int],
    image_width: int,
    image_height: int,
    expand_ratio: float = 0.25,
) -> Tuple[int, int, int, int]:
    """Expand box by a ratio on each side and clip to image boundary."""
    x1, y1, x2, y2 = box_xyxy
    box_w = x2 - x1
    box_h = y2 - y1

    expand_w = int(round(box_w * expand_ratio))
    expand_h = int(round(box_h * expand_ratio))

    ex1 = x1 - expand_w
    ey1 = y1 - expand_h
    ex2 = x2 + expand_w
    ey2 = y2 + expand_h

    return clip_box_to_image((ex1, ey1, ex2, ey2), image_width, image_height)


def crop_by_box(image: np.ndarray, box_xyxy: Sequence[int]) -> np.ndarray:
    """Crop image by xyxy box from original image."""
    x1, y1, x2, y2 = box_xyxy
    return image[y1:y2, x1:x2]


def resize_with_padding(
    image: np.ndarray,
    target_size: int = 512,
    pad_color: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Resize with fixed aspect ratio and pad to square target size."""
    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")

    h, w = image.shape[:2]
    scale = min(target_size / w, target_size / h)

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((target_size, target_size, 3), pad_color, dtype=np.uint8)
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2
    canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized
    return canvas


def select_primary_detection(
    detections: Iterable[DetectionCandidate], conf_tie_margin: float = 0.05
) -> Optional[DetectionCandidate]:
    """
    Select one main box: confidence first; if close, use larger area.

    Args:
        detections: list of candidates.
        conf_tie_margin: boxes within this confidence range from max confidence
            are considered a tie and area is used as tiebreaker.
    """
    det_list: List[DetectionCandidate] = list(detections)
    if not det_list:
        return None

    max_conf = max(d.confidence for d in det_list)
    tie_group = [d for d in det_list if (max_conf - d.confidence) <= conf_tie_margin]

    # Primary: confidence (by tie-group definition), secondary: area.
    tie_group.sort(key=lambda d: (d.area, d.confidence), reverse=True)
    return tie_group[0]
