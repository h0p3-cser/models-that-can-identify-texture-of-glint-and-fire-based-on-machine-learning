from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class CropMeta:
    crop_side: int
    padded_left: int
    padded_top: int
    padded_right: int
    padded_bottom: int
    square_x1: int
    square_y1: int
    square_x2: int
    square_y2: int


def _clip_bbox_to_image(bbox_xyxy: Sequence[float], image_width: int, image_height: int) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox_xyxy
    x1 = float(np.clip(x1, 0, image_width - 1))
    y1 = float(np.clip(y1, 0, image_height - 1))
    x2 = float(np.clip(x2, 1, image_width))
    y2 = float(np.clip(y2, 1, image_height))

    if x2 <= x1:
        x2 = min(float(image_width), x1 + 1.0)
    if y2 <= y1:
        y2 = min(float(image_height), y1 + 1.0)

    return x1, y1, x2, y2


def select_main_detection(
    detections: List[Dict],
    strategy: str = "score",
) -> Optional[Dict]:
    """
    Select one detection.

    strategy='score': confidence * sqrt(area)
    strategy='conf' : highest confidence, area as tiebreaker
    """
    if not detections:
        return None

    if strategy == "conf":
        return max(detections, key=lambda d: (float(d["confidence"]), float(d["area"])))

    return max(
        detections,
        key=lambda d: float(d["confidence"]) * sqrt(max(float(d["area"]), 1.0)),
    )


def build_square_crop_from_bbox(
    bbox_xyxy: Sequence[float],
    target_ratio: float = 0.6,
) -> Dict[str, float]:
    """
    Build square crop using target occupancy ratio.

    side = max(bw, bh) / target_ratio
    """
    if target_ratio <= 0:
        raise ValueError("target_ratio must be > 0")

    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    side = max(bw, bh) / target_ratio
    side = max(2.0, side)

    return {
        "cx": cx,
        "cy": cy,
        "bw": bw,
        "bh": bh,
        "side": side,
        "crop_x1": cx - side / 2.0,
        "crop_y1": cy - side / 2.0,
        "crop_x2": cx + side / 2.0,
        "crop_y2": cy + side / 2.0,
    }


def pad_image_if_needed(
    image: np.ndarray,
    crop_xyxy: Sequence[float],
    pad_color: Tuple[int, int, int] = (0, 0, 0),
) -> Tuple[np.ndarray, Tuple[int, int, int, int], Dict[str, int]]:
    """
    Pad image when crop box goes out of boundary.

    Returns:
        padded_image,
        adjusted_crop_box_in_padded_image (int xyxy),
        padding dict.
    """
    h, w = image.shape[:2]
    x1f, y1f, x2f, y2f = crop_xyxy

    x1 = int(np.floor(x1f))
    y1 = int(np.floor(y1f))
    x2 = int(np.ceil(x2f))
    y2 = int(np.ceil(y2f))

    if x2 <= x1:
        x2 = x1 + 1
    if y2 <= y1:
        y2 = y1 + 1

    padded_left = max(0, -x1)
    padded_top = max(0, -y1)
    padded_right = max(0, x2 - w)
    padded_bottom = max(0, y2 - h)

    if padded_left or padded_top or padded_right or padded_bottom:
        padded_image = cv2.copyMakeBorder(
            image,
            padded_top,
            padded_bottom,
            padded_left,
            padded_right,
            borderType=cv2.BORDER_CONSTANT,
            value=pad_color,
        )
    else:
        padded_image = image

    adj_x1 = x1 + padded_left
    adj_y1 = y1 + padded_top
    adj_x2 = x2 + padded_left
    adj_y2 = y2 + padded_top

    return padded_image, (adj_x1, adj_y1, adj_x2, adj_y2), {
        "left": padded_left,
        "top": padded_top,
        "right": padded_right,
        "bottom": padded_bottom,
    }


def _make_square_if_needed(image: np.ndarray, pad_color: Tuple[int, int, int]) -> np.ndarray:
    h, w = image.shape[:2]
    if h == w:
        return image

    side = max(h, w)
    top = (side - h) // 2
    bottom = side - h - top
    left = (side - w) // 2
    right = side - w - left

    return cv2.copyMakeBorder(
        image,
        top,
        bottom,
        left,
        right,
        borderType=cv2.BORDER_CONSTANT,
        value=pad_color,
    )


def crop_and_resize_to_512(
    image: np.ndarray,
    bbox_xyxy: Sequence[float],
    img_size: int = 512,
    target_ratio: float = 0.6,
    pad_color: Tuple[int, int, int] = (0, 0, 0),
) -> Tuple[np.ndarray, CropMeta]:
    """Core pipeline: center-square crop + boundary padding + resize to fixed size."""
    h, w = image.shape[:2]
    clipped_bbox = _clip_bbox_to_image(bbox_xyxy, image_width=w, image_height=h)

    square = build_square_crop_from_bbox(clipped_bbox, target_ratio=target_ratio)
    square_box = (square["crop_x1"], square["crop_y1"], square["crop_x2"], square["crop_y2"])

    padded_image, adjusted_box, pad_info = pad_image_if_needed(
        image=image,
        crop_xyxy=square_box,
        pad_color=pad_color,
    )

    x1, y1, x2, y2 = adjusted_box
    crop = padded_image[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError("Empty crop after padding.")

    crop = _make_square_if_needed(crop, pad_color=pad_color)
    resized = cv2.resize(crop, (img_size, img_size), interpolation=cv2.INTER_LINEAR)

    meta = CropMeta(
        crop_side=max(crop.shape[0], crop.shape[1]),
        padded_left=pad_info["left"],
        padded_top=pad_info["top"],
        padded_right=pad_info["right"],
        padded_bottom=pad_info["bottom"],
        square_x1=x1,
        square_y1=y1,
        square_x2=x2,
        square_y2=y2,
    )
    return resized, meta


def draw_preview(
    image: np.ndarray,
    bbox_xyxy: Sequence[float],
    square_xyxy: Sequence[int],
) -> np.ndarray:
    """Visualization helper: draw selected bbox and square crop."""
    vis = image.copy()

    bx1, by1, bx2, by2 = [int(round(v)) for v in bbox_xyxy]
    sx1, sy1, sx2, sy2 = [int(v) for v in square_xyxy]

    cv2.rectangle(vis, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
    cv2.rectangle(vis, (sx1, sy1), (sx2, sy2), (255, 255, 0), 2)
    return vis
