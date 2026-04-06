from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: Tuple[float, float, float, float]
    area: float
    bbox_source: str  # 'mask' or 'box'


class YoloInfer:
    def __init__(
        self,
        model_path: str | Path,
        conf_thres: float = 0.25,
        iou_thres: float = 0.7,
        device: str = "auto",
    ) -> None:
        self.model_path = str(Path(model_path).resolve())
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = self._resolve_device(device)

        self.model = YOLO(self.model_path)
        self.class_names = self._extract_class_names()

    @staticmethod
    def _resolve_device(device_arg: str):
        if device_arg != "auto":
            return device_arg

        try:
            import torch

            return 0 if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _extract_class_names(self) -> Dict[int, str]:
        names = getattr(self.model.model, "names", None)
        if names is None:
            return {}
        if isinstance(names, list):
            return {i: str(n) for i, n in enumerate(names)}
        return {int(k): str(v) for k, v in names.items()}

    @staticmethod
    def _bbox_from_polygon(points: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
        if points is None or len(points) == 0:
            return None

        x = points[:, 0]
        y = points[:, 1]
        x1 = float(np.min(x))
        y1 = float(np.min(y))
        x2 = float(np.max(x))
        y2 = float(np.max(y))

        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    def infer(self, image: np.ndarray) -> List[Detection]:
        result = self.model.predict(
            source=image,
            conf=self.conf_thres,
            iou=self.iou_thres,
            device=self.device,
            verbose=False,
        )[0]

        if result.boxes is None or len(result.boxes) == 0:
            return []

        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        clses = result.boxes.cls.cpu().numpy().astype(int)

        mask_polygons: Sequence[np.ndarray] | None = None
        if result.masks is not None and getattr(result.masks, "xy", None) is not None:
            mask_polygons = result.masks.xy

        detections: List[Detection] = []

        for idx, (bbox, conf, cls_id) in enumerate(zip(boxes, confs, clses)):
            bbox_source = "box"
            final_bbox: Tuple[float, float, float, float]

            if mask_polygons is not None and idx < len(mask_polygons):
                poly = np.asarray(mask_polygons[idx], dtype=np.float32)
                poly_bbox = self._bbox_from_polygon(poly)
                if poly_bbox is not None:
                    final_bbox = poly_bbox
                    bbox_source = "mask"
                else:
                    final_bbox = tuple(float(v) for v in bbox)
            else:
                final_bbox = tuple(float(v) for v in bbox)

            x1, y1, x2, y2 = final_bbox
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            if area <= 0:
                continue

            detections.append(
                Detection(
                    class_id=int(cls_id),
                    class_name=self.class_names.get(int(cls_id), f"class_{cls_id}"),
                    confidence=float(conf),
                    bbox_xyxy=final_bbox,
                    area=float(area),
                    bbox_source=bbox_source,
                )
            )

        return detections
