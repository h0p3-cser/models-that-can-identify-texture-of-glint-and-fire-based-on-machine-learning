from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

LOG_COLUMNS = [
    "source_folder",
    "source_image",
    "class_name",
    "detected_confidence",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "crop_side",
    "padded_left",
    "padded_top",
    "padded_right",
    "padded_bottom",
    "output_path",
    "status",
    "error_message",
]


def new_log_row() -> Dict:
    return {k: "" for k in LOG_COLUMNS}


def save_crop_log(rows: List[Dict], log_path: Path | str) -> Path:
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    normalized_rows: List[Dict] = []
    for row in rows:
        fixed = {k: row.get(k, "") for k in LOG_COLUMNS}
        normalized_rows.append(fixed)

    df = pd.DataFrame(normalized_rows, columns=LOG_COLUMNS)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path
