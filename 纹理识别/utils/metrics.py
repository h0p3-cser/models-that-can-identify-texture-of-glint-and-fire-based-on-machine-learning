from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def compute_classification_metrics(y_true: Iterable[int], y_pred: Iterable[int]) -> dict[str, object]:
    """Compute the required binary-classification metrics in one place."""

    y_true = np.asarray(list(y_true))
    y_pred = np.asarray(list(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": cm.tolist(),
    }


def metrics_table(results: list[dict[str, object]]) -> pd.DataFrame:
    """Convert model results into a compact comparison table."""

    rows = []
    for result in results:
        row = {
            "model": result["model"],
            "accuracy": result["accuracy"],
            "precision": result["precision"],
            "recall": result["recall"],
            "f1_score": result["f1_score"],
            "tn": result["confusion_matrix"][0][0],
            "fp": result["confusion_matrix"][0][1],
            "fn": result["confusion_matrix"][1][0],
            "tp": result["confusion_matrix"][1][1],
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values(by="f1_score", ascending=False)
