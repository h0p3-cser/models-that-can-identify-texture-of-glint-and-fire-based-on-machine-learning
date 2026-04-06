from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.feature_extraction import get_feature_columns
from utils.metrics import compute_classification_metrics, metrics_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved models on the holdout test split.")
    parser.add_argument(
        "--feature-csv",
        type=Path,
        default=PROJECT_ROOT / "features" / "dataset_features.csv",
        help="Feature CSV created by scripts/extract_features.py or scripts/train_model.py.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=PROJECT_ROOT / "models",
        help="Directory that stores trained .joblib models and training_config.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.models_dir / "training_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {config_path}. Run scripts/train_model.py first.")
    if not args.feature_csv.exists():
        raise FileNotFoundError(f"Missing feature CSV: {args.feature_csv}.")

    with open(config_path, "r", encoding="utf-8") as fp:
        config = json.load(fp)

    df = pd.read_csv(args.feature_csv)
    feature_columns = config.get("feature_columns") or get_feature_columns(df)
    X = df[feature_columns]
    y = df["label"].astype(int)

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=config["test_size"],
        random_state=config["random_state"],
        stratify=y,
    )

    results = []
    for model_path in sorted(args.models_dir.glob("*.joblib")):
        if model_path.stem == "best_model":
            continue
        model_name = model_path.stem
        model = joblib.load(model_path)
        predictions = model.predict(X_test)
        metrics = compute_classification_metrics(y_test, predictions)
        metrics["model"] = model_name
        results.append(metrics)

        print(f"\n=== {model_name} ===")
        print(
            f"accuracy={metrics['accuracy']:.4f}, precision={metrics['precision']:.4f}, "
            f"recall={metrics['recall']:.4f}, f1={metrics['f1_score']:.4f}"
        )
        print(f"confusion_matrix={metrics['confusion_matrix']}")

    comparison_df = metrics_table(results)
    comparison_path = args.models_dir / "evaluation_summary.csv"
    comparison_df.to_csv(comparison_path, index=False, encoding="utf-8-sig")

    print("\n=== Evaluation Summary ===")
    print(comparison_df.to_string(index=False))
    print(f"\nSaved evaluation summary to: {comparison_path}")


if __name__ == "__main__":
    main()
