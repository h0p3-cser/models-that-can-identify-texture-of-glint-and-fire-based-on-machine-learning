from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.feature_extraction import CLASS_TO_LABEL, build_feature_dataframe, get_feature_columns
from utils.metrics import compute_classification_metrics, metrics_table

try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train handcrafted-texture classifiers for fire vs glint.")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT, help="Project root containing fire/ and glint/ folders.")
    parser.add_argument(
        "--feature-csv",
        type=Path,
        default=PROJECT_ROOT / "features" / "dataset_features.csv",
        help="Feature CSV location. The latest features are rebuilt by default.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Holdout test ratio.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Reuse an existing feature CSV instead of rebuilding features from the current folders.",
    )
    parser.add_argument(
        "--skip-tuning",
        action="store_true",
        help="Train with default model settings and skip GridSearchCV tuning.",
    )
    return parser.parse_args()


def load_or_create_features(data_root: Path, feature_csv: Path, use_cache: bool = False) -> pd.DataFrame:
    feature_csv.parent.mkdir(parents=True, exist_ok=True)
    if use_cache and feature_csv.exists():
        return pd.read_csv(feature_csv)

    df = build_feature_dataframe(data_root)
    df.to_csv(feature_csv, index=False, encoding="utf-8-sig")
    return df


def save_linear_model_explanations(model: Pipeline, feature_columns: list[str], output_path: Path) -> None:
    coefficients = model.named_steps["clf"].coef_[0]
    coef_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "coefficient": coefficients,
            "abs_coefficient": np.abs(coefficients),
        }
    ).sort_values(by="abs_coefficient", ascending=False)
    coef_df.to_csv(output_path, index=False, encoding="utf-8-sig")


def save_random_forest_explanations(model: RandomForestClassifier, feature_columns: list[str], output_path: Path) -> None:
    importance_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": model.feature_importances_,
        }
    ).sort_values(by="importance", ascending=False)
    importance_df.to_csv(output_path, index=False, encoding="utf-8-sig")


def build_model_zoo(random_state: int) -> dict[str, object]:
    models: dict[str, object] = {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=4000, class_weight="balanced", random_state=random_state)),
            ]
        ),
        "svm_rbf": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=random_state)),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
    }

    if HAS_XGBOOST:
        models["xgboost"] = XGBClassifier(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
        )

    return models


def build_search_spaces() -> dict[str, dict[str, list[object]]]:
    spaces = {
        "logistic_regression": {
            "clf__C": [0.5, 1.0, 2.0, 4.0],
        },
        "svm_rbf": {
            "clf__C": [0.5, 1.0, 2.0, 4.0, 8.0],
            "clf__gamma": ["scale", 0.01, 0.03, 0.1],
        },
        "random_forest": {
            "n_estimators": [300, 500],
            "max_depth": [None, 10],
            "min_samples_leaf": [1, 2],
            "max_features": ["sqrt", 0.7],
        },
    }

    if HAS_XGBOOST:
        spaces["xgboost"] = {
            "n_estimators": [200, 300],
            "max_depth": [3, 4, 5],
            "learning_rate": [0.03, 0.05, 0.08],
            "subsample": [0.8, 0.9],
        }

    return spaces


def maybe_tune_model(
    model_name: str,
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int,
    skip_tuning: bool,
) -> tuple[object, dict[str, object]]:
    if skip_tuning:
        return model, {}

    search_spaces = build_search_spaces()
    if model_name not in search_spaces:
        return model, {}

    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=random_state)
    search = GridSearchCV(
        estimator=model,
        param_grid=search_spaces[model_name],
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


def main() -> None:
    args = parse_args()
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    df = load_or_create_features(args.data_root, args.feature_csv, use_cache=args.use_cache)
    feature_columns = get_feature_columns(df)

    X = df[feature_columns].astype(np.float32)
    y = df["label"].astype(int)
    metadata = df[["image_path", "file_name", "class_name", "label"]].copy()

    (
        X_train,
        X_test,
        y_train,
        y_test,
        meta_train,
        meta_test,
    ) = train_test_split(
        X,
        y,
        metadata,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    models = build_model_zoo(args.random_state)
    results: list[dict[str, object]] = []
    trained_models: dict[str, object] = {}
    best_params_by_model: dict[str, dict[str, object]] = {}
    test_predictions = meta_test.reset_index(drop=True).copy()

    for model_name, model in models.items():
        final_model, best_params = maybe_tune_model(
            model_name,
            model,
            X_train,
            y_train,
            random_state=args.random_state,
            skip_tuning=args.skip_tuning,
        )
        final_model.fit(X_train, y_train)
        predictions = final_model.predict(X_test)

        metrics = compute_classification_metrics(y_test, predictions)
        metrics["model"] = model_name
        results.append(metrics)
        trained_models[model_name] = final_model
        best_params_by_model[model_name] = best_params
        test_predictions[f"pred_{model_name}"] = predictions

        model_path = models_dir / f"{model_name}.joblib"
        joblib.dump(final_model, model_path)

        print(f"\n=== {model_name} ===")
        if best_params:
            print(f"best_params={best_params}")
        print(
            f"accuracy={metrics['accuracy']:.4f}, precision={metrics['precision']:.4f}, "
            f"recall={metrics['recall']:.4f}, f1={metrics['f1_score']:.4f}"
        )
        print(f"confusion_matrix={metrics['confusion_matrix']}")

    comparison_df = metrics_table(results)
    comparison_path = models_dir / "model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False, encoding="utf-8-sig")

    best_model_name = comparison_df.iloc[0]["model"]
    best_model = trained_models[best_model_name]
    joblib.dump(best_model, models_dir / "best_model.joblib")

    training_config = {
        "class_to_label": CLASS_TO_LABEL,
        "test_size": args.test_size,
        "random_state": args.random_state,
        "feature_csv": str(args.feature_csv),
        "feature_columns": feature_columns,
        "feature_count": len(feature_columns),
        "best_model": best_model_name,
        "use_cache": args.use_cache,
        "tuning_enabled": not args.skip_tuning,
        "best_params_by_model": best_params_by_model,
    }
    with open(models_dir / "training_config.json", "w", encoding="utf-8") as fp:
        json.dump(training_config, fp, ensure_ascii=False, indent=2)

    with open(models_dir / "model_metrics.json", "w", encoding="utf-8") as fp:
        json.dump(results, fp, ensure_ascii=False, indent=2)

    test_predictions.to_csv(models_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")

    X_train.assign(label=y_train.values).to_csv(PROJECT_ROOT / "features" / "train_split.csv", index=False, encoding="utf-8-sig")
    X_test.assign(label=y_test.values).to_csv(PROJECT_ROOT / "features" / "test_split.csv", index=False, encoding="utf-8-sig")

    save_linear_model_explanations(
        trained_models["logistic_regression"],
        feature_columns,
        models_dir / "logistic_coefficients.csv",
    )
    save_random_forest_explanations(
        trained_models["random_forest"],
        feature_columns,
        models_dir / "random_forest_feature_importance.csv",
    )

    print("\n=== Model Comparison ===")
    print(comparison_df.to_string(index=False))
    print(f"\nFeature count: {len(feature_columns)}")
    print(f"Best model saved as: {best_model_name}")


if __name__ == "__main__":
    main()
