from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.image_utils import load_rgb_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create visual evaluation reports for the trained fire/glint classifiers.")
    parser.add_argument(
        "--feature-csv",
        type=Path,
        default=PROJECT_ROOT / "features" / "dataset_features.csv",
        help="Feature CSV created during training.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=PROJECT_ROOT / "models",
        help="Directory with trained models and training_config.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "models" / "figures",
        help="Folder used to store all generated charts.",
    )
    return parser.parse_args()


def load_training_artifacts(feature_csv: Path, models_dir: Path):
    config = json.loads((models_dir / "training_config.json").read_text(encoding="utf-8"))
    df = pd.read_csv(feature_csv)
    feature_columns = config["feature_columns"]
    metadata_columns = ["image_path", "file_name", "class_name", "label"]

    X = df[feature_columns].astype(np.float32)
    y = df["label"].astype(int)
    metadata = df[metadata_columns].copy()

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
        test_size=config["test_size"],
        random_state=config["random_state"],
        stratify=y,
    )

    models = {}
    for model_path in sorted(models_dir.glob("*.joblib")):
        if model_path.stem == "best_model":
            continue
        models[model_path.stem] = joblib.load(model_path)

    return config, df, feature_columns, X_train, X_test, y_train, y_test, meta_train, meta_test, models


def get_model_scores(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        scores = np.asarray(scores, dtype=np.float64)
        min_score, max_score = scores.min(), scores.max()
        if math.isclose(min_score, max_score):
            return np.zeros_like(scores)
        return (scores - min_score) / (max_score - min_score)
    return model.predict(X).astype(np.float64)


def plot_model_comparison(comparison_df: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].bar(comparison_df["model"], comparison_df["accuracy"], color=["#4C78A8", "#F58518", "#54A24B"])
    axes[0].set_title("Holdout Accuracy")
    axes[0].set_ylim(0, 1.05)
    axes[0].tick_params(axis="x", rotation=15)
    for idx, value in enumerate(comparison_df["accuracy"]):
        axes[0].text(idx, value + 0.02, f"{value:.3f}", ha="center")

    axes[1].bar(comparison_df["model"], comparison_df["f1_score"], color=["#4C78A8", "#F58518", "#54A24B"])
    axes[1].set_title("Holdout F1 Score")
    axes[1].set_ylim(0, 1.05)
    axes[1].tick_params(axis="x", rotation=15)
    for idx, value in enumerate(comparison_df["f1_score"]):
        axes[1].text(idx, value + 0.02, f"{value:.3f}", ha="center")

    fig.suptitle("Model Comparison on the 20% Holdout Set")
    fig.tight_layout()
    fig.savefig(output_dir / "model_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrices(models: dict[str, object], X_test: pd.DataFrame, y_test: pd.Series, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4.5))
    if len(models) == 1:
        axes = [axes]

    class_labels = ["glint", "fire"]
    for ax, (model_name, model) in zip(axes, models.items()):
        predictions = model.predict(X_test)
        disp = ConfusionMatrixDisplay.from_predictions(
            y_test,
            predictions,
            display_labels=class_labels,
            cmap="YlOrRd",
            colorbar=False,
            ax=ax,
        )
        disp.ax_.set_title(model_name)

    fig.suptitle("Confusion Matrices")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrices.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curves(models: dict[str, object], X_test: pd.DataFrame, y_test: pd.Series, output_dir: Path) -> pd.DataFrame:
    fig, ax = plt.subplots(figsize=(6, 6))
    summary = []

    for model_name, model in models.items():
        scores = get_model_scores(model, X_test)
        RocCurveDisplay.from_predictions(y_test, scores, name=model_name, ax=ax)
        auc = roc_auc_score(y_test, scores)
        summary.append({"model": model_name, "roc_auc": float(auc)})

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_title("ROC Curves on the Holdout Set")
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    summary_df = pd.DataFrame(summary).sort_values(by="roc_auc", ascending=False)
    summary_df.to_csv(output_dir / "roc_auc_summary.csv", index=False, encoding="utf-8-sig")
    return summary_df


def plot_top_features(models_dir: Path, output_dir: Path) -> None:
    rf_path = models_dir / "random_forest_feature_importance.csv"
    lr_path = models_dir / "logistic_coefficients.csv"
    rf_df = pd.read_csv(rf_path).head(12)
    lr_df = pd.read_csv(lr_path).head(12).copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].barh(rf_df["feature"][::-1], rf_df["importance"][::-1], color="#E45756")
    axes[0].set_title("Random Forest Top Features")
    axes[0].set_xlabel("Importance")

    colors = ["#72B7B2" if coef >= 0 else "#B279A2" for coef in lr_df["coefficient"]]
    axes[1].barh(lr_df["feature"][::-1], lr_df["coefficient"][::-1], color=colors[::-1])
    axes[1].set_title("Logistic Regression Top Coefficients")
    axes[1].set_xlabel("Coefficient")

    fig.tight_layout()
    fig.savefig(output_dir / "top_features.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_pca_embedding(df: pd.DataFrame, feature_columns: list[str], output_dir: Path) -> None:
    pca = PCA(n_components=2, random_state=42)
    embedding = pca.fit_transform(df[feature_columns].astype(np.float32))

    colors = df["label"].map({0: "#4C78A8", 1: "#F58518"})
    labels = df["class_name"].map({"glint": "glint", "fire": "fire"})

    fig, ax = plt.subplots(figsize=(7, 6))
    for class_name, color in [("glint", "#4C78A8"), ("fire", "#F58518")]:
        idx = labels == class_name
        ax.scatter(embedding[idx, 0], embedding[idx, 1], c=color, label=class_name, s=55, alpha=0.8, edgecolors="white")

    ax.set_title("PCA Projection of Handcrafted Features")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% var)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "pca_feature_space.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_prediction_gallery(
    best_model_name: str,
    best_model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    meta_test: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    preds = best_model.predict(X_test)
    scores = get_model_scores(best_model, X_test)

    gallery_df = meta_test.reset_index(drop=True).copy()
    gallery_df["prediction"] = preds
    gallery_df["score_fire"] = scores
    gallery_df["correct"] = gallery_df["prediction"] == gallery_df["label"]
    gallery_df["true_label_name"] = gallery_df["label"].map({0: "glint", 1: "fire"})
    gallery_df["pred_label_name"] = gallery_df["prediction"].map({0: "glint", 1: "fire"})

    n = len(gallery_df)
    cols = 4
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax in axes[n:]:
        ax.axis("off")

    for ax, row in zip(axes, gallery_df.itertuples(index=False)):
        image_path = Path(row.image_path)
        if image_path.exists():
            image = load_rgb_image(image_path)
            ax.imshow(image)
        else:
            ax.imshow(np.zeros((64, 64, 3), dtype=np.uint8))
        border_color = "#2E8B57" if row.correct else "#C0392B"
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(4)
        ax.set_title(
            f"{row.file_name}\ntrue={row.true_label_name}, pred={row.pred_label_name}\nfire_score={row.score_fire:.3f}",
            fontsize=9,
        )
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"Holdout Predictions of {best_model_name} (green=correct, red=wrong)", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / f"{best_model_name}_prediction_gallery.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    gallery_df.to_csv(output_dir / f"{best_model_name}_prediction_gallery.csv", index=False, encoding="utf-8-sig")
    return gallery_df


def run_cross_validation(models: dict[str, object], df: pd.DataFrame, feature_columns: list[str], output_dir: Path) -> pd.DataFrame:
    X = df[feature_columns].astype(np.float32)
    y = df["label"].astype(int)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)

    rows = []
    for model_name, model in models.items():
        result = cross_validate(
            clone(model),
            X,
            y,
            cv=cv,
            scoring=["accuracy", "precision", "recall", "f1"],
            n_jobs=-1,
        )
        rows.append(
            {
                "model": model_name,
                "accuracy_mean": float(result["test_accuracy"].mean()),
                "accuracy_std": float(result["test_accuracy"].std()),
                "precision_mean": float(result["test_precision"].mean()),
                "precision_std": float(result["test_precision"].std()),
                "recall_mean": float(result["test_recall"].mean()),
                "recall_std": float(result["test_recall"].std()),
                "f1_mean": float(result["test_f1"].mean()),
                "f1_std": float(result["test_f1"].std()),
            }
        )

    cv_df = pd.DataFrame(rows).sort_values(by="f1_mean", ascending=False)
    cv_df.to_csv(output_dir / "cross_validation_summary.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(cv_df["model"], cv_df["f1_mean"], yerr=cv_df["f1_std"], color=["#E45756", "#4C78A8", "#54A24B"], capsize=6)
    ax.set_ylim(0, 1.05)
    ax.set_title("Repeated 5-Fold CV F1 Score")
    ax.set_ylabel("F1 mean +/- std")
    ax.tick_params(axis="x", rotation=15)
    for idx, value in enumerate(cv_df["f1_mean"]):
        ax.text(idx, value + 0.02, f"{value:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(output_dir / "cross_validation_f1.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    return cv_df


def save_summary_text(
    comparison_df: pd.DataFrame,
    roc_df: pd.DataFrame,
    cv_df: pd.DataFrame,
    gallery_df: pd.DataFrame,
    best_model_name: str,
    output_dir: Path,
) -> None:
    wrong_df = gallery_df[~gallery_df["correct"]]

    lines = []
    lines.append("Fire vs Glint Classifier Evaluation Summary")
    lines.append("=" * 42)
    lines.append("")
    lines.append("1. Holdout metrics")
    lines.append(comparison_df.to_string(index=False))
    lines.append("")
    lines.append("2. ROC-AUC")
    lines.append(roc_df.to_string(index=False))
    lines.append("")
    lines.append("3. Repeated cross-validation")
    lines.append(cv_df.to_string(index=False))
    lines.append("")
    lines.append(f"4. Best model: {best_model_name}")
    lines.append(f"   Misclassified holdout samples: {len(wrong_df)} / {len(gallery_df)}")
    if wrong_df.empty:
        lines.append("   No holdout errors.")
    else:
        for row in wrong_df.itertuples(index=False):
            lines.append(
                f"   {row.file_name}: true={row.true_label_name}, pred={row.pred_label_name}, fire_score={row.score_fire:.3f}"
            )

    (output_dir / "evaluation_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    (
        config,
        df,
        feature_columns,
        _X_train,
        X_test,
        _y_train,
        y_test,
        _meta_train,
        meta_test,
        models,
    ) = load_training_artifacts(args.feature_csv, args.models_dir)

    comparison_df = pd.read_csv(args.models_dir / "model_comparison.csv")
    plot_model_comparison(comparison_df, args.output_dir)
    plot_confusion_matrices(models, X_test, y_test, args.output_dir)
    roc_df = plot_roc_curves(models, X_test, y_test, args.output_dir)
    plot_top_features(args.models_dir, args.output_dir)
    plot_pca_embedding(df, feature_columns, args.output_dir)
    cv_df = run_cross_validation(models, df, feature_columns, args.output_dir)

    best_model_name = config["best_model"]
    best_model = models[best_model_name]
    gallery_df = build_prediction_gallery(best_model_name, best_model, X_test, y_test, meta_test, args.output_dir)

    save_summary_text(comparison_df, roc_df, cv_df, gallery_df, best_model_name, args.output_dir)

    print(f"Saved figures to: {args.output_dir}")
    print("Generated files:")
    for path in sorted(args.output_dir.iterdir()):
        print(f" - {path.name}")


if __name__ == "__main__":
    main()

