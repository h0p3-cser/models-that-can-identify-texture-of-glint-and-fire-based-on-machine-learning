from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.feature_extraction import build_feature_dataframe, get_feature_columns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze feature importance for the fire vs glint classifier.")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT, help="Project root containing fire/ and glint/ folders.")
    parser.add_argument(
        "--feature-csv",
        type=Path,
        default=PROJECT_ROOT / "features" / "dataset_features.csv",
        help="Feature CSV path. Rebuilt by default unless --use-cache is passed.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "models" / "feature_analysis",
        help="Directory used to save rankings and charts.",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Reuse the existing feature CSV instead of rebuilding it from current images.",
    )
    return parser.parse_args()


def load_or_create_features(data_root: Path, feature_csv: Path, use_cache: bool = False) -> pd.DataFrame:
    feature_csv.parent.mkdir(parents=True, exist_ok=True)
    if use_cache and feature_csv.exists():
        return pd.read_csv(feature_csv)

    df = build_feature_dataframe(data_root)
    df.to_csv(feature_csv, index=False, encoding="utf-8-sig")
    return df


def infer_feature_group(feature_name: str) -> str:
    if feature_name.startswith("highlight_"):
        return "highlight_core"
    if feature_name.startswith("clahe_"):
        return "clahe_texture"
    if feature_name.startswith("ms256_"):
        return "multiscale_256"
    if feature_name.startswith("center_"):
        return "multiscale_center"
    if feature_name.startswith("block_"):
        return "spatial_blocks"
    if feature_name.startswith("HOG_") or feature_name in {"Sobel_x_over_y", "dominant_direction_ratio", "gradient_direction_entropy"}:
        return "directional_texture"
    if feature_name in {
        "Sobel_mean",
        "Sobel_std",
        "Laplacian_variance",
        "Canny_edge_ratio",
    }:
        return "base_edges"
    if feature_name.startswith("LBP_") or feature_name.startswith("GLCM_") or feature_name == "entropy":
        return "base_texture"
    if feature_name.startswith("pixel_") or feature_name.startswith("R_") or feature_name.startswith("G_") or feature_name.startswith("B_") or feature_name.startswith("RGB_"):
        return "color_relationships"
    if feature_name.startswith("H_") or feature_name.startswith("S_") or feature_name.startswith("V_"):
        return "hsv_features"
    if feature_name.startswith("gray_") or feature_name == "bright_pixel_ratio":
        return "brightness_features"
    return "other"


def plot_feature_importance(feature_df: pd.DataFrame, group_df: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    top_features = feature_df.head(20).copy()
    axes[0].barh(top_features["feature"][::-1], top_features["importance"][::-1], color="#E45756")
    axes[0].set_title("Top 20 Random Forest Features")
    axes[0].set_xlabel("Importance")

    axes[1].barh(group_df["group"][::-1], group_df["importance_sum"][::-1], color="#4C78A8")
    axes[1].set_title("Feature Group Importance")
    axes[1].set_xlabel("Total Importance")

    fig.tight_layout()
    fig.savefig(output_dir / "feature_importance_report.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_or_create_features(args.data_root, args.feature_csv, use_cache=args.use_cache)
    feature_columns = get_feature_columns(df)
    X = df[feature_columns].astype("float32")
    y = df["label"].astype(int)

    model = RandomForestClassifier(
        n_estimators=600,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X, y)

    importance_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": model.feature_importances_,
        }
    ).sort_values(by="importance", ascending=False)
    importance_df["group"] = importance_df["feature"].map(infer_feature_group)
    importance_df.to_csv(args.output_dir / "feature_importance_full_dataset.csv", index=False, encoding="utf-8-sig")

    group_df = (
        importance_df.groupby("group", as_index=False)
        .agg(importance_sum=("importance", "sum"), importance_mean=("importance", "mean"), feature_count=("feature", "count"))
        .sort_values(by="importance_sum", ascending=False)
    )
    group_df.to_csv(args.output_dir / "feature_group_importance.csv", index=False, encoding="utf-8-sig")

    plot_feature_importance(importance_df, group_df, args.output_dir)

    print("Top 15 features:")
    print(importance_df.head(15).to_string(index=False))
    print("\nFeature group importance:")
    print(group_df.to_string(index=False))
    print(f"\nSaved analysis to: {args.output_dir}")


if __name__ == "__main__":
    main()
