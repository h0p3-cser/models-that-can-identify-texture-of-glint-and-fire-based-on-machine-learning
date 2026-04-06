from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.feature_extraction import build_feature_dataframe, get_feature_columns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract mask-aware handcrafted features for fire/glint classification.")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT, help="Project root containing fire/ and glint/ folders.")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=PROJECT_ROOT / "features" / "dataset_features.csv",
        help="Where to save the extracted feature table.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    df = build_feature_dataframe(args.data_root)
    df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")

    print(f"Saved features to: {args.output_csv}")
    print(f"Number of samples: {len(df)}")
    print(f"Number of handcrafted features: {len(get_feature_columns(df))}")
    print(df["class_name"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
