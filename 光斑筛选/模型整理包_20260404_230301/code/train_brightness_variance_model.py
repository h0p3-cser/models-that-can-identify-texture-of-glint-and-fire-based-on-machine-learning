import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class DatasetEntry:
    image_path: Path
    filename: str
    label: int
    folder: str


class BrightnessFeatureExtractor:
    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        grid_size: int = 5,
        bright_threshold: int = 200,
        red_dominance_delta: int = 30,
    ):
        self.image_size = image_size
        self.grid_size = grid_size
        self.bright_threshold = bright_threshold
        self.red_dominance_delta = red_dominance_delta

    def _read_and_preprocess(self, image_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        buffer = np.fromfile(str(image_path), dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        resized = cv2.resize(image, self.image_size, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        return resized, gray

    def _brightness_stats(self, gray: np.ndarray) -> Dict[str, float]:
        values = gray.astype(np.float32)
        return {
            "brightness_mean": float(np.mean(values)),
            "brightness_std": float(np.std(values)),
            "brightness_max": float(np.max(values)),
            "brightness_min": float(np.min(values)),
            "brightness_p95": float(np.percentile(values, 95)),
        }

    def _block_features(self, gray: np.ndarray) -> Dict[str, float]:
        height, width = gray.shape
        row_edges = np.linspace(0, height, self.grid_size + 1, dtype=int)
        col_edges = np.linspace(0, width, self.grid_size + 1, dtype=int)
        block_means: List[float] = []

        for row in range(self.grid_size):
            for col in range(self.grid_size):
                block = gray[row_edges[row] : row_edges[row + 1], col_edges[col] : col_edges[col + 1]]
                if block.size == 0:
                    block_means.append(0.0)
                else:
                    block_means.append(float(np.mean(block)))

        block_means_np = np.array(block_means, dtype=np.float32)
        top_k = np.sort(block_means_np)[-3:]
        return {
            "block_mean_var": float(np.var(block_means_np)),
            "block_mean_max": float(np.max(block_means_np)),
            "block_mean_min": float(np.min(block_means_np)),
            "block_top3_mean": float(np.mean(top_k)),
            "bright_block_count": float(np.sum(block_means_np > self.bright_threshold)),
        }

    def _connected_component_features(self, gray: np.ndarray) -> Dict[str, float]:
        _, binary = cv2.threshold(gray, self.bright_threshold, 255, cv2.THRESH_BINARY)
        bright_ratio = float(np.mean(binary > 0))

        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if num_labels <= 1:
            component_count = 0
            largest_area = 0.0
        else:
            areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float32)
            component_count = int(len(areas))
            largest_area = float(np.max(areas))

        total_pixels = float(gray.shape[0] * gray.shape[1])
        largest_ratio = largest_area / total_pixels if total_pixels > 0 else 0.0
        return {
            "bright_pixel_ratio": bright_ratio,
            "connected_component_count": float(component_count),
            "largest_component_area": largest_area,
            "largest_component_ratio": largest_ratio,
        }

    def _texture_features(self, gray: np.ndarray) -> Dict[str, float]:
        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        sobel_mag = cv2.magnitude(sobel_x, sobel_y)
        laplacian = cv2.Laplacian(gray, cv2.CV_32F)
        return {
            "sobel_mean": float(np.mean(sobel_mag)),
            "laplacian_var": float(np.var(laplacian)),
        }

    def _rgb_stats(self, image_bgr: np.ndarray) -> Dict[str, float]:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        return {
            "rgb_r_mean": float(np.mean(image_rgb[:, :, 0])),
            "rgb_g_mean": float(np.mean(image_rgb[:, :, 1])),
            "rgb_b_mean": float(np.mean(image_rgb[:, :, 2])),
        }

    def _red_hotspot_features(self, image_bgr: np.ndarray, gray: np.ndarray) -> Dict[str, float]:
        image_bgr_f = image_bgr.astype(np.float32)
        b_channel = image_bgr_f[:, :, 0]
        g_channel = image_bgr_f[:, :, 1]
        r_channel = image_bgr_f[:, :, 2]

        red_over_green = r_channel - g_channel
        red_over_blue = r_channel - b_channel
        red_dominance = r_channel - np.maximum(g_channel, b_channel)
        red_ratio = r_channel / (r_channel + g_channel + b_channel + 1e-6)

        bright_mask = gray.astype(np.float32) >= self.bright_threshold
        red_dominant_mask = red_dominance >= float(self.red_dominance_delta)
        bright_red_mask = np.logical_and(bright_mask, red_dominant_mask)

        bright_red_pixel_ratio = float(np.mean(bright_red_mask))
        if np.any(bright_red_mask):
            bright_red_mean_dominance = float(np.mean(red_dominance[bright_red_mask]))
            bright_red_mean_intensity = float(np.mean(r_channel[bright_red_mask]))
        else:
            bright_red_mean_dominance = 0.0
            bright_red_mean_intensity = 0.0

        bright_red_binary = (bright_red_mask.astype(np.uint8)) * 255
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(bright_red_binary, connectivity=8)
        if num_labels <= 1:
            bright_red_component_count = 0
            bright_red_largest_area = 0.0
        else:
            areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float32)
            bright_red_component_count = int(len(areas))
            bright_red_largest_area = float(np.max(areas))

        total_pixels = float(gray.shape[0] * gray.shape[1])
        bright_red_largest_component_ratio = bright_red_largest_area / total_pixels if total_pixels > 0 else 0.0

        return {
            "rgb_r_p95": float(np.percentile(r_channel, 95)),
            "rgb_red_ratio_mean": float(np.mean(red_ratio)),
            "red_over_green_mean": float(np.mean(red_over_green)),
            "red_over_blue_mean": float(np.mean(red_over_blue)),
            "red_dominance_mean": float(np.mean(red_dominance)),
            "bright_red_pixel_ratio": bright_red_pixel_ratio,
            "bright_red_mean_dominance": bright_red_mean_dominance,
            "bright_red_mean_intensity": bright_red_mean_intensity,
            "bright_red_component_count": float(bright_red_component_count),
            "bright_red_largest_component_ratio": bright_red_largest_component_ratio,
        }

    def extract(self, image_path: Path) -> Dict[str, float]:
        image_bgr, gray = self._read_and_preprocess(image_path)
        features: Dict[str, float] = {}
        features.update(self._brightness_stats(gray))
        features.update(self._block_features(gray))
        features.update(self._connected_component_features(gray))
        features.update(self._texture_features(gray))
        features.update(self._rgb_stats(image_bgr))
        features.update(self._red_hotspot_features(image_bgr, gray))
        return features


def list_filenames(directory: Path) -> List[str]:
    return sorted([path.stem for path in directory.iterdir() if path.is_file()])


def natural_numeric_sort(files: Sequence[Path]) -> List[Path]:
    def sort_key(path: Path) -> Tuple[int, int, str]:
        stem = path.stem
        if stem.isdigit():
            return (0, int(stem), path.name.lower())
        return (1, 0, path.name.lower())

    return sorted(files, key=sort_key)


def parse_dir_list(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def build_dataset(positive_dirs: Sequence[Path], negative_dirs: Sequence[Path]) -> List[DatasetEntry]:
    entries: List[DatasetEntry] = []

    for folder in positive_dirs:
        files = natural_numeric_sort([path for path in folder.iterdir() if path.is_file()])
        for path in files:
            entries.append(DatasetEntry(image_path=path, filename=path.stem, label=1, folder=folder.name))

    for folder in negative_dirs:
        files = natural_numeric_sort([path for path in folder.iterdir() if path.is_file()])
        for path in files:
            entries.append(DatasetEntry(image_path=path, filename=path.stem, label=0, folder=folder.name))

    return entries


def build_feature_table(entries: Sequence[DatasetEntry], extractor: BrightnessFeatureExtractor) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for entry in entries:
        features = extractor.extract(entry.image_path)
        row: Dict[str, float] = {
            "filename": entry.filename,
            "folder": entry.folder,
            "label": float(entry.label),
        }
        row.update(features)
        rows.append(row)
    if not rows:
        raise ValueError("数据集中没有可用图像。")
    df = pd.DataFrame(rows)
    df["label"] = df["label"].astype(int)
    return df


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, object]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": cm.tolist(),
    }


def train_variance_threshold_baseline(train_df: pd.DataFrame) -> float:
    variances = np.sort(train_df["block_mean_var"].to_numpy(dtype=np.float32))
    if len(variances) == 0:
        raise ValueError("训练集为空。")
    candidates = np.unique(variances)
    best_threshold = float(candidates[0])
    best_f1 = -1.0

    y_true = train_df["label"].to_numpy(dtype=int)
    for threshold in candidates:
        pred = (train_df["block_mean_var"].to_numpy(dtype=np.float32) >= threshold).astype(int)
        score = f1_score(y_true, pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold


def save_json(path: Path, content: dict) -> None:
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="高亮区域检测：多特征 + 经典机器学习训练脚本")
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--light-dir", type=str, default="light")
    parser.add_argument("--without-light-dir", type=str, default="without_light")
    parser.add_argument("--positive-dirs", type=str, default="")
    parser.add_argument("--negative-dirs", type=str, default="")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--grid-size", type=int, default=5)
    parser.add_argument("--bright-threshold", type=int, default=200)
    parser.add_argument("--red-dominance-delta", type=int, default=30)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lists-output", type=Path, default=Path("序号列表.json"))
    parser.add_argument("--report-output", type=Path, default=Path("亮度方差阈值模型.json"))
    parser.add_argument("--feature-output", type=Path, default=Path("特征表.csv"))
    parser.add_argument("--model-output", type=Path, default=Path("最佳高亮分类器.joblib"))
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    positive_dir_names = parse_dir_list(args.positive_dirs) if args.positive_dirs else [args.light_dir]
    negative_dir_names = parse_dir_list(args.negative_dirs) if args.negative_dirs else [args.without_light_dir]
    positive_dirs = [data_root / name for name in positive_dir_names]
    negative_dirs = [data_root / name for name in negative_dir_names]

    missing = [path for path in positive_dirs + negative_dirs if not path.exists()]
    if missing:
        joined = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"以下数据目录不存在: {joined}")

    positive_list: List[str] = []
    negative_list: List[str] = []
    positive_counts: Dict[str, int] = {}
    negative_counts: Dict[str, int] = {}
    for folder in positive_dirs:
        names = list_filenames(folder)
        positive_list.extend([f"{folder.name}/{name}" for name in names])
        positive_counts[folder.name] = len(names)
    for folder in negative_dirs:
        names = list_filenames(folder)
        negative_list.extend([f"{folder.name}/{name}" for name in names])
        negative_counts[folder.name] = len(names)

    save_json(
        args.lists_output.resolve(),
        {
            "正样本目录": [folder.name for folder in positive_dirs],
            "负样本目录": [folder.name for folder in negative_dirs],
            "正样本各目录数量": positive_counts,
            "负样本各目录数量": negative_counts,
            "有目标序号列表": positive_list,
            "无对应目标列表": negative_list,
            "有目标样本数": len(positive_list),
            "无目标样本数": len(negative_list),
        },
    )

    entries = build_dataset(positive_dirs=positive_dirs, negative_dirs=negative_dirs)
    extractor = BrightnessFeatureExtractor(
        image_size=(args.image_size, args.image_size),
        grid_size=args.grid_size,
        bright_threshold=args.bright_threshold,
        red_dominance_delta=args.red_dominance_delta,
    )
    df = build_feature_table(entries=entries, extractor=extractor)
    df.to_csv(args.feature_output.resolve(), index=False, encoding="utf-8-sig")

    feature_cols = [col for col in df.columns if col not in {"filename", "folder", "label"}]
    train_df, test_df = train_test_split(
        df,
        test_size=args.test_size,
        stratify=df["label"],
        random_state=args.seed,
    )

    x_train = train_df[feature_cols].to_numpy(dtype=np.float32)
    y_train = train_df["label"].to_numpy(dtype=int)
    x_test = test_df[feature_cols].to_numpy(dtype=np.float32)
    y_test = test_df["label"].to_numpy(dtype=int)

    models = {
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        random_state=args.seed,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            random_state=args.seed,
            class_weight="balanced_subsample",
            n_jobs=-1,
        ),
    }

    evaluation_results: Dict[str, Dict[str, object]] = {}
    trained_models = {}
    for model_name, model in models.items():
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        evaluation_results[model_name] = evaluate_predictions(y_test, pred)
        trained_models[model_name] = model

    baseline_threshold = train_variance_threshold_baseline(train_df)
    baseline_pred = (test_df["block_mean_var"].to_numpy(dtype=np.float32) >= baseline_threshold).astype(int)
    evaluation_results["variance_threshold_baseline"] = evaluate_predictions(y_test, baseline_pred)
    evaluation_results["variance_threshold_baseline"]["threshold"] = baseline_threshold

    best_model_name = max(
        ["logistic_regression", "random_forest"],
        key=lambda name: (
            evaluation_results[name]["f1_score"],
            evaluation_results[name]["accuracy"],
        ),
    )
    best_model = trained_models[best_model_name]
    joblib.dump(
        {
            "model_name": best_model_name,
            "model": best_model,
            "feature_columns": feature_cols,
            "image_size": args.image_size,
            "grid_size": args.grid_size,
            "bright_threshold": args.bright_threshold,
            "red_dominance_delta": args.red_dominance_delta,
        },
        args.model_output.resolve(),
    )

    report_payload = {
        "task": "亮度区域二分类(0=无高亮,1=有高亮)",
        "pipeline": "图像 -> 预处理 -> 特征提取 -> 特征向量 -> 分类器训练 -> 预测",
        "dataset_dirs": {
            "positive": [folder.name for folder in positive_dirs],
            "negative": [folder.name for folder in negative_dirs],
        },
        "libraries": ["opencv-python(cv2)", "numpy", "pandas", "scikit-learn"],
        "preprocess": {
            "image_size": [args.image_size, args.image_size],
            "gray": True,
            "keep_rgb_features": True,
            "bright_threshold": args.bright_threshold,
            "red_dominance_delta": args.red_dominance_delta,
        },
        "feature_columns": feature_cols,
        "train_test_split": {
            "test_size": args.test_size,
            "seed": args.seed,
            "stratify": True,
            "train_samples": int(len(train_df)),
            "test_samples": int(len(test_df)),
        },
        "results": evaluation_results,
        "best_model": {
            "name": best_model_name,
            "metrics": evaluation_results[best_model_name],
            "saved_model_path": str(args.model_output.resolve()),
        },
        "artifacts": {
            "lists_output": str(args.lists_output.resolve()),
            "feature_output": str(args.feature_output.resolve()),
            "report_output": str(args.report_output.resolve()),
        },
    }
    save_json(args.report_output.resolve(), report_payload)

    print(
        f"正样本目录: {[folder.name for folder in positive_dirs]} | "
        f"负样本目录: {[folder.name for folder in negative_dirs]}"
    )
    print(f"样本总数: {len(df)} | 训练集: {len(train_df)} | 测试集: {len(test_df)}")
    print(f"特征维度: {len(feature_cols)}")
    for name in ["logistic_regression", "random_forest", "variance_threshold_baseline"]:
        metric = evaluation_results[name]
        print(
            f"{name}: accuracy={metric['accuracy']:.6f}, "
            f"precision={metric['precision']:.6f}, recall={metric['recall']:.6f}, "
            f"f1={metric['f1_score']:.6f}, confusion_matrix={metric['confusion_matrix']}"
        )
    print(f"最佳模型: {best_model_name}")
    print(f"模型已保存: {args.model_output.resolve()}")
    print(f"训练报告已保存: {args.report_output.resolve()}")


if __name__ == "__main__":
    main()
