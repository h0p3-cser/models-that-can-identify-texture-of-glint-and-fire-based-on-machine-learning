import argparse
from pathlib import Path

import joblib

from train_brightness_variance_model import BrightnessFeatureExtractor


def main() -> None:
    parser = argparse.ArgumentParser(description="使用已训练模型预测图片是否存在高亮区域")
    parser.add_argument("--image", type=Path, required=True, help="待预测图片路径")
    parser.add_argument("--model", type=Path, default=Path("最佳高亮分类器.joblib"), help="模型文件路径")
    args = parser.parse_args()

    model_bundle = joblib.load(args.model.resolve())
    model = model_bundle["model"]
    feature_columns = model_bundle["feature_columns"]
    extractor = BrightnessFeatureExtractor(
        image_size=(model_bundle["image_size"], model_bundle["image_size"]),
        grid_size=model_bundle["grid_size"],
        bright_threshold=model_bundle["bright_threshold"],
        red_dominance_delta=model_bundle.get("red_dominance_delta", 30),
    )

    features = extractor.extract(args.image.resolve())
    x = [[features[name] for name in feature_columns]]
    prediction = int(model.predict(x)[0])
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(x)[0][1])
    else:
        prob = None

    print(f"image: {args.image.resolve()}")
    print(f"prediction: {prediction} (1=有高亮区域, 0=无高亮区域)")
    if prob is not None:
        print(f"probability_of_highlight: {prob:.6f}")


if __name__ == "__main__":
    main()
