# 高亮区域检测模型整理包

## 目录结构
- code: 主要代码
- data: 训练数据(light / without_light)
- artifacts: 训练产物(模型、报告、特征表、序号列表)
- docs: 任务说明与提示词

## 快速使用
1. 训练: `python code\train_brightness_variance_model.py --data-root .`
2. 预测: `python code\predict_highlight.py --image data\light\1.jpg --model artifacts\最佳高亮分类器.joblib`
