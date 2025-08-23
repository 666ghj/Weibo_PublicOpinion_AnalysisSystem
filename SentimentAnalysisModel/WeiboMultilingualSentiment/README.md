# 多语言情感分析 - Multilingual Sentiment Analysis

本模块使用HuggingFace上的多语言情感分析模型进行情感分析，支持22种语言。

## 模型信息

- **模型名称**: tabularisai/multilingual-sentiment-analysis  
- **基础模型**: distilbert-base-multilingual-cased
- **支持语言**: 22种语言，包括：
  - 中文 (中文)
  - English (英语)
  - Español (西班牙语)
  - 日本語 (日语)
  - 한국어 (韩语)
  - Français (法语)
  - Deutsch (德语)
  - Русский (俄语)
  - العربية (阿拉伯语)
  - हिन्दी (印地语)
  - Português (葡萄牙语)
  - Italiano (意大利语)
  - 等等...

- **输出类别**: 5级情感分类
  - 非常负面 (Very Negative)
  - 负面 (Negative)
  - 中性 (Neutral)
  - 正面 (Positive)
  - 非常正面 (Very Positive)

## 快速开始

1. 确保已安装依赖：
```bash
pip install transformers torch
```

2. 运行预测程序：
```bash
python predict.py
```

3. 输入任意语言的文本进行分析：
```
请输入文本: I love this product!
预测结果: 非常正面 (置信度: 0.9456)
```

4. 查看多语言示例：
```
请输入文本: demo
```

## 代码示例

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 加载模型
model_name = "tabularisai/multilingual-sentiment-analysis"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# 预测
texts = [
    "今天心情很好",  # 中文
    "I love this!",  # 英文
    "¡Me encanta!"   # 西班牙文
]

for text in texts:
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    sentiment_map = {0: "非常负面", 1: "负面", 2: "中性", 3: "正面", 4: "非常正面"}
    print(f"{text} -> {sentiment_map[prediction]}")
```

## 特色功能

- **多语言支持**: 无需指定语言，自动识别22种语言
- **5级精细分类**: 比传统二分类更细致的情感分析
- **高精度**: 基于DistilBERT的先进架构
- **本地缓存**: 首次下载后保存到本地，加快后续使用

## 应用场景

- 国际社交媒体监控
- 多语言客户反馈分析
- 全球产品评论情感分类
- 跨语言品牌情感追踪
- 多语言客服优化
- 国际市场研究

## 模型存储

- 首次运行时会自动下载模型到当前目录的 `model` 文件夹
- 后续运行会直接从本地加载，无需重复下载
- 模型大小约135MB，首次下载需要网络连接

## 文件说明

- `predict.py`: 主预测程序，使用直接模型调用
- `README.md`: 使用说明

## 注意事项

- 首次运行时会自动下载模型，需要网络连接
- 模型会保存到当前目录，方便后续使用
- 支持GPU加速，会自动检测可用设备
- 如需清理模型文件，删除 `model` 文件夹即可
- 该模型基于合成数据训练，在实际应用中建议进行验证