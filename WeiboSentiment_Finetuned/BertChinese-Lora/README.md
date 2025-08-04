# 微博情感分析 - HuggingFace预训练模型

本模块使用HuggingFace上的预训练微博情感分析模型进行情感分析。

## 模型信息

- **模型名称**: wsqstar/GISchat-weibo-100k-fine-tuned-bert  
- **模型类型**: BERT中文情感分类模型
- **训练数据**: 10万条微博数据
- **输出**: 二分类（正面/负面情感）

## 使用方法

### 方法1: 直接模型调用 (推荐)
```bash
python predict.py
```

### 方法2: Pipeline方式
```bash
python predict_pipeline.py
```

## 快速开始

1. 确保已安装依赖：
```bash
pip install transformers torch
```

2. 运行预测程序：
```bash
python predict.py
```

3. 输入微博文本进行分析：
```
请输入微博内容: 今天天气真好，心情特别棒！
预测结果: 正面情感 (置信度: 0.9234)
```

## 代码示例

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 加载模型
model_name = "wsqstar/GISchat-weibo-100k-fine-tuned-bert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# 预测
text = "今天心情很好"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
prediction = torch.argmax(outputs.logits, dim=1).item()
print("正面情感" if prediction == 1 else "负面情感")
```

## 文件说明

- `predict.py`: 主预测程序，使用直接模型调用
- `predict_pipeline.py`: 使用pipeline方式的预测程序  
- `README.md`: 使用说明

## 注意事项

- 首次运行时会自动下载模型，需要网络连接
- 模型大小约400MB，下载可能需要一些时间
- 支持GPU加速，会自动检测可用设备