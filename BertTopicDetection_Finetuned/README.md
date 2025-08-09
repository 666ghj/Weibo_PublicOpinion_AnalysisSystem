## 话题分类（BERT 中文基座）

本目录提供一个使用 `google-bert/bert-base-chinese` 的中文话题分类实现：
- 自动处理本地/缓存/远程三段式加载逻辑；
- `train.py` 进行微调训练；`predict.py` 进行单条或交互式预测；
- 所有模型与权重统一保存至本目录的 `model/`。

参考模型卡片： [google-bert/bert-base-chinese](https://huggingface.co/google-bert/bert-base-chinese)

### 数据集亮点

- 约 **410 万**条预过滤高质量问题与回复；
- 每个问题对应一个“【话题】”，覆盖 **约 2.8 万**个多样主题；
- 从 **1400 万**原始问答中筛选，保留至少 **3 个点赞以上**的答案，确保内容质量与有趣度；
- 除了问题、话题与一个或多个回复外，每个回复还带有点赞数、回复 ID、回复者标签；
- 数据清洗去重后划分三部分：示例划分训练集约 **412 万**、验证/测试若干（可按需调整）。

> 实际训练时，请以 `dataset/` 下的 CSV 为准；脚本会自动识别常见列名或允许通过命令参数显式指定。

### 目录结构

```
BertTopicDetection_Finetuned/
  ├─ dataset/                   # 已放置数据
  ├─ model/                     # 训练生成；亦缓存基础 BERT
  ├─ train.py
  ├─ predict.py
  └─ README.md
```

### 环境

```
pip install torch transformers scikit-learn pandas
```

或使用你既有的 Conda 环境。

### 数据格式

CSV 至少包含文本列与标签列，脚本会尝试自动识别：
- 文本列候选：`text`/`content`/`sentence`/`title`/`desc`/`question`
- 标签列候选：`label`/`labels`/`category`/`topic`/`class`

如需显式指定，请使用 `--text_col` 与 `--label_col`。

### 训练

```
python train.py \
  --train_file ./dataset/web_text_zh_train.csv \
  --valid_file ./dataset/web_text_zh_valid.csv \
  --text_col auto \
  --label_col auto \
  --model_root ./model \
  --save_subdir bert-chinese-classifier \
  --num_epochs 10 --batch_size 16 --learning_rate 2e-5 --fp16
```

要点：
- 首次运行会检查 `model/bert-base-chinese`；若无则尝试本机缓存，再不行则自动下载并保存；
- 训练过程按步评估与保存（默认每 1/4 个 epoch），最多保留 5 个最近 checkpoint（可通过环境变量 `SAVE_TOTAL_LIMIT` 调整）；
- 支持早停（默认耐心 5 次评估），并在评估/保存策略一致时自动回滚到最佳模型；
- 分词器、权重与 `label_map.json` 保存到 `model/bert-chinese-classifier/`。

### 预测

单条：
```
python predict.py --text "这条微博讨论的是哪个话题？" --model_root ./model --finetuned_subdir bert-chinese-classifier
```

交互：
```
python predict.py --interactive --model_root ./model --finetuned_subdir bert-chinese-classifier
```

示例输出：
```
预测结果: 体育-足球 (置信度: 0.9412)
```

### 说明

- 训练与预测均内置简易中文文本清洗。
- 标签集合以训练集为准，脚本自动生成并保存 `label_map.json`。

### 训练策略（简述）

- 基座：`google-bert/bert-base-chinese`；分类头维度=训练集唯一标签数。
- 学习率与正则：`lr=2e-5`，`weight_decay=0.01`，可在大型数据上微调到 `1e-5~3e-5`。
- 序列长度与批量：`max_length=128`，`batch_size=16`；若截断严重可升至 256（成本上升）。
- Warmup：若环境支持，使用 `warmup_ratio=0.1`；否则回退 `warmup_steps=0`。
- 评估/保存：按 `--eval_fraction` 折算步数（默认 0.25），`save_total_limit=5` 限制磁盘占用。
- 早停：监控加权 F1（越大越好），默认耐心 5、改善阈值 0.0。
- 单卡稳定运行：默认仅使用一张 GPU，可通过 `--gpu` 指定；脚本会清理分布式环境变量。


