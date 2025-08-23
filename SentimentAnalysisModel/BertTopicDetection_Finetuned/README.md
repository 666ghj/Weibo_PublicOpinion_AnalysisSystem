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

### 可选中文基座模型（训练前交互选择）

默认基座：`google-bert/bert-base-chinese`。启动训练时，若终端可交互，程序会提示从下列选项中选择（或输入任意 Hugging Face 模型 ID）：

1) `google-bert/bert-base-chinese`
2) `hfl/chinese-roberta-wwm-ext-large`
3) `hfl/chinese-macbert-large`
4) `IDEA-CCNL/Erlangshen-DeBERTa-v2-710M-Chinese`
5) `IDEA-CCNL/Erlangshen-DeBERTa-v3-Base-Chinese`
6) `Langboat/mengzi-bert-base`
7) `BAAI/bge-base-zh`（更适合检索式/对比学习范式）
8) `nghuyong/ernie-3.0-base-zh`

说明：
- 非交互环境（如调度系统）或设置 `NON_INTERACTIVE=1` 时，会直接使用命令行参数 `--pretrained_name` 指定的模型（默认为 `google-bert/bert-base-chinese`）。
- 选择后，基础模型将下载/缓存至 `model/` 目录，统一管理。

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


### 作者说明（关于超大规模多分类）

- 当话题类别达到上万级时，直接在编码器后接单一线性分类头（大 softmax）往往受限：长尾类别难学、语义稀疏、新增话题无法增量适配、上线后需频繁重训。
- 改进思路（推荐优先级）：
  - 检索式/双塔范式（文本 vs. 话题名称/描述 对比学习）+ 近邻检索 + 小头重排，天然支持增量扩类与快速更新；
  - 分层分类（先粗分再细分），显著降低单头难度与计算；
  - 文本-标签联合建模（使用标签描述），提升近义话题的可迁移性；
  - 训练细节：class-balanced/focal/label smoothing、sampled softmax、对比预训练等。
- 重要声明：本目录使用的“静态分类头微调”仅作为备选与学习参考。对于英文/多语微短文场景，话题变化极快，传统静态分类器难以及时覆盖，我们的工作重点在 `TopicGPT` 等生成式/自监督话题发现与动态体系构建方向；本实现旨在提供一个可运行的基线与工程示例。


