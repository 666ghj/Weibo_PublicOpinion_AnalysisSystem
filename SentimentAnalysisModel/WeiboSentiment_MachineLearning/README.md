# 微博情感分析 - 传统机器学习方法

## 项目介绍

本项目使用5种传统机器学习方法对中文微博进行情感二分类（正面/负面）：

- **朴素贝叶斯**: 基于词袋模型的概率分类
- **SVM**: 基于TF-IDF特征的支持向量机  
- **XGBoost**: 梯度提升决策树
- **LSTM**: 循环神经网络 + Word2Vec词向量
- **BERT+分类头**: 预训练语言模型接分类器（我认为也属于传统ML范畴）

## 模型性能

在微博情感数据集上的表现（训练集10000条，测试集500条）：

| 模型 | 准确率 | AUC | 特点 |
|------|--------|-----|------|
| 朴素贝叶斯 | 85.6% | - | 速度快，内存占用小 |
| SVM | 85.6% | - | 泛化能力好 |
| XGBoost | 86.0% | 90.4% | 性能稳定，支持特征重要性 |
| LSTM | 87.0% | 93.1% | 理解序列信息和上下文 |
| BERT+分类头 | 87.0% | 92.9% | 强大的语义理解能力 |

## 环境配置

```bash
pip install -r requirements.txt
```

数据文件结构：
```
data/
├── weibo2018/
│   ├── train.txt
│   └── test.txt
└── stopwords.txt
```

## 训练模型（后面可以不接参数直接运行）

### 朴素贝叶斯
```bash
python bayes_train.py
```

### SVM
```bash
python svm_train.py --kernel rbf --C 1.0
```

### XGBoost
```bash
python xgboost_train.py --max_depth 6 --eta 0.3 --num_boost_round 200
```

### LSTM
```bash
python lstm_train.py --epochs 5 --batch_size 100 --hidden_size 64
```

### BERT
```bash
python bert_train.py --epochs 10 --batch_size 100 --learning_rate 1e-3
```

注：BERT模型会自动下载中文预训练模型（bert-base-chinese）

## 使用预测

### 交互式预测（推荐）
```bash
python predict.py
```

### 命令行预测
```bash
# 单模型预测
python predict.py --model_type bert --text "今天天气真好，心情很棒"

# 多模型集成预测
python predict.py --ensemble --text "这部电影太无聊了"
```

## 文件结构

```
WeiboSentiment_MachineLearning/
├── bayes_train.py           # 朴素贝叶斯训练
├── svm_train.py             # SVM训练
├── xgboost_train.py         # XGBoost训练
├── lstm_train.py            # LSTM训练
├── bert_train.py            # BERT训练
├── predict.py               # 统一预测程序
├── base_model.py            # 基础模型类
├── utils.py                 # 工具函数
├── requirements.txt         # 依赖包
├── model/                   # 模型保存目录
└── data/                    # 数据目录
```

## 注意事项

1. **BERT模型**首次运行会自动下载预训练模型（约400MB）
2. **LSTM模型**训练时间较长，建议使用GPU
3. **模型保存**在 `model/` 目录下，确保有足够磁盘空间
4. **内存需求**BERT > LSTM > XGBoost > SVM > 朴素贝叶斯
