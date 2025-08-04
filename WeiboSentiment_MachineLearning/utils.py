# -*- coding: utf-8 -*-
import jieba
import re
import os
import pickle
from typing import List, Tuple, Any


# 加载停用词
stopwords = []
stopwords_path = "data/stopwords.txt"
if os.path.exists(stopwords_path):
    with open(stopwords_path, "r", encoding="utf8") as f:
        for w in f:
            stopwords.append(w.strip())
else:
    print(f"警告: 停用词文件 {stopwords_path} 不存在，将使用空停用词列表")


def load_corpus(path):
    """
    加载语料库
    """
    data = []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            [_, seniment, content] = line.split(",", 2)
            content = processing(content)
            data.append((content, int(seniment)))
    return data


def load_corpus_bert(path):
    """
    加载语料库
    """
    data = []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            [_, seniment, content] = line.split(",", 2)
            content = processing_bert(content)
            data.append((content, int(seniment)))
    return data


def processing(text):
    """
    数据预处理, 可以根据自己的需求进行重载
    """
    # 数据清洗部分
    text = re.sub("\{%.+?%\}", " ", text)           # 去除 {%xxx%} (地理定位, 微博话题等)
    text = re.sub("@.+?( |$)", " ", text)           # 去除 @xxx (用户名)
    text = re.sub("【.+?】", " ", text)              # 去除 【xx】 (里面的内容通常都不是用户自己写的)
    text = re.sub("\u200b", " ", text)              # '\u200b'是这个数据集中的一个bad case, 不用特别在意
    # 分词
    words = [w for w in jieba.lcut(text) if w.isalpha()]
    # 对否定词`不`做特殊处理: 与其后面的词进行拼接
    while "不" in words:
        index = words.index("不")
        if index == len(words) - 1:
            break
        words[index: index+2] = ["".join(words[index: index+2])]  # 列表切片赋值的酷炫写法
    # 用空格拼接成字符串
    result = " ".join(words)
    return result


def processing_bert(text):
    """
    数据预处理, 可以根据自己的需求进行重载
    """
    # 数据清洗部分
    text = re.sub("\{%.+?%\}", " ", text)           # 去除 {%xxx%} (地理定位, 微博话题等)
    text = re.sub("@.+?( |$)", " ", text)           # 去除 @xxx (用户名)
    text = re.sub("【.+?】", " ", text)              # 去除 【xx】 (里面的内容通常都不是用户自己写的)
    text = re.sub("\u200b", " ", text)              # '\u200b'是这个数据集中的一个bad case, 不用特别在意
    return text


def save_model(model: Any, model_path: str) -> None:
    """
    保存模型到文件
    
    Args:
        model: 要保存的模型对象
        model_path: 保存路径
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"模型已保存到: {model_path}")


def load_model(model_path: str) -> Any:
    """
    从文件加载模型
    
    Args:
        model_path: 模型文件路径
        
    Returns:
        加载的模型对象
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print(f"已加载模型: {model_path}")
    return model


def preprocess_text_simple(text: str) -> str:
    """
    简单的文本预处理函数，用于预测时的文本清洗
    
    Args:
        text: 原始文本
        
    Returns:
        清洗后的文本
    """
    # 数据清洗
    text = re.sub("\{%.+?%\}", " ", text)           # 去除 {%xxx%}
    text = re.sub("@.+?( |$)", " ", text)           # 去除 @xxx
    text = re.sub("【.+?】", " ", text)              # 去除 【xx】
    text = re.sub("\u200b", " ", text)              # 去除特殊字符
    
    # 删除表情符号
    text = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002600-\U000027BF\U0001f900-\U0001f9ff\U0001f018-\U0001f270\U0000231a-\U0000231b\U0000238d-\U0000238d\U000024c2-\U0001f251]+', '', text)
    
    # 多个空格合并为一个
    text = re.sub(r"\s+", " ", text)
    
    return text.strip()