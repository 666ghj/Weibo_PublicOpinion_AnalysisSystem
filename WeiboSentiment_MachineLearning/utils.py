# -*- coding: utf-8 -*-
import jieba
import re


stopwords = []
with open("data/stopwords.txt", "r", encoding="utf8") as f:
    for w in f:
        stopwords.append(w.strip())


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