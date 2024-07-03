import pandas as pd  # 用于数据处理
import numpy as np  # 用于科学计算
import csv  # 用于读取CSV文件
from snownlp import SnowNLP  # 用于中文自然语言处理（此处未实际使用）
from sklearn.feature_extraction.text import TfidfVectorizer  # 用于文本特征提取
from sklearn.naive_bayes import MultinomialNB  # 用于多项式朴素贝叶斯分类
from sklearn.model_selection import train_test_split  # 用于划分训练集和测试集
from sklearn.metrics import accuracy_score  # 用于计算模型准确度

def getSentiment_data():
    # 从CSV文件中读取情感数据
    sentiment_data = []
    with open('./target.csv', 'r', encoding='utf8') as readerFile:
        reader = csv.reader(readerFile)
        for data in reader:
            sentiment_data.append(data)
    return sentiment_data

def model_train():
    # 获取情感数据并转换为DataFrame
    sentiment_data = getSentiment_data()
    df = pd.DataFrame(sentiment_data, columns=['text', 'sentiment'])

    # 将数据集划分为训练集和测试集，测试集占20%
    train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

    # 初始化TfidfVectorizer，并对训练集和测试集进行文本特征提取
    vectorize = TfidfVectorizer()
    X_train = vectorize.fit_transform(train_data['text'])
    y_train = train_data['sentiment']
    X_test = vectorize.transform(test_data['text'])
    y_test = test_data['sentiment']

    # 初始化多项式朴素贝叶斯分类器，并进行训练
    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)

    # 对测试集进行预测
    y_pred = classifier.predict(X_test)

    # 计算模型准确度
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)

if __name__ == "__main__":
    model_train()  # 训练模型并计算准确度
