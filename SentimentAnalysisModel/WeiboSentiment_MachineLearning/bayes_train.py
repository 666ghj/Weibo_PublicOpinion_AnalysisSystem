# -*- coding: utf-8 -*-
"""
朴素贝叶斯情感分析模型训练脚本
"""
import argparse
import pandas as pd
from typing import List, Tuple
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score

from base_model import BaseModel
from utils import stopwords


class BayesModel(BaseModel):
    """朴素贝叶斯情感分析模型"""
    
    def __init__(self):
        super().__init__("Bayes")
        
    def train(self, train_data: List[Tuple[str, int]], **kwargs) -> None:
        """训练朴素贝叶斯模型
        
        Args:
            train_data: 训练数据，格式为[(text, label), ...]
            **kwargs: 其他参数
        """
        print(f"开始训练 {self.model_name} 模型...")
        
        # 准备数据
        df_train = pd.DataFrame(train_data, columns=["words", "label"])
        
        # 特征编码（词袋模型）
        print("构建词袋模型...")
        self.vectorizer = CountVectorizer(
            token_pattern=r'\[?\w+\]?', 
            stop_words=stopwords
        )
        
        X_train = self.vectorizer.fit_transform(df_train["words"])
        y_train = df_train["label"]
        
        print(f"特征维度: {X_train.shape[1]}")
        
        # 训练模型
        print("训练朴素贝叶斯分类器...")
        self.model = MultinomialNB()
        self.model.fit(X_train, y_train)
        
        self.is_trained = True
        print(f"{self.model_name} 模型训练完成！")
        
    def predict(self, texts: List[str]) -> List[int]:
        """预测文本情感
        
        Args:
            texts: 待预测文本列表
            
        Returns:
            预测结果列表
        """
        if not self.is_trained:
            raise ValueError(f"模型 {self.model_name} 尚未训练，请先调用train方法")
            
        # 特征转换
        X = self.vectorizer.transform(texts)
        
        # 预测
        predictions = self.model.predict(X)
        
        return predictions.tolist()
    
    def predict_single(self, text: str) -> Tuple[int, float]:
        """预测单条文本的情感
        
        Args:
            text: 待预测文本
            
        Returns:
            (predicted_label, confidence)
        """
        if not self.is_trained:
            raise ValueError(f"模型 {self.model_name} 尚未训练，请先调用train方法")
            
        # 特征转换
        X = self.vectorizer.transform([text])
        
        # 预测
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        confidence = max(probabilities)
        
        return int(prediction), float(confidence)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='朴素贝叶斯情感分析模型训练')
    parser.add_argument('--train_path', type=str, default='./data/weibo2018/train.txt',
                        help='训练数据路径')
    parser.add_argument('--test_path', type=str, default='./data/weibo2018/test.txt',
                        help='测试数据路径')
    parser.add_argument('--model_path', type=str, default='./model/bayes_model.pkl',
                        help='模型保存路径')
    parser.add_argument('--eval_only', action='store_true',
                        help='仅评估已有模型，不进行训练')
    
    args = parser.parse_args()
    
    # 创建模型
    model = BayesModel()
    
    if args.eval_only:
        # 仅评估模式
        print("评估模式：加载已有模型进行评估")
        model.load_model(args.model_path)
        
        # 加载测试数据
        _, test_data = BaseModel.load_data(args.train_path, args.test_path)
        
        # 评估模型
        model.evaluate(test_data)
    else:
        # 训练模式
        # 加载数据
        train_data, test_data = BaseModel.load_data(args.train_path, args.test_path)
        
        # 训练模型
        model.train(train_data)
        
        # 评估模型
        model.evaluate(test_data)
        
        # 保存模型
        model.save_model(args.model_path)
        
        # 示例预测
        print("\n示例预测:")
        test_texts = [
            "今天天气真好，心情很棒",
            "这部电影太无聊了，浪费时间",
            "哈哈哈，太有趣了"
        ]
        
        for text in test_texts:
            pred, conf = model.predict_single(text)
            sentiment = "正面" if pred == 1 else "负面"
            print(f"文本: {text}")
            print(f"预测: {sentiment} (置信度: {conf:.4f})")
            print()


if __name__ == "__main__":
    main()