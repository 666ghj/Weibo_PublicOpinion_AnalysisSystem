# -*- coding: utf-8 -*-
"""
XGBoost情感分析模型训练脚本
"""
import argparse
import pandas as pd
import numpy as np
from typing import List, Tuple
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import xgboost as xgb

from base_model import BaseModel
from utils import stopwords


class XGBoostModel(BaseModel):
    """XGBoost情感分析模型"""
    
    def __init__(self):
        super().__init__("XGBoost")
        
    def train(self, train_data: List[Tuple[str, int]], **kwargs) -> None:
        """训练XGBoost模型
        
        Args:
            train_data: 训练数据，格式为[(text, label), ...]
            **kwargs: 其他参数，支持XGBoost的各种参数
        """
        print(f"开始训练 {self.model_name} 模型...")
        
        # 准备数据
        df_train = pd.DataFrame(train_data, columns=["words", "label"])
        
        # 特征编码（词袋模型，限制特征数量）
        max_features = kwargs.get('max_features', 2000)
        print(f"构建词袋模型 (max_features={max_features})...")
        self.vectorizer = CountVectorizer(
            token_pattern=r'\[?\w+\]?', 
            stop_words=stopwords,
            max_features=max_features
        )
        
        X_train = self.vectorizer.fit_transform(df_train["words"])
        y_train = df_train["label"]
        
        print(f"特征维度: {X_train.shape[1]}")
        
        # XGBoost参数设置
        params = {
            'booster': kwargs.get('booster', 'gbtree'),
            'max_depth': kwargs.get('max_depth', 6),
            'scale_pos_weight': kwargs.get('scale_pos_weight', 0.5),
            'colsample_bytree': kwargs.get('colsample_bytree', 0.8),
            'objective': 'binary:logistic',
            'eval_metric': 'error',
            'eta': kwargs.get('eta', 0.3),
            'nthread': kwargs.get('nthread', 10),
        }
        
        num_boost_round = kwargs.get('num_boost_round', 200)
        
        print(f"训练XGBoost分类器...")
        print(f"参数: {params}")
        print(f"迭代轮数: {num_boost_round}")
        
        # 创建DMatrix
        dmatrix = xgb.DMatrix(X_train, label=y_train)
        
        # 训练模型
        self.model = xgb.train(params, dmatrix, num_boost_round=num_boost_round)
        
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
        
        # 创建DMatrix
        dmatrix = xgb.DMatrix(X)
        
        # 预测概率
        y_prob = self.model.predict(dmatrix)
        
        # 转换为类别标签
        y_pred = (y_prob > 0.5).astype(int)
        
        return y_pred.tolist()
    
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
        
        # 创建DMatrix
        dmatrix = xgb.DMatrix(X)
        
        # 预测概率
        prob = self.model.predict(dmatrix)[0]
        
        # 转换为类别标签和置信度
        prediction = int(prob > 0.5)
        confidence = prob if prediction == 1 else 1 - prob
        
        return prediction, float(confidence)
    
    def evaluate(self, test_data: List[Tuple[str, int]]) -> dict:
        """评估模型性能，包含AUC指标"""
        if not self.is_trained:
            raise ValueError(f"模型 {self.model_name} 尚未训练，请先调用train方法")
            
        texts = [item[0] for item in test_data]
        labels = [item[1] for item in test_data]
        
        # 预测类别
        predictions = self.predict(texts)
        
        # 预测概率（用于计算AUC）
        X = self.vectorizer.transform(texts)
        dmatrix = xgb.DMatrix(X)
        probabilities = self.model.predict(dmatrix)
        
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='weighted')
        auc = roc_auc_score(labels, probabilities)
        
        print(f"\n{self.model_name} 模型评估结果:")
        print(f"准确率: {accuracy:.4f}")
        print(f"F1分数: {f1:.4f}")
        print(f"AUC: {auc:.4f}")
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'auc': auc
        }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='XGBoost情感分析模型训练')
    parser.add_argument('--train_path', type=str, default='./data/weibo2018/train.txt',
                        help='训练数据路径')
    parser.add_argument('--test_path', type=str, default='./data/weibo2018/test.txt',
                        help='测试数据路径')
    parser.add_argument('--model_path', type=str, default='./model/xgboost_model.pkl',
                        help='模型保存路径')
    parser.add_argument('--max_features', type=int, default=2000,
                        help='最大特征数量')
    parser.add_argument('--max_depth', type=int, default=6,
                        help='XGBoost最大深度')
    parser.add_argument('--eta', type=float, default=0.3,
                        help='XGBoost学习率')
    parser.add_argument('--num_boost_round', type=int, default=200,
                        help='XGBoost迭代轮数')
    parser.add_argument('--eval_only', action='store_true',
                        help='仅评估已有模型，不进行训练')
    
    args = parser.parse_args()
    
    # 创建模型
    model = XGBoostModel()
    
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
        model.train(
            train_data,
            max_features=args.max_features,
            max_depth=args.max_depth,
            eta=args.eta,
            num_boost_round=args.num_boost_round
        )
        
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