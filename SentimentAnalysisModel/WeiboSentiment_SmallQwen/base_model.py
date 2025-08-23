# -*- coding: utf-8 -*-
"""
Qwen3模型基础类，统一接口
"""
import os
import pickle
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split


class BaseQwenModel(ABC):
    """Qwen3情感分析模型基类"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.is_trained = False
        
    @abstractmethod
    def train(self, train_data: List[Tuple[str, int]], **kwargs) -> None:
        """训练模型"""
        pass
    
    @abstractmethod
    def predict(self, texts: List[str]) -> List[int]:
        """预测文本情感"""
        pass
    
    def predict_single(self, text: str) -> Tuple[int, float]:
        """预测单条文本的情感
        
        Args:
            text: 待预测文本
            
        Returns:
            (predicted_label, confidence)
        """
        predictions = self.predict([text])
        return predictions[0], 0.0  # 默认置信度为0
    
    def evaluate(self, test_data: List[Tuple[str, int]]) -> Dict[str, float]:
        """评估模型性能"""
        if not self.is_trained:
            raise ValueError(f"模型 {self.model_name} 尚未训练，请先调用train方法")
            
        texts = [item[0] for item in test_data]
        labels = [item[1] for item in test_data]
        
        predictions = self.predict(texts)
        
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='weighted')
        
        print(f"\n{self.model_name} 模型评估结果:")
        print(f"准确率: {accuracy:.4f}")
        print(f"F1分数: {f1:.4f}")
        print("\n详细报告:")
        print(classification_report(labels, predictions))
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'classification_report': classification_report(labels, predictions)
        }
    
    @abstractmethod
    def save_model(self, model_path: str = None) -> None:
        """保存模型到文件"""
        pass
    
    @abstractmethod
    def load_model(self, model_path: str) -> None:
        """从文件加载模型"""
        pass
    
    @staticmethod
    def load_data(train_path: str = None, test_path: str = None, csv_path: str = 'dataset/weibo_senti_100k.csv') -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
        """加载训练和测试数据
        
        Args:
            train_path: 训练数据txt文件路径（可选）
            test_path: 测试数据txt文件路径（可选）
            csv_path: CSV数据文件路径（默认使用）
        """
        
        # 优先尝试使用CSV文件
        if os.path.exists(csv_path):
            print(f"从CSV文件加载数据: {csv_path}")
            df = pd.read_csv(csv_path)
            
            # 检查数据格式
            if 'review' in df.columns and 'label' in df.columns:
                # 将DataFrame转换为元组列表
                data = [(row['review'], row['label']) for _, row in df.iterrows()]
                
                # 分割训练和测试数据，固定测试集为5000条
                total_samples = len(data)
                if total_samples > 5000:
                    test_size = 5000
                    train_data, test_data = train_test_split(
                        data, 
                        test_size=test_size, 
                        random_state=42, 
                        stratify=[label for _, label in data]
                    )
                else:
                    # 如果总数据不足5000条，使用20%作为测试集
                    train_data, test_data = train_test_split(
                        data, 
                        test_size=0.2, 
                        random_state=42, 
                        stratify=[label for _, label in data]
                    )
                
                print(f"训练数据量: {len(train_data)}")
                print(f"测试数据量: {len(test_data)}")
                
                return train_data, test_data
            else:
                print(f"CSV文件格式不正确，缺少'review'或'label'列")
        
        # 如果CSV不存在，尝试使用txt文件
        elif train_path and test_path and os.path.exists(train_path) and os.path.exists(test_path):
            def load_corpus(path):
                data = []
                with open(path, "r", encoding="utf8") as f:
                    for line in f:
                        parts = line.strip().split("\t")
                        if len(parts) >= 2:
                            content = parts[0]
                            sentiment = int(parts[1])
                            data.append((content, sentiment))
                return data
            
            print("从txt文件加载训练数据...")
            train_data = load_corpus(train_path)
            print(f"训练数据量: {len(train_data)}")
            
            print("从txt文件加载测试数据...")
            test_data = load_corpus(test_path)
            print(f"测试数据量: {len(test_data)}")
            
            return train_data, test_data
        
        else:
            # 如果都没有，提供样例数据创建指导
            print("未找到数据文件!")
            print("请确保以下文件之一存在:")
            print(f"1. CSV文件: {csv_path}")
            print(f"2. txt文件: {train_path} 和 {test_path}")
            print("\n数据格式要求:")
            print("CSV文件: 包含'review'和'label'列")
            print("txt文件: 每行格式为'文本内容\\t标签'")
            
            # 创建样例数据
            sample_data = [
                ("今天天气真好，心情很棒!", 1),
                ("这部电影太无聊了", 0),
                ("非常喜欢这个产品", 1),
                ("服务态度很差", 0),
                ("质量不错，值得推荐", 1)
            ]
            
            print("使用样例数据进行演示...")
            train_data = sample_data * 20  # 扩充样例数据
            test_data = sample_data * 5
            
            return train_data, test_data