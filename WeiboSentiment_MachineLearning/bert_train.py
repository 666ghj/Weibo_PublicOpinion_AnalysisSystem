# -*- coding: utf-8 -*-
"""
BERT情感分析模型训练脚本
"""
import argparse
import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score
from typing import List, Tuple
import warnings
import requests
from pathlib import Path

from base_model import BaseModel
from utils import load_corpus_bert

# 忽略transformers的警告
warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class BertDataset(Dataset):
    """BERT数据集"""
    
    def __init__(self, data: List[Tuple[str, int]]):
        self.data = [item[0] for item in data]
        self.labels = [item[1] for item in data]
    
    def __getitem__(self, index):
        return self.data[index], self.labels[index]
    
    def __len__(self):
        return len(self.labels)


class BertClassifier(nn.Module):
    """BERT分类器网络"""
    
    def __init__(self, input_size):
        super(BertClassifier, self).__init__()
        self.fc = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        out = self.fc(x)
        out = self.sigmoid(out)
        return out


class BertModel_Custom(BaseModel):
    """BERT情感分析模型"""
    
    def __init__(self, model_path: str = "./model/chinese_wwm_pytorch"):
        super().__init__("BERT")
        self.model_path = model_path
        self.tokenizer = None
        self.bert = None
        self.classifier = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def _download_bert_model(self):
        """自动下载BERT预训练模型"""
        print(f"BERT模型不存在，正在下载中文BERT预训练模型...")
        print("下载来源: bert-base-chinese (Hugging Face)")
        
        try:
            # 创建模型目录
            os.makedirs(self.model_path, exist_ok=True)
            
            # 使用Hugging Face的中文BERT模型
            model_name = "bert-base-chinese"
            print(f"正在从Hugging Face下载 {model_name}...")
            
            # 下载tokenizer
            print("下载分词器...")
            tokenizer = BertTokenizer.from_pretrained(model_name)
            tokenizer.save_pretrained(self.model_path)
            
            # 下载模型
            print("下载BERT模型...")
            bert_model = BertModel.from_pretrained(model_name)
            bert_model.save_pretrained(self.model_path)
            
            print(f"✅ BERT模型下载完成，保存在: {self.model_path}")
            return True
            
        except Exception as e:
            print(f"❌ BERT模型下载失败: {e}")
            print("\n💡 您可以手动下载BERT模型:")
            print("1. 访问 https://huggingface.co/bert-base-chinese")
            print("2. 或使用哈工大中文BERT: https://github.com/ymcui/Chinese-BERT-wwm")
            print(f"3. 将模型文件解压到: {self.model_path}")
            return False
    
    def _load_bert(self):
        """加载BERT模型和分词器"""
        print(f"加载BERT模型: {self.model_path}")
        
        # 如果模型不存在，尝试自动下载
        if not os.path.exists(self.model_path) or not any(os.scandir(self.model_path)):
            print("BERT模型不存在，尝试自动下载...")
            if not self._download_bert_model():
                raise FileNotFoundError(f"BERT模型下载失败，请手动下载到: {self.model_path}")
        
        try:
            self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
            self.bert = BertModel.from_pretrained(self.model_path).to(self.device)
            
            # 冻结BERT参数
            for param in self.bert.parameters():
                param.requires_grad = False
                
            print("✅ BERT模型加载完成")
            
        except Exception as e:
            print(f"❌ BERT模型加载失败: {e}")
            print("尝试使用在线模型...")
            
            # 如果本地加载失败，尝试直接使用在线模型
            try:
                model_name = "bert-base-chinese"
                self.tokenizer = BertTokenizer.from_pretrained(model_name)
                self.bert = BertModel.from_pretrained(model_name).to(self.device)
                
                # 冻结BERT参数
                for param in self.bert.parameters():
                    param.requires_grad = False
                    
                print("✅ 在线BERT模型加载完成")
                
            except Exception as e2:
                print(f"❌ 在线模型也加载失败: {e2}")
                raise FileNotFoundError(f"无法加载BERT模型，请检查网络连接或手动下载模型到: {self.model_path}")
    
    def train(self, train_data: List[Tuple[str, int]], **kwargs) -> None:
        """训练BERT模型"""
        print(f"开始训练 {self.model_name} 模型...")
        
        # 加载BERT
        self._load_bert()
        
        # 超参数
        learning_rate = kwargs.get('learning_rate', 1e-3)
        num_epochs = kwargs.get('num_epochs', 10)
        batch_size = kwargs.get('batch_size', 100)
        input_size = kwargs.get('input_size', 768)
        decay_rate = kwargs.get('decay_rate', 0.9)
        
        print(f"BERT超参数: lr={learning_rate}, epochs={num_epochs}, "
              f"batch_size={batch_size}, input_size={input_size}")
        
        # 创建数据集
        train_dataset = BertDataset(train_data)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # 创建分类器
        self.classifier = BertClassifier(input_size).to(self.device)
        
        # 损失函数和优化器
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        
        # 训练循环
        self.bert.eval()  # BERT始终保持评估模式
        self.classifier.train()
        
        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0
            
            for i, (words, labels) in enumerate(train_loader):
                # 分词和编码
                tokens = self.tokenizer(words, padding=True, truncation=True, 
                                      max_length=512, return_tensors='pt')
                input_ids = tokens["input_ids"].to(self.device)
                attention_mask = tokens["attention_mask"].to(self.device)
                labels = torch.tensor(labels, dtype=torch.float32).to(self.device)
                
                # 获取BERT输出（冻结参数）
                with torch.no_grad():
                    bert_outputs = self.bert(input_ids, attention_mask=attention_mask)
                    bert_output = bert_outputs[0][:, 0]  # [CLS] token的输出
                
                # 分类器前向传播
                optimizer.zero_grad()
                outputs = self.classifier(bert_output)
                logits = outputs.view(-1)
                loss = criterion(logits, labels)
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                if (i + 1) % 10 == 0:
                    avg_loss = total_loss / num_batches
                    print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}], Loss: {avg_loss:.4f}")
                    total_loss = 0
                    num_batches = 0
            
            # 学习率衰减
            scheduler.step()
            
            # 保存每个epoch的模型
            if kwargs.get('save_each_epoch', False):
                epoch_model_path = f"./model/bert_epoch_{epoch+1}.pth"
                os.makedirs(os.path.dirname(epoch_model_path), exist_ok=True)
                torch.save(self.classifier.state_dict(), epoch_model_path)
                print(f"已保存模型: {epoch_model_path}")
        
        self.is_trained = True
        print(f"{self.model_name} 模型训练完成！")
    
    def predict(self, texts: List[str]) -> List[int]:
        """预测文本情感"""
        if not self.is_trained:
            raise ValueError(f"模型 {self.model_name} 尚未训练，请先调用train方法")
        
        predictions = []
        batch_size = 32
        
        self.bert.eval()
        self.classifier.eval()
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                # 分词和编码
                tokens = self.tokenizer(batch_texts, padding=True, truncation=True,
                                      max_length=512, return_tensors='pt')
                input_ids = tokens["input_ids"].to(self.device)
                attention_mask = tokens["attention_mask"].to(self.device)
                
                # 获取BERT输出
                bert_outputs = self.bert(input_ids, attention_mask=attention_mask)
                bert_output = bert_outputs[0][:, 0]
                
                # 分类器预测
                outputs = self.classifier(bert_output)
                outputs = outputs.view(-1)
                
                # 转换为类别标签
                preds = (outputs > 0.5).cpu().numpy()
                predictions.extend(preds.astype(int).tolist())
        
        return predictions
    
    def predict_single(self, text: str) -> Tuple[int, float]:
        """预测单条文本的情感"""
        if not self.is_trained:
            raise ValueError(f"模型 {self.model_name} 尚未训练，请先调用train方法")
        
        self.bert.eval()
        self.classifier.eval()
        
        with torch.no_grad():
            # 分词和编码
            tokens = self.tokenizer([text], padding=True, truncation=True,
                                  max_length=512, return_tensors='pt')
            input_ids = tokens["input_ids"].to(self.device)
            attention_mask = tokens["attention_mask"].to(self.device)
            
            # 获取BERT输出
            bert_outputs = self.bert(input_ids, attention_mask=attention_mask)
            bert_output = bert_outputs[0][:, 0]
            
            # 分类器预测
            output = self.classifier(bert_output)
            prob = output.item()
            
            prediction = int(prob > 0.5)
            confidence = prob if prediction == 1 else 1 - prob
        
        return prediction, confidence
    
    def save_model(self, model_path: str = None) -> None:
        """保存模型"""
        if not self.is_trained:
            raise ValueError(f"模型 {self.model_name} 尚未训练，无法保存")
        
        if model_path is None:
            model_path = f"./model/{self.model_name.lower()}_model.pth"
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # 保存分类器和相关信息
        model_data = {
            'classifier_state_dict': self.classifier.state_dict(),
            'model_path': self.model_path,
            'input_size': 768,
            'device': str(self.device)
        }
        
        torch.save(model_data, model_path)
        print(f"模型已保存到: {model_path}")
    
    def load_model(self, model_path: str) -> None:
        """加载模型"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        model_data = torch.load(model_path, map_location=self.device)
        
        # 设置BERT模型路径
        self.model_path = model_data['model_path']
        
        # 加载BERT
        self._load_bert()
        
        # 重建分类器
        input_size = model_data['input_size']
        self.classifier = BertClassifier(input_size).to(self.device)
        
        # 加载分类器权重
        self.classifier.load_state_dict(model_data['classifier_state_dict'])
        
        self.is_trained = True
        print(f"已加载模型: {model_path}")
    
    @staticmethod
    def load_data(train_path: str, test_path: str) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
        """加载BERT格式的数据"""
        print("加载训练数据...")
        train_data = load_corpus_bert(train_path)
        print(f"训练数据量: {len(train_data)}")
        
        print("加载测试数据...")
        test_data = load_corpus_bert(test_path)
        print(f"测试数据量: {len(test_data)}")
        
        return train_data, test_data


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='BERT情感分析模型训练')
    parser.add_argument('--train_path', type=str, default='./data/weibo2018/train.txt',
                        help='训练数据路径')
    parser.add_argument('--test_path', type=str, default='./data/weibo2018/test.txt',
                        help='测试数据路径')
    parser.add_argument('--model_path', type=str, default='./model/bert_model.pth',
                        help='模型保存路径')
    parser.add_argument('--bert_path', type=str, default='./model/chinese_wwm_pytorch',
                        help='BERT预训练模型路径')
    parser.add_argument('--epochs', type=int, default=10,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='批大小')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='学习率')
    parser.add_argument('--eval_only', action='store_true',
                        help='仅评估已有模型，不进行训练')
    
    args = parser.parse_args()
    
    # 创建模型
    model = BertModel_Custom(args.bert_path)
    
    if args.eval_only:
        # 仅评估模式
        print("评估模式：加载已有模型进行评估")
        model.load_model(args.model_path)
        
        # 加载测试数据
        _, test_data = model.load_data(args.train_path, args.test_path)
        
        # 评估模型
        model.evaluate(test_data)
    else:
        # 训练模式
        # 加载数据
        train_data, test_data = model.load_data(args.train_path, args.test_path)
        
        # 训练模型
        model.train(
            train_data,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
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