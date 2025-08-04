# -*- coding: utf-8 -*-
"""
LSTM情感分析模型训练脚本
"""
import argparse
import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from gensim import models
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score
from typing import List, Tuple, Dict, Any
import numpy as np

from base_model import BaseModel


class LSTMDataset(Dataset):
    """LSTM数据集"""
    
    def __init__(self, data: List[Tuple[str, int]], word2vec_model):
        self.data = []
        self.label = []
        
        for text, label in data:
            vectors = []
            for word in text.split(" "):
                if word in word2vec_model.wv.key_to_index:
                    vectors.append(word2vec_model.wv[word])
            
            if len(vectors) > 0:  # 确保有有效的词向量
                vectors = torch.Tensor(vectors)
                self.data.append(vectors)
                self.label.append(label)
    
    def __getitem__(self, index):
        return self.data[index], self.label[index]
    
    def __len__(self):
        return len(self.label)


def collate_fn(data):
    """批处理函数"""
    data.sort(key=lambda x: len(x[0]), reverse=True)
    data_length = [len(sq[0]) for sq in data]
    x = [i[0] for i in data]
    y = [i[1] for i in data]
    data = pad_sequence(x, batch_first=True, padding_value=0)
    return data, torch.tensor(y, dtype=torch.float32), data_length


class LSTMNet(nn.Module):
    """LSTM网络结构"""
    
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, 1)  # 双向LSTM
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, lengths):
        device = x.device
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        
        packed_input = pack_padded_sequence(input=x, lengths=lengths, batch_first=True)
        packed_out, (h_n, h_c) = self.lstm(packed_input, (h0, c0))
        
        # 双向LSTM，拼接最后的隐藏状态
        lstm_out = torch.cat([h_n[-2], h_n[-1]], 1)
        out = self.fc(lstm_out)
        out = self.sigmoid(out)
        return out


class LSTMModel(BaseModel):
    """LSTM情感分析模型"""
    
    def __init__(self):
        super().__init__("LSTM")
        self.word2vec_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def _train_word2vec(self, train_data: List[Tuple[str, int]], **kwargs):
        """训练Word2Vec词向量"""
        print("训练Word2Vec词向量...")
        
        # 准备Word2Vec输入数据
        wv_input = [text.split(" ") for text, _ in train_data]
        
        vector_size = kwargs.get('vector_size', 64)
        min_count = kwargs.get('min_count', 1)
        epochs = kwargs.get('epochs', 1000)
        
        # 训练Word2Vec
        self.word2vec_model = models.Word2Vec(
            wv_input,
            vector_size=vector_size,
            min_count=min_count,
            epochs=epochs
        )
        
        print(f"Word2Vec训练完成，词向量维度: {vector_size}")
        
    def train(self, train_data: List[Tuple[str, int]], **kwargs) -> None:
        """训练LSTM模型"""
        print(f"开始训练 {self.model_name} 模型...")
        
        # 训练Word2Vec
        self._train_word2vec(train_data, **kwargs)
        
        # 超参数
        learning_rate = kwargs.get('learning_rate', 5e-4)
        num_epochs = kwargs.get('num_epochs', 5)
        batch_size = kwargs.get('batch_size', 100)
        embed_size = kwargs.get('embed_size', 64)
        hidden_size = kwargs.get('hidden_size', 64)
        num_layers = kwargs.get('num_layers', 2)
        
        print(f"LSTM超参数: lr={learning_rate}, epochs={num_epochs}, "
              f"batch_size={batch_size}, hidden_size={hidden_size}")
        
        # 创建数据集
        train_dataset = LSTMDataset(train_data, self.word2vec_model)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                 collate_fn=collate_fn, shuffle=True)
        
        # 创建模型
        self.model = LSTMNet(embed_size, hidden_size, num_layers).to(self.device)
        
        # 损失函数和优化器
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # 训练循环
        self.model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0
            
            for i, (x, labels, lengths) in enumerate(train_loader):
                x = x.to(self.device)
                labels = labels.to(self.device)
                
                # 前向传播
                outputs = self.model(x, lengths)
                logits = outputs.view(-1)
                loss = criterion(logits, labels)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                if (i + 1) % 10 == 0:
                    avg_loss = total_loss / num_batches
                    print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}], Loss: {avg_loss:.4f}")
            
            # 保存每个epoch的模型
            if kwargs.get('save_each_epoch', False):
                epoch_model_path = f"./model/lstm_epoch_{epoch+1}.pth"
                os.makedirs(os.path.dirname(epoch_model_path), exist_ok=True)
                torch.save(self.model.state_dict(), epoch_model_path)
                print(f"已保存模型: {epoch_model_path}")
        
        self.is_trained = True
        print(f"{self.model_name} 模型训练完成！")
    
    def predict(self, texts: List[str]) -> List[int]:
        """预测文本情感"""
        if not self.is_trained:
            raise ValueError(f"模型 {self.model_name} 尚未训练，请先调用train方法")
        
        # 创建数据集
        test_data = [(text, 0) for text in texts]  # 标签无关紧要
        test_dataset = LSTMDataset(test_data, self.word2vec_model)
        test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn)
        
        predictions = []
        self.model.eval()
        
        with torch.no_grad():
            for x, _, lengths in test_loader:
                x = x.to(self.device)
                outputs = self.model(x, lengths)
                outputs = outputs.view(-1)
                
                # 转换为类别标签
                preds = (outputs > 0.5).cpu().numpy()
                predictions.extend(preds.astype(int).tolist())
        
        return predictions
    
    def predict_single(self, text: str) -> Tuple[int, float]:
        """预测单条文本的情感"""
        if not self.is_trained:
            raise ValueError(f"模型 {self.model_name} 尚未训练，请先调用train方法")
        
        # 转换为词向量
        vectors = []
        for word in text.split(" "):
            if word in self.word2vec_model.wv.key_to_index:
                vectors.append(self.word2vec_model.wv[word])
        
        if len(vectors) == 0:
            return 0, 0.5  # 如果没有有效词向量，返回默认值
        
        # 转换为tensor
        x = torch.Tensor(vectors).unsqueeze(0).to(self.device)  # 添加batch维度
        lengths = [len(vectors)]
        
        self.model.eval()
        with torch.no_grad():
            output = self.model(x, lengths)
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
        
        # 保存模型状态和Word2Vec
        model_data = {
            'model_state_dict': self.model.state_dict(),
            'word2vec_model': self.word2vec_model,
            'model_config': {
                'embed_size': 64,
                'hidden_size': 64,
                'num_layers': 2
            },
            'device': str(self.device)
        }
        
        torch.save(model_data, model_path)
        print(f"模型已保存到: {model_path}")
    
    def load_model(self, model_path: str) -> None:
        """加载模型"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        model_data = torch.load(model_path, map_location=self.device)
        
        # 加载Word2Vec
        self.word2vec_model = model_data['word2vec_model']
        
        # 重建LSTM网络
        config = model_data['model_config']
        self.model = LSTMNet(
            config['embed_size'],
            config['hidden_size'],
            config['num_layers']
        ).to(self.device)
        
        # 加载模型权重
        self.model.load_state_dict(model_data['model_state_dict'])
        
        self.is_trained = True
        print(f"已加载模型: {model_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='LSTM情感分析模型训练')
    parser.add_argument('--train_path', type=str, default='./data/weibo2018/train.txt',
                        help='训练数据路径')
    parser.add_argument('--test_path', type=str, default='./data/weibo2018/test.txt',
                        help='测试数据路径')
    parser.add_argument('--model_path', type=str, default='./model/lstm_model.pth',
                        help='模型保存路径')
    parser.add_argument('--epochs', type=int, default=5,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='批大小')
    parser.add_argument('--hidden_size', type=int, default=64,
                        help='LSTM隐藏层大小')
    parser.add_argument('--learning_rate', type=float, default=5e-4,
                        help='学习率')
    parser.add_argument('--eval_only', action='store_true',
                        help='仅评估已有模型，不进行训练')
    
    args = parser.parse_args()
    
    # 创建模型
    model = LSTMModel()
    
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
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            hidden_size=args.hidden_size,
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