# -*- coding: utf-8 -*-
"""
Qwen3-Embedding通用训练脚本
支持0.6B、4B、8B三种规模的模型
"""
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from typing import List, Tuple
import warnings
from tqdm import tqdm

from base_model import BaseQwenModel
from models_config import QWEN3_MODELS, MODEL_PATHS

warnings.filterwarnings("ignore")


class SentimentDataset(Dataset):
    """情感分析数据集"""
    
    def __init__(self, data: List[Tuple[str, int]], tokenizer, max_length=512):
        self.texts = [item[0] for item in data]
        self.labels = [item[1] for item in data]
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.float)
        }


class SentimentClassifier(nn.Module):
    """情感分类器"""
    
    def __init__(self, embedding_model, embedding_dim, hidden_dim=256):
        super(SentimentClassifier, self).__init__()
        self.embedding_model = embedding_model
        
        # 冻结embedding模型参数
        for param in self.embedding_model.parameters():
            param.requires_grad = False
            
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, input_ids, attention_mask):
        # 获取embedding
        with torch.no_grad():
            outputs = self.embedding_model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state[:, 0, :]
        
        # 通过分类头
        logits = self.classifier(embeddings)
        return logits.squeeze()


class Qwen3EmbeddingUniversal(BaseQwenModel):
    """通用Qwen3-Embedding模型"""
    
    def __init__(self, model_size: str = "0.6B"):
        if model_size not in QWEN3_MODELS:
            raise ValueError(f"不支持的模型大小: {model_size}")
            
        super().__init__(f"Qwen3-Embedding-{model_size}")
        self.model_size = model_size
        self.config = QWEN3_MODELS[model_size]
        self.model_name_hf = self.config["embedding_model"]
        self.embedding_dim = self.config["embedding_dim"]
        
        self.tokenizer = None
        self.embedding_model = None
        self.classifier_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def _load_embedding_model(self):
        """加载Qwen3 Embedding模型"""
        print(f"加载{self.model_size}模型: {self.model_name_hf}")
        
        # 第一步：检查当前文件夹的models目录
        local_model_dir = f"./models/qwen3-embedding-{self.model_size.lower()}"
        if os.path.exists(local_model_dir) and os.path.exists(os.path.join(local_model_dir, "config.json")):
            try:
                print(f"发现本地模型，从本地加载: {local_model_dir}")
                self.tokenizer = AutoTokenizer.from_pretrained(local_model_dir)
                self.embedding_model = AutoModel.from_pretrained(local_model_dir).to(self.device)
                print(f"从本地模型加载{self.model_size}模型成功")
                return
                
            except Exception as e:
                print(f"本地模型加载失败: {e}")
        
        # 第二步：检查HuggingFace缓存
        try:
            from transformers.utils import default_cache_path
            cache_path = default_cache_path
            print(f"检查HuggingFace缓存: {cache_path}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_hf)
            self.embedding_model = AutoModel.from_pretrained(self.model_name_hf).to(self.device)
            print(f"从HuggingFace缓存加载{self.model_size}模型成功")
            
            # 保存到本地models目录
            print(f"保存模型到本地: {local_model_dir}")
            os.makedirs(local_model_dir, exist_ok=True)
            self.tokenizer.save_pretrained(local_model_dir)
            self.embedding_model.save_pretrained(local_model_dir)
            print(f"模型已保存到: {local_model_dir}")
            
        except Exception as e:
            print(f"从HuggingFace缓存加载失败: {e}")
            
            # 第三步：从HuggingFace下载
            try:
                print(f"正在从HuggingFace下载{self.model_size}模型...")
                
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name_hf,
                    force_download=True
                )
                self.embedding_model = AutoModel.from_pretrained(
                    self.model_name_hf,
                    force_download=True
                ).to(self.device)
                
                # 保存到本地models目录
                os.makedirs(local_model_dir, exist_ok=True)
                self.tokenizer.save_pretrained(local_model_dir)
                self.embedding_model.save_pretrained(local_model_dir)
                print(f"{self.model_size}模型下载并保存到: {local_model_dir}")
                
            except Exception as e2:
                print(f"从HuggingFace下载也失败: {e2}")
                raise RuntimeError(f"无法加载{self.model_size}模型，所有方法都失败了")
    
    def train(self, train_data: List[Tuple[str, int]], **kwargs) -> None:
        """训练模型"""
        print(f"开始训练 Qwen3-Embedding-{self.model_size} 模型...")
        
        # 加载embedding模型
        self._load_embedding_model()
        
        # 超参数（使用配置文件的推荐值或用户指定值）
        batch_size = kwargs.get('batch_size', self.config['recommended_batch_size'])
        learning_rate = kwargs.get('learning_rate', self.config['recommended_lr'])
        num_epochs = kwargs.get('num_epochs', 5)
        max_length = kwargs.get('max_length', 512)
        
        print(f"超参数: batch_size={batch_size}, lr={learning_rate}, epochs={num_epochs}")
        print(f"嵌入维度: {self.embedding_dim}")
        
        # 创建数据集
        train_dataset = SentimentDataset(train_data, self.tokenizer, max_length)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # 创建分类器
        self.classifier_model = SentimentClassifier(
            self.embedding_model, 
            self.embedding_dim
        ).to(self.device)
        
        # 损失函数和优化器
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.classifier_model.classifier.parameters(), lr=learning_rate)
        
        # 训练循环
        self.classifier_model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # 前向传播
                outputs = self.classifier_model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                progress_bar.set_postfix({'loss': total_loss / num_batches})
            
            avg_loss = total_loss / num_batches
            print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")
        
        self.model = self.classifier_model
        self.is_trained = True
        print(f"Qwen3-Embedding-{self.model_size} 模型训练完成！")
    
    def predict(self, texts: List[str]) -> List[int]:
        """预测文本情感"""
        if not self.is_trained:
            raise ValueError(f"模型 {self.model_name} 尚未训练")
        
        predictions = []
        batch_size = 32
        
        self.classifier_model.eval()
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                encodings = self.tokenizer(
                    batch_texts,
                    max_length=512,
                    padding=True,
                    truncation=True,
                    return_tensors='pt'
                )
                
                input_ids = encodings['input_ids'].to(self.device)
                attention_mask = encodings['attention_mask'].to(self.device)
                
                outputs = self.classifier_model(input_ids, attention_mask)
                preds = (outputs > 0.5).cpu().numpy()
                predictions.extend(preds.astype(int).tolist())
        
        return predictions
    
    def predict_single(self, text: str) -> Tuple[int, float]:
        """预测单条文本的情感"""
        if not self.is_trained:
            raise ValueError(f"模型 {self.model_name} 尚未训练")
        
        self.classifier_model.eval()
        with torch.no_grad():
            encoding = self.tokenizer(
                text,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            output = self.classifier_model(input_ids, attention_mask)
            prob = output.item()
            prediction = int(prob > 0.5)
            confidence = prob if prediction == 1 else 1 - prob
        
        return prediction, confidence
    
    def save_model(self, model_path: str = None) -> None:
        """保存模型"""
        if not self.is_trained:
            raise ValueError(f"模型 {self.model_name} 尚未训练")
        
        if model_path is None:
            model_path = MODEL_PATHS["embedding"][self.model_size]
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        model_data = {
            'classifier_state_dict': self.classifier_model.classifier.state_dict(),
            'model_size': self.model_size,
            'model_name_hf': self.model_name_hf,
            'embedding_dim': self.embedding_dim,
            'device': str(self.device)
        }
        
        torch.save(model_data, model_path)
        print(f"模型已保存到: {model_path}")
    
    def load_model(self, model_path: str) -> None:
        """加载模型"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 加载模型数据
        model_data = torch.load(model_path, map_location=self.device)
        
        # 验证模型大小匹配
        if model_data['model_size'] != self.model_size:
            raise ValueError(f"模型大小不匹配: 期望{self.model_size}, 实际{model_data['model_size']}")
        
        # 加载embedding模型
        self._load_embedding_model()
        
        # 重建分类器
        self.classifier_model = SentimentClassifier(
            self.embedding_model, 
            model_data['embedding_dim']
        ).to(self.device)
        self.classifier_model.classifier.load_state_dict(model_data['classifier_state_dict'])
        
        self.model = self.classifier_model
        self.is_trained = True
        print(f"已加载Qwen3-Embedding-{self.model_size}模型: {model_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Qwen3-Embedding通用训练脚本')
    parser.add_argument('--model_size', type=str, choices=['0.6B', '4B', '8B'], 
                        help='模型大小')
    parser.add_argument('--train_path', type=str, default='./dataset/train.txt',
                        help='训练数据路径')
    parser.add_argument('--test_path', type=str, default='./dataset/test.txt',
                        help='测试数据路径')
    parser.add_argument('--model_path', type=str, help='模型保存路径（可选）')
    parser.add_argument('--epochs', type=int, default=5, help='训练轮数')
    parser.add_argument('--batch_size', type=int, help='批大小（可选，使用推荐值）')
    parser.add_argument('--learning_rate', type=float, help='学习率（可选，使用推荐值）')
    parser.add_argument('--eval_only', action='store_true', help='仅评估模式')
    
    args = parser.parse_args()
    
    # 如果没有指定模型大小，则询问用户
    if not args.model_size:
        print("Qwen3-Embedding模型训练")
        print("="*40)
        print("可用模型大小:")
        print("  1. 0.6B - 轻量级，训练快速，显存需求约4GB")
        print("  2. 4B  - 中等规模，性能均衡，显存需求约16GB") 
        print("  3. 8B  - 大规模，性能最佳，显存需求约32GB")
        
        while True:
            choice = input("\n请选择模型大小 (1/2/3): ").strip()
            if choice == '1':
                args.model_size = '0.6B'
                break
            elif choice == '2':
                args.model_size = '4B'
                break
            elif choice == '3':
                args.model_size = '8B'
                break
            else:
                print("无效选择，请输入 1、2 或 3")
        
        print(f"已选择: Qwen3-Embedding-{args.model_size}")
        print()
    
    # 确保models目录存在
    os.makedirs('./models', exist_ok=True)
    
    # 创建模型
    model = Qwen3EmbeddingUniversal(args.model_size)
    
    # 确定模型保存路径
    model_path = args.model_path or MODEL_PATHS["embedding"][args.model_size]
    
    if args.eval_only:
        # 仅评估模式
        print(f"评估模式：加载Qwen3-Embedding-{args.model_size}模型")
        model.load_model(model_path)
        
        _, test_data = BaseQwenModel.load_data(args.train_path, args.test_path)
        model.evaluate(test_data)
    else:
        # 训练模式
        train_data, test_data = BaseQwenModel.load_data(args.train_path, args.test_path)
        
        # 准备训练参数
        train_kwargs = {'num_epochs': args.epochs}
        if args.batch_size:
            train_kwargs['batch_size'] = args.batch_size
        if args.learning_rate:
            train_kwargs['learning_rate'] = args.learning_rate
        
        # 训练模型
        model.train(train_data, **train_kwargs)
        
        # 评估模型
        model.evaluate(test_data)
        
        # 保存模型
        model.save_model(model_path)
        
        # 示例预测
        print(f"\nQwen3-Embedding-{args.model_size} 示例预测:")
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