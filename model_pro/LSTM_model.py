import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import jieba
from transformers import BertTokenizer
import logging
import os

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('LSTM_model')

class TextDataset(Dataset):
    """文本数据集类，用于加载和预处理文本数据"""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # BERT分词并获得输入ID和注意力掩码
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class LSTMSentimentModel(nn.Module):
    """基于LSTM的情感分析模型"""
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers=2, 
                 bidirectional=True, dropout=0.5, pad_idx=0):
        super().__init__()
        
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        
        # LSTM层
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True
        )
        
        # 全连接层，如果是双向LSTM，输入维度需要翻倍
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, attention_mask=None):
        # 文本通过嵌入层 [batch_size, seq_len] -> [batch_size, seq_len, embedding_dim]
        embedded = self.embedding(text)
        
        # 应用dropout
        embedded = self.dropout(embedded)
        
        # 通过LSTM [batch_size, seq_len, embedding_dim] -> [batch_size, seq_len, hidden_dim*2]
        if attention_mask is not None:
            # 创建打包的序列
            lengths = attention_mask.sum(dim=1).to('cpu')
            packed_embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths, batch_first=True, enforce_sorted=False
            )
            packed_output, (hidden, cell) = self.lstm(packed_embedded)
            # 解包序列
            output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        else:
            output, (hidden, cell) = self.lstm(embedded)
        
        # 如果是双向LSTM，需要拼接最后一层的前向和后向隐藏状态
        if self.lstm.bidirectional:
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            hidden = hidden[-1]
        
        # 应用dropout
        hidden = self.dropout(hidden)
        
        # 全连接层
        return self.fc(hidden)

class LSTMModelManager:
    """LSTM模型管理类，用于训练、评估和预测"""
    
    def __init__(self, bert_model_path, model_save_path=None, vocab_size=30522, 
                 embedding_dim=128, hidden_dim=256, output_dim=2, n_layers=2, 
                 bidirectional=True, dropout=0.5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_path)
        self.vocab_size = vocab_size
        self.model = LSTMSentimentModel(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            n_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            pad_idx=self.tokenizer.pad_token_id
        ).to(self.device)
        
        self.model_save_path = model_save_path
        if model_save_path and os.path.exists(model_save_path):
            self.model.load_state_dict(torch.load(model_save_path, map_location=self.device))
            logger.info(f"已从 {model_save_path} 加载模型")
    
    def train(self, train_texts, train_labels, val_texts=None, val_labels=None, 
              batch_size=32, learning_rate=2e-5, epochs=10, validation_split=0.2):
        """训练模型"""
        logger.info("开始训练模型...")
        
        # 如果没有提供验证集，从训练集中划分
        if val_texts is None or val_labels is None:
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                train_texts, train_labels, test_size=validation_split, random_state=42
            )
        
        # 创建数据集和数据加载器
        train_dataset = TextDataset(train_texts, train_labels, self.tokenizer)
        val_dataset = TextDataset(val_texts, val_labels, self.tokenizer)
        
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
        
        # 优化器和损失函数
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # 训练循环
        best_val_loss = float('inf')
        for epoch in range(epochs):
            # 训练模式
            self.model.train()
            train_loss = 0
            train_preds = []
            train_labels_list = []
            
            for batch in train_dataloader:
                # 获取数据
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # 前向传播
                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask)
                
                # 计算损失
                loss = criterion(outputs, labels)
                train_loss += loss.item()
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                # 收集预测和标签
                _, predicted = torch.max(outputs, 1)
                train_preds.extend(predicted.cpu().numpy())
                train_labels_list.extend(labels.cpu().numpy())
            
            # 计算训练集的评估指标
            train_accuracy = accuracy_score(train_labels_list, train_preds)
            train_f1 = f1_score(train_labels_list, train_preds, average='macro')
            
            # 验证模式
            self.model.eval()
            val_loss = 0
            val_preds = []
            val_labels_list = []
            
            with torch.no_grad():
                for batch in val_dataloader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    outputs = self.model(input_ids, attention_mask)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs, 1)
                    val_preds.extend(predicted.cpu().numpy())
                    val_labels_list.extend(labels.cpu().numpy())
            
            # 计算验证集的评估指标
            val_accuracy = accuracy_score(val_labels_list, val_preds)
            val_f1 = f1_score(val_labels_list, val_preds, average='macro')
            
            # 计算平均损失
            train_loss /= len(train_dataloader)
            val_loss /= len(val_dataloader)
            
            logger.info(f'Epoch {epoch+1}/{epochs} | '
                        f'Train Loss: {train_loss:.4f} | '
                        f'Train Acc: {train_accuracy:.4f} | '
                        f'Train F1: {train_f1:.4f} | '
                        f'Val Loss: {val_loss:.4f} | '
                        f'Val Acc: {val_accuracy:.4f} | '
                        f'Val F1: {val_f1:.4f}')
            
            # 保存最佳模型
            if val_loss < best_val_loss and self.model_save_path:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.model_save_path)
                logger.info(f"模型已保存到 {self.model_save_path}")
        
        # 如果有保存路径但没有保存过模型，保存最后一轮的模型
        if self.model_save_path and best_val_loss == float('inf'):
            torch.save(self.model.state_dict(), self.model_save_path)
            logger.info(f"最终模型已保存到 {self.model_save_path}")
        
        return train_loss, val_loss, val_accuracy, val_f1
    
    def evaluate(self, test_texts, test_labels, batch_size=32):
        """评估模型"""
        logger.info("评估模型...")
        
        # 创建测试数据集和数据加载器
        test_dataset = TextDataset(test_texts, test_labels, self.tokenizer)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
        
        # 设置为评估模式
        self.model.eval()
        
        # 损失函数
        criterion = nn.CrossEntropyLoss()
        test_loss = 0
        test_preds = []
        test_probs = []
        test_labels_list = []
        
        with torch.no_grad():
            for batch in test_dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                test_preds.extend(predicted.cpu().numpy())
                test_probs.extend(probs.cpu().numpy())
                test_labels_list.extend(labels.cpu().numpy())
        
        # 计算平均损失
        test_loss /= len(test_dataloader)
        
        # 计算评估指标
        accuracy = accuracy_score(test_labels_list, test_preds)
        precision = precision_score(test_labels_list, test_preds, average='macro')
        recall = recall_score(test_labels_list, test_preds, average='macro')
        f1 = f1_score(test_labels_list, test_preds, average='macro')
        conf_matrix = confusion_matrix(test_labels_list, test_preds)
        
        logger.info(f'Test Loss: {test_loss:.4f}')
        logger.info(f'Accuracy: {accuracy:.4f}')
        logger.info(f'Precision: {precision:.4f}')
        logger.info(f'Recall: {recall:.4f}')
        logger.info(f'F1 Score: {f1:.4f}')
        logger.info(f'Confusion Matrix:\n{conf_matrix}')
        
        return {
            'loss': test_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': conf_matrix,
            'predictions': test_preds,
            'probabilities': test_probs
        }
    
    def predict_batch(self, texts, batch_size=32):
        """批量预测文本的情感"""
        if not texts:
            return None, None
            
        # 确保文本是列表格式
        if isinstance(texts, str):
            texts = [texts]
        
        # 创建数据集（没有标签，使用占位符）
        dummy_labels = [0] * len(texts)
        dataset = TextDataset(texts, dummy_labels, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        
        # 设置为评估模式
        self.model.eval()
        
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        return all_preds, all_probs
    
    def predict(self, text):
        """预测单个文本的情感"""
        predictions, probabilities = self.predict_batch([text])
        if predictions is not None and len(predictions) > 0:
            return predictions[0], probabilities[0]
        return None, None

# 创建全局模型实例
lstm_model_manager = LSTMModelManager(
    bert_model_path='model_pro/bert_model',
    model_save_path='model_pro/lstm_model.pt'
)

# 测试代码
if __name__ == "__main__":
    # 加载数据
    train_data = pd.read_csv('model_pro/train.csv')
    dev_data = pd.read_csv('model_pro/dev.csv')
    test_data = pd.read_csv('model_pro/test.csv')
    
    # 处理数据
    train_texts = train_data['text'].values
    train_labels = train_data['label'].values
    
    dev_texts = dev_data['text'].values
    dev_labels = dev_data['label'].values
    
    test_texts = test_data['text'].values
    test_labels = test_data['label'].values
    
    # 训练模型
    lstm_model_manager.train(
        train_texts, train_labels,
        val_texts=dev_texts, val_labels=dev_labels,
        batch_size=32, epochs=5
    )
    
    # 评估模型
    results = lstm_model_manager.evaluate(test_texts, test_labels)
    
    # 测试预测功能
    test_sentences = [
        "这件事情做得非常好",
        "服务太差了，态度恶劣",
        "这个产品质量一般，但价格便宜",
        "我对这家公司非常满意",
    ]
    
    for sentence in test_sentences:
        pred, prob = lstm_model_manager.predict(sentence)
        label = '良好' if pred == 0 else '不良'
        confidence = prob[pred]
        print(f"句子: '{sentence}' 预测结果: {label} (置信度: {confidence:.2%})") 