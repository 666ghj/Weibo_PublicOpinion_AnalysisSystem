import pandas as pd  # 用于数据处理
import numpy as np  # 用于科学计算
import csv  # 用于读取CSV文件
# from utils.mynlp import SnowNLP  # 用于中文自然语言处理（此处未实际使用）
from sklearn.feature_extraction.text import TfidfVectorizer  # 用于文本特征提取
from sklearn.naive_bayes import MultinomialNB  # 用于多项式朴素贝叶斯分类
from sklearn.model_selection import train_test_split  # 用于划分训练集和测试集
from sklearn.metrics import accuracy_score  # 用于计算模型准确度
import torch
from transformers import BertTokenizer, BertModel
from torch import nn
from torch.utils.data import Dataset, DataLoader
from utils.logger import model_logger as logging

def getSentiment_data():
    # 从CSV文件中读取情感数据
    sentiment_data = []
    with open('./target.csv', 'r', encoding='utf8') as readerFile:
        reader = csv.reader(readerFile)
        for data in reader:
            sentiment_data.append(data)
    return sentiment_data

class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
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

class BertClassifier(nn.Module):
    def __init__(self, n_classes):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.drop = nn.Dropout(p=0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, n_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs[1]
        output = self.drop(pooled_output)
        return self.fc(output)

def train_model(model, train_loader, val_loader, learning_rate=2e-5, epochs=4):
    """训练模型"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"使用设备: {device}")
        
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            logging.info(f"开始训练 Epoch {epoch + 1}/{epochs}")
            
            for batch in train_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            avg_train_loss = total_loss / len(train_loader)
            logging.info(f"Epoch {epoch + 1} 平均训练损失: {avg_train_loss:.4f}")
            
            # 验证
            model.eval()
            val_preds = []
            val_labels = []
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['label'].to(device)
                    
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    _, preds = torch.max(outputs, dim=1)
                    
                    val_preds.extend(preds.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())
            
            val_accuracy = accuracy_score(val_labels, val_preds)
            logging.info(f"Epoch {epoch + 1} 验证准确率: {val_accuracy:.4f}")
            
        logging.info("模型训练完成")
        return model
        
    except Exception as e:
        logging.error(f"模型训练过程中发生错误: {e}")
        raise

def model_train():
    """训练模型并计算准确度"""
    try:
        # 加载数据
        logging.info("开始加载数据...")
        data = pd.read_csv('data/train_data.csv')
        texts = data['text'].values
        labels = data['label'].values
        
        # 数据集分割
        X_train, X_val, y_train, y_val = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )
        logging.info(f"训练集大小: {len(X_train)}, 验证集大小: {len(X_val)}")
        
        # 初始化tokenizer和数据集
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        train_dataset = TextClassificationDataset(X_train, y_train, tokenizer)
        val_dataset = TextClassificationDataset(X_val, y_val, tokenizer)
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16)
        
        # 初始化模型
        model = BertClassifier(n_classes=len(np.unique(labels)))
        logging.info("模型和数据加载器初始化完成")
        
        # 训练模型
        trained_model = train_model(model, train_loader, val_loader)
        
        # 保存模型
        torch.save(trained_model.state_dict(), 'model/saved_model.pth')
        logging.info("模型已保存到 model/saved_model.pth")
        
    except Exception as e:
        logging.error(f"模型训练主函数发生错误: {e}")
        raise

if __name__ == "__main__":
    try:
        model_train()
    except Exception as e:
        logging.error(f"程序执行失败: {e}")
