import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import jieba
from transformers import BertTokenizer, BertModel
import logging
import os
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau
from gensim.models import KeyedVectors
import json
import torch.nn.functional as F

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

class AttentionLayer(nn.Module):
    """注意力层"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)
        
    def forward(self, lstm_output):
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        return context_vector, attention_weights

# 添加数据增强类
class TextAugmenter:
    def __init__(self, language='zh', synonyms_file=None):
        self.language = language
        self.synonyms_dict = self._load_synonyms(synonyms_file)
        
    def _load_synonyms(self, file_path):
        base_dict = {
            "很好": ["非常好", "太好了", "特别好", "相当好", "真不错"],
            "糟糕": ["差劲", "很差", "不好", "太差", "糟透了"],
            "一般": ["还行", "凑合", "普通", "马马虎虎", "中等"],
            "满意": ["很满意", "挺好", "不错", "称心如意"],
            "生气": ["愤怒", "恼火", "不爽", "气愤"],
            "失望": ["伤心", "难过", "不满意", "遗憾"],
            # 添加更多情感词汇对
        }
        
        if file_path and os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    custom_dict = json.load(f)
                base_dict.update(custom_dict)
            except Exception as e:
                logger.warning(f"加载同义词典失败: {e}")
                
        return base_dict
    
    def synonym_replacement(self, text, n=1):
        words = list(jieba.cut(text))
        new_words = words.copy()
        num_replaced = 0
        
        for word in list(set(words)):
            if len(word) > 1 and num_replaced < n:
                synonyms = self._get_synonyms(word)
                if synonyms:
                    synonym = random.choice(synonyms)
                    new_words = [synonym if w == word else w for w in new_words]
                    num_replaced += 1
                    
        return ''.join(new_words)
    
    def _get_synonyms(self, word):
        return self.synonyms_dict.get(word, [])
    
    def augment(self, texts, labels, augment_ratio=0.5):
        augmented_texts = []
        augmented_labels = []
        
        for text, label in zip(texts, labels):
            augmented_texts.append(text)
            augmented_labels.append(label)
            
            if random.random() < augment_ratio:
                aug_text = self.synonym_replacement(text)
                augmented_texts.append(aug_text)
                augmented_labels.append(label)
        
        return np.array(augmented_texts), np.array(augmented_labels)

class LSTMSentimentModel(nn.Module):
    """基于LSTM的情感分析模型"""
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers=1, 
                 bidirectional=True, dropout=0.3, pad_idx=0, pretrained_embeddings=None):
        super().__init__()
        
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            self.embedding.weight.requires_grad = True
        
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True
        )
        
        # 注意力层
        self.attention = AttentionLayer(hidden_dim * 2 if bidirectional else hidden_dim)
        
        # 全连接层
        fc_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc1 = nn.Linear(fc_dim, fc_dim // 2)
        self.fc2 = nn.Linear(fc_dim // 2, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(fc_dim // 2)
        self.relu = nn.ReLU()
        
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
        
        # 应用注意力机制
        context_vector, attention_weights = self.attention(output)
        
        # 应用dropout和全连接层
        x = self.dropout(context_vector)
        x = self.fc1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x, attention_weights

# 添加早停类
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

class LSTMModelManager:
    """LSTM模型管理类，用于训练、评估和预测"""
    
    def __init__(self, bert_model_path, model_save_path=None, vocab_size=30522, 
                 embedding_dim=100, hidden_dim=64, output_dim=2, n_layers=1, 
                 bidirectional=True, dropout=0.3, word2vec_path=None, random_seed=42):
        # 设置随机种子以确保可重现性
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_path)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
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
        
        self.augmenter = TextAugmenter()
        self.early_stopping = EarlyStopping(patience=5)
        
        # 加载预训练词向量
        self.pretrained_embeddings = None
        if word2vec_path and os.path.exists(word2vec_path):
            try:
                word_vectors = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
                self.pretrained_embeddings = self._build_embedding_matrix(word_vectors)
                logger.info("成功加载预训练词向量")
            except Exception as e:
                logger.warning(f"加载预训练词向量失败: {e}")
        
        # 初始化对抗训练参数
        self.epsilon = 0.01
        self.alpha = 0.001
        
    def _build_embedding_matrix(self, word_vectors):
        embedding_matrix = torch.zeros(self.vocab_size, self.embedding_dim)
        for i in range(self.vocab_size):
            try:
                word = self.tokenizer.convert_ids_to_tokens(i)
                if word in word_vectors:
                    embedding_matrix[i] = torch.tensor(word_vectors[word])
            except:
                continue
        return embedding_matrix
    
    def adversarial_training(self, batch, criterion):
        """对抗训练步骤"""
        # 计算原始损失
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['label'].to(self.device)
        
        outputs, _ = self.model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        
        # 计算梯度
        loss.backward(retain_graph=True)
        
        # 获取嵌入层的梯度
        grad_embed = self.model.embedding.weight.grad.data
        
        # 生成对抗扰动
        perturb = self.epsilon * torch.sign(grad_embed)
        
        # 应用扰动
        self.model.embedding.weight.data.add_(perturb)
        
        # 计算对抗损失
        outputs_adv, _ = self.model(input_ids, attention_mask)
        loss_adv = criterion(outputs_adv, labels)
        
        # 恢复原始嵌入
        self.model.embedding.weight.data.sub_(perturb)
        
        return loss + self.alpha * loss_adv
    
    def train_logistic_regression(self, train_texts, train_labels, val_texts=None, val_labels=None):
        vectorizer = TfidfVectorizer(max_features=5000)
        X_train = vectorizer.fit_transform(train_texts)
        
        if val_texts is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, train_labels, test_size=0.2, 
                stratify=train_labels, 
                random_state=self.random_seed  # 添加随机种子
            )
        else:
            X_val = vectorizer.transform(val_texts)
            y_train, y_val = train_labels, val_labels
        
        lr_model = LogisticRegression(
            class_weight='balanced',
            random_state=self.random_seed  # 添加随机种子
        )
        lr_model.fit(X_train, y_train)
        
        val_pred = lr_model.predict(X_val)
        lr_accuracy = accuracy_score(y_val, val_pred)
        lr_f1 = f1_score(y_val, val_pred, average='macro')
        
        return lr_accuracy, lr_f1
    
    def train(self, train_texts, train_labels, val_texts=None, val_labels=None, 
              batch_size=16, epochs=10, learning_rate=2e-4):
        """训练模型"""
        logger.info("开始训练模型...")
        
        # 首先训练逻辑回归作为基线
        lr_accuracy, lr_f1 = self.train_logistic_regression(train_texts, train_labels, val_texts, val_labels)
        logger.info(f"逻辑回归基线模型 - 准确率: {lr_accuracy:.4f}, F1: {lr_f1:.4f}")
        
        # 如果数据量小于1000，进行数据增强
        if len(train_texts) < 1000:
            train_texts, train_labels = self.augmenter.augment(train_texts, train_labels)
            logger.info(f"数据增强后的训练集大小: {len(train_texts)}")
        
        # 创建K折交叉验证
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(train_texts, train_labels)):
            logger.info(f"训练第 {fold+1} 折...")
            
            # 重置模型
            self.model = self._create_model()
            optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
            scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2)
            criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0]).to(self.device))
            
            # 准备数据
            X_train, X_val = train_texts[train_idx], train_texts[val_idx]
            y_train, y_val = train_labels[train_idx], train_labels[val_idx]
            
            train_dataset = TextDataset(X_train, y_train, self.tokenizer)
            val_dataset = TextDataset(X_val, y_val, self.tokenizer)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            
            best_val_loss = float('inf')
            for epoch in range(epochs):
                # 训练和验证逻辑
                train_loss = self._train_epoch(train_loader, optimizer, criterion)
                val_loss, val_acc, val_f1 = self._validate(val_loader, criterion)
                
                scheduler.step(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    if self.model_save_path:
                        torch.save(self.model.state_dict(), 
                                 f"{self.model_save_path}_fold{fold}.pt")
                
                if self.early_stopping(val_loss):
                    break
            
            fold_results.append({
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'val_f1': val_f1
            })
        
        # 计算平均结果
        avg_val_loss = np.mean([res['val_loss'] for res in fold_results])
        avg_val_acc = np.mean([res['val_accuracy'] for res in fold_results])
        avg_val_f1 = np.mean([res['val_f1'] for res in fold_results])
        
        logger.info(f"交叉验证平均结果 - 损失: {avg_val_loss:.4f}, 准确率: {avg_val_acc:.4f}, F1: {avg_val_f1:.4f}")
        
        # 如果LSTM模型效果比逻辑回归差，给出警告
        if avg_val_acc < lr_accuracy:
            logger.warning("LSTM模型性能低于逻辑回归基线，建议使用逻辑回归模型")
        
        return avg_val_loss, avg_val_acc, avg_val_f1

    def _train_epoch(self, train_loader, optimizer, criterion):
        self.model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            
            # 使用对抗训练
            loss = self.adversarial_training(batch, criterion)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(train_loader)
    
    def _validate(self, val_loader, criterion):
        self.model.eval()
        total_loss = 0
        val_preds = []
        val_labels_list = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs, _ = self.model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels_list.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(val_labels_list, val_preds)
        f1 = f1_score(val_labels_list, val_preds, average='macro')
        
        return avg_loss, accuracy, f1
    
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
                
                outputs, _ = self.model(input_ids, attention_mask)
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
                
                outputs, _ = self.model(input_ids, attention_mask)
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