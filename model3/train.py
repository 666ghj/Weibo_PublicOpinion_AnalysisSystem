import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, GPT2ForSequenceClassification, BertTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from adapter import AdapterLayer
from gpt2_adapter import GPT2BlockWithAdapter

# 设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# 定义微博情感分析数据集
class WeiboSentimentDataset(Dataset):
    def __init__(self, reviews, labels, tokenizer, max_length=128):
        self.reviews = reviews
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.reviews)
    
    def __getitem__(self, idx):
        review = str(self.reviews[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            review,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 定义GPT2分类模型，带Adapter
class GPT2ClassifierWithAdapter(nn.Module):
    def __init__(self, pretrained_model_name, num_labels=2):
        super(GPT2ClassifierWithAdapter, self).__init__()
        # 加载预训练模型
        self.gpt2 = GPT2ForSequenceClassification.from_pretrained(
            pretrained_model_name,
            num_labels=num_labels
        )
        
        # 确保模型配置中设置了pad_token_id
        self.gpt2.config.pad_token_id = self.gpt2.config.eos_token_id
        
        # 替换原始的GPT2Block为带Adapter的版本
        config = self.gpt2.config
        for i in range(len(self.gpt2.transformer.h)):
            # 保存原始权重
            old_block = self.gpt2.transformer.h[i]
            # 创建带Adapter的新Block
            new_block = GPT2BlockWithAdapter(config)
            # 复制原始权重
            new_block.load_state_dict(old_block.state_dict(), strict=False)
            # 替换
            self.gpt2.transformer.h[i] = new_block
            
        # 冻结原始GPT2参数
        for param in self.gpt2.parameters():
            param.requires_grad = False
            
        # 解冻分类器层和Adapter层参数
        for param in self.gpt2.score.parameters():
            param.requires_grad = True
            
        # 解冻所有Adapter层
        for i in range(len(self.gpt2.transformer.h)):
            for param in self.gpt2.transformer.h[i].adapter.parameters():
                param.requires_grad = True
    
    def forward(self, input_ids, attention_mask, labels=None):
        return self.gpt2(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

# 训练函数
def train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, device, epochs=3):
    best_f1 = 0.0
    
    for epoch in range(epochs):
        print(f"======== Epoch {epoch+1} / {epochs} ========")
        model.train()
        total_loss = 0
        
        # 训练循环
        progress_bar = tqdm(train_dataloader, desc="Training", position=0, leave=True)
        for batch in progress_bar:
            # 将数据移到GPU
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # 参数更新
            optimizer.step()
            scheduler.step()
            
            # 更新进度条
            progress_bar.set_postfix({"loss": loss.item()})
        
        # 计算平均训练损失
        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Average training loss: {avg_train_loss:.4f}")
        
        # 评估模型
        val_metrics = evaluate_model(model, val_dataloader, device)
        print(f"Validation Loss: {val_metrics['loss']:.4f}")
        print(f"Validation Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"Validation F1 Score: {val_metrics['f1']:.4f}")
        
        # 保存最佳模型
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            torch.save(model.state_dict(), "best_weibo_sentiment_model.pth")
            print("Saved best model!")

# 评估函数
def evaluate_model(model, dataloader, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # 获取预测结果
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            labels = batch['labels'].cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels)
    
    # 计算评估指标
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    avg_loss = total_loss / len(dataloader)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'f1': f1
    }

def main():
    # 设置模型本地保存路径
    model_name = 'uer/gpt2-chinese-cluecorpussmall'
    local_model_path = './models/gpt2-chinese'
    
    # 确保目录存在
    os.makedirs(local_model_path, exist_ok=True)
    
    # 加载数据集
    print("加载微博情感数据集...")
    df = pd.read_csv('dataset/weibo_senti_100k.csv')
    
    # 分割数据集
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df['label'])
    
    # 加载tokenizer和模型
    print("加载预训练模型和tokenizer...")
    
    # 检查本地是否已有模型
    if os.path.exists(os.path.join(local_model_path, 'config.json')):
        print(f"从本地路径加载模型: {local_model_path}")
        tokenizer = BertTokenizer.from_pretrained(local_model_path)
    else:
        print(f"从Hugging Face下载模型到: {local_model_path}")
        tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir=local_model_path)
        # 保存tokenizer到本地
        tokenizer.save_pretrained(local_model_path)
    
    # 设置padding token (BertTokenizer通常已有[PAD]作为padding token)
    if tokenizer.pad_token is None:
        # 如果没有，显式设置为[PAD]
        tokenizer.pad_token = '[PAD]'
    
    # 记录pad_token的ID，确保模型和tokenizer使用相同的pad_token_id
    pad_token_id = tokenizer.pad_token_id
    
    # 创建数据集
    train_dataset = WeiboSentimentDataset(
        train_df['review'].values,
        train_df['label'].values,
        tokenizer
    )
    
    val_dataset = WeiboSentimentDataset(
        val_df['review'].values,
        val_df['label'].values,
        tokenizer
    )
    
    # 创建数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 初始化模型
    if (os.path.exists(os.path.join(local_model_path, 'pytorch_model.bin')) or 
        os.path.exists(os.path.join(local_model_path, 'model.safetensors'))):
        print(f"从本地路径加载模型权重: {local_model_path}")
        model = GPT2ClassifierWithAdapter(local_model_path)
    else:
        print(f"从Hugging Face下载模型权重到: {local_model_path}")
        # 直接从Hugging Face下载并保存完整模型
        temp_model = GPT2ForSequenceClassification.from_pretrained(model_name)
        temp_model.save_pretrained(local_model_path)
        # 然后用保存的模型创建GPT2ClassifierWithAdapter
        model = GPT2ClassifierWithAdapter(local_model_path)
    
    # 确保模型使用与tokenizer相同的pad_token_id
    model.gpt2.config.pad_token_id = pad_token_id
    model.to(device)
    
    # 统计需要训练的参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"模型总参数量: {total_params}")
    print(f"需要训练的参数量: {trainable_params} ({trainable_params/total_params*100:.2f}%)")
    
    # 设置优化器和学习率调度器
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=5e-5,
        eps=1e-8
    )
    
    # 设置总训练步数和warmup步数
    total_steps = len(train_dataloader) * 2  # 2个epoch
    warmup_steps = int(total_steps * 0.1)  # 10%的warmup
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # 训练模型
    print("开始训练...")
    train_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=2
    )
    
    print("训练完成!")

if __name__ == "__main__":
    main() 