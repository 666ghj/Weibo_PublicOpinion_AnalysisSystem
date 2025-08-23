import torch
from transformers import BertTokenizer
from train import GPT2ClassifierWithAdapter
import re

def preprocess_text(text):
    """简单的文本预处理"""
    return text

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 使用本地模型路径而不是在线模型名称
    local_model_path = './models/gpt2-chinese'
    model_path = 'best_weibo_sentiment_model.pth'
    
    print(f"加载模型: {model_path}")
    # 从本地加载tokenizer
    tokenizer = BertTokenizer.from_pretrained(local_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = '[PAD]'
    
    # 加载模型，使用本地模型路径
    model = GPT2ClassifierWithAdapter(local_model_path)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print("\n============= 微博情感分析 =============")
    print("输入微博内容进行分析 (输入 'q' 退出):")
    
    while True:
        text = input("\n请输入微博内容: ")
        if text.lower() == 'q':
            break
        
        # 预处理文本
        processed_text = preprocess_text(text)
        
        # 对文本进行编码
        encoding = tokenizer(
            processed_text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 转移到设备
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        # 预测
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
        
        # 输出结果
        confidence = probabilities[0][prediction].item()
        label = "正面情感" if prediction == 1 else "负面情感"
        
        print(f"预测结果: {label} (置信度: {confidence:.4f})")

if __name__ == "__main__":
    main() 