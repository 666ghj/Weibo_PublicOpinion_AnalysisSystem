import torch
from transformers import GPT2ForSequenceClassification, BertTokenizer
from peft import PeftModel
import os
import re

def preprocess_text(text):
    return text

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 模型和权重路径
    base_model_path = './models/gpt2-chinese'
    lora_model_path = './best_weibo_sentiment_lora'
    
    print("加载模型和tokenizer...")
    
    # 检查LoRA模型是否存在
    if not os.path.exists(lora_model_path):
        print(f"错误: 找不到LoRA模型路径 {lora_model_path}")
        print("请先运行 train.py 进行训练")
        return
    
    # 加载tokenizer
    try:
        tokenizer = BertTokenizer.from_pretrained(base_model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = '[PAD]'
    except Exception as e:
        print(f"加载tokenizer失败: {e}")
        print("请确保models/gpt2-chinese目录包含tokenizer文件")
        return
    
    # 加载基础模型
    try:
        base_model = GPT2ForSequenceClassification.from_pretrained(
            base_model_path, 
            num_labels=2
        )
        base_model.config.pad_token_id = tokenizer.pad_token_id
    except Exception as e:
        print(f"加载基础模型失败: {e}")
        print("请确保models/gpt2-chinese目录包含模型文件")
        return
    
    # 加载LoRA权重
    try:
        model = PeftModel.from_pretrained(base_model, lora_model_path)
        model.to(device)
        model.eval()
        print("LoRA模型加载成功!")
    except Exception as e:
        print(f"加载LoRA权重失败: {e}")
        print("请确保LoRA权重文件存在且格式正确")
        return
    
    print("\n============= 微博情感分析 (LoRA版) =============")
    print("输入微博内容进行分析 (输入 'q' 退出):")
    
    while True:
        text = input("\n请输入微博内容: ")
        if text.lower() == 'q':
            break
        
        if not text.strip():
            print("输入不能为空，请重新输入")
            continue
        
        try:
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
            
        except Exception as e:
            print(f"预测时发生错误: {e}")
            continue

if __name__ == "__main__":
    main()