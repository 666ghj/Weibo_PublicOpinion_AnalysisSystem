import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import re

def preprocess_text(text):
    return text

def main():
    print("正在加载微博情感分析模型...")
    
    # 使用HuggingFace预训练模型
    model_name = "wsqstar/GISchat-weibo-100k-fine-tuned-bert"
    local_model_path = "./model"
    
    try:
        # 检查本地是否已有模型
        import os
        if os.path.exists(local_model_path):
            print("从本地加载模型...")
            tokenizer = AutoTokenizer.from_pretrained(local_model_path)
            model = AutoModelForSequenceClassification.from_pretrained(local_model_path)
        else:
            print("首次使用，正在下载模型到本地...")
            # 下载并保存到本地
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            # 保存到本地
            tokenizer.save_pretrained(local_model_path)
            model.save_pretrained(local_model_path)
            print(f"模型已保存到: {local_model_path}")
        
        # 设置设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        print(f"模型加载成功! 使用设备: {device}")
        
    except Exception as e:
        print(f"模型加载失败: {e}")
        print("请检查网络连接或使用pipeline方式")
        return
    
    print("\n============= 微博情感分析 =============")
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
            
            # 分词编码
            inputs = tokenizer(
                processed_text,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
            
            # 转移到设备
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # 预测
            with torch.no_grad():
                outputs = model(**inputs)
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