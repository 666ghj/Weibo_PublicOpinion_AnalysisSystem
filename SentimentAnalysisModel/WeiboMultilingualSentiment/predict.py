import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import re

def preprocess_text(text):
    """简单的文本预处理，适用于多语言文本"""
    return text

def main():
    print("正在加载多语言情感分析模型...")
    
    # 使用多语言情感分析模型
    model_name = "tabularisai/multilingual-sentiment-analysis"
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
        
        # 情感标签映射（5级分类）
        sentiment_map = {
            0: "非常负面", 1: "负面", 2: "中性", 3: "正面", 4: "非常正面"
        }
        
    except Exception as e:
        print(f"模型加载失败: {e}")
        print("请检查网络连接")
        return
    
    print("\n============= 多语言情感分析 =============")
    print("支持语言: 中文、英文、西班牙文、阿拉伯文、日文、韩文等22种语言")
    print("情感等级: 非常负面、负面、中性、正面、非常正面")
    print("输入文本进行分析 (输入 'q' 退出):")
    print("输入 'demo' 查看多语言示例")
    
    while True:
        text = input("\n请输入文本: ")
        if text.lower() == 'q':
            break
        
        if text.lower() == 'demo':
            show_multilingual_demo(tokenizer, model, device, sentiment_map)
            continue
        
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
            label = sentiment_map[prediction]
            
            print(f"预测结果: {label} (置信度: {confidence:.4f})")
            
            # 显示所有类别的概率
            print("详细概率分布:")
            for i, (label_name, prob) in enumerate(zip(sentiment_map.values(), probabilities[0])):
                print(f"  {label_name}: {prob:.4f}")
            
        except Exception as e:
            print(f"预测时发生错误: {e}")
            continue

def show_multilingual_demo(tokenizer, model, device, sentiment_map):
    """展示多语言情感分析示例"""
    print("\n=== 多语言情感分析示例 ===")
    
    demo_texts = [
        # 中文
        ("今天天气真好，心情特别棒！", "中文"),
        ("这家餐厅的菜味道非常棒！", "中文"),
        ("服务态度太差了，很失望", "中文"),
        
        # 英文
        ("I absolutely love this product!", "英文"),
        ("The customer service was disappointing.", "英文"),
        ("The weather is fine, nothing special.", "英文"),
        
        # 日文
        ("このレストランの料理は本当に美味しいです！", "日文"),
        ("このホテルのサービスはがっかりしました。", "日文"),
        
        # 韩文
        ("이 가게의 케이크는 정말 맛있어요！", "韩文"),
        ("서비스가 너무 별로였어요。", "韩文"),
        
        # 西班牙文
        ("¡Me encanta cómo quedó la decoración!", "西班牙文"),
        ("El servicio fue terrible y muy lento.", "西班牙文"),
    ]
    
    for text, language in demo_texts:
        try:
            inputs = tokenizer(
                text,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
            
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)
                prediction = torch.argmax(probabilities, dim=1).item()
            
            confidence = probabilities[0][prediction].item()
            label = sentiment_map[prediction]
            
            print(f"\n{language}: {text}")
            print(f"结果: {label} (置信度: {confidence:.4f})")
            
        except Exception as e:
            print(f"处理 {text} 时出错: {e}")
    
    print("\n=== 示例结束 ===")
    
    '''
    正在加载多语言情感分析模型...
从本地加载模型...
模型加载成功! 使用设备: cuda

============= 多语言情感分析 =============
支持语言: 中文、英文、西班牙文、阿拉伯文、日文、韩文等22种语言
情感等级: 非常负面、负面、中性、正面、非常正面
输入文本进行分析 (输入 'q' 退出):
输入 'demo' 查看多语言示例

请输入文本: 我喜欢你
C:\Users\67093\.conda\envs\pytorch_python11\Lib\site-packages\transformers\models\distilbert\modeling_distilbert.py:401: UserWarning: 1Torch was not compiled with flash attention. (Triggered internally at C:\cb\pytorch_1000000000000\work\aten\src\ATen\native\transformers\cuda\sdp_utils.cpp:263.)
  attn_output = torch.nn.functional.scaled_dot_product_attention(
预测结果: 正面 (置信度: 0.5204)
详细概率分布:
  非常负面: 0.0329
  负面: 0.0263
  中性: 0.1987
  正面: 0.5204
  非常正面: 0.2216

请输入文本:
    '''

if __name__ == "__main__":
    main()