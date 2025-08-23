from transformers import pipeline
import re

def preprocess_text(text):
    """简单的文本预处理"""
    text = re.sub(r"\{%.+?%\}", " ", text)           # 去除 {%xxx%}
    text = re.sub(r"@.+?( |$)", " ", text)           # 去除 @xxx
    text = re.sub(r"【.+?】", " ", text)              # 去除 【xx】
    text = re.sub(r"\u200b", " ", text)              # 去除特殊字符
    # 删除表情符号
    text = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002600-\U000027BF\U0001f900-\U0001f9ff\U0001f018-\U0001f270\U0000231a-\U0000231b\U0000238d-\U0000238d\U000024c2-\U0001f251]+', '', text)
    text = re.sub(r"\s+", " ", text)                 # 多个空格合并
    return text.strip()

def main():
    print("正在加载微博情感分析模型...")
    
    # 使用pipeline方式 - 更简单
    model_name = "wsqstar/GISchat-weibo-100k-fine-tuned-bert"
    local_model_path = "./model"
    
    try:
        # 检查本地是否已有模型
        import os
        if os.path.exists(local_model_path):
            print("从本地加载模型...")
            classifier = pipeline(
                "text-classification", 
                model=local_model_path,
                return_all_scores=True
            )
        else:
            print("首次使用，正在下载模型到本地...")
            # 先下载模型
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            # 保存到本地
            tokenizer.save_pretrained(local_model_path)
            model.save_pretrained(local_model_path)
            print(f"模型已保存到: {local_model_path}")
            
            # 使用本地模型创建pipeline
            classifier = pipeline(
                "text-classification", 
                model=local_model_path,
                return_all_scores=True
            )
        print("模型加载成功!")
        
    except Exception as e:
        print(f"模型加载失败: {e}")
        print("请检查网络连接")
        return
    
    print("\n============= 微博情感分析 (Pipeline版) =============")
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
            
            # 预测
            outputs = classifier(processed_text)
            
            # 解析结果
            positive_score = None
            negative_score = None
            
            for output in outputs[0]:
                if output['label'] == 'LABEL_1':  # 正面
                    positive_score = output['score']
                elif output['label'] == 'LABEL_0':  # 负面
                    negative_score = output['score']
            
            # 确定预测结果
            if positive_score > negative_score:
                label = "正面情感"
                confidence = positive_score
            else:
                label = "负面情感"
                confidence = negative_score
            
            print(f"预测结果: {label} (置信度: {confidence:.4f})")
            
        except Exception as e:
            print(f"预测时发生错误: {e}")
            continue

if __name__ == "__main__":
    main()