import os
from transformers.models.bert import BertTokenizer, BertModel
import torch
from tqdm import tqdm
import numpy as np
import jieba

class BERT_CTM_Model:
    def __init__(self, bert_model_path):
        # 加载BERT模型和tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_path)
        self.model = BertModel.from_pretrained(bert_model_path)

    def get_bert_embeddings(self, texts):
        """使用BERT模型批量生成文本的嵌入向量"""
        embeddings = []
        for text in tqdm(texts, desc="Processing texts with BERT"):
            inputs = self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=80)
            with torch.no_grad():
                outputs = self.model(**inputs)
            embeddings.append(outputs.last_hidden_state.cpu().numpy())  # [batch_size, sequence_length, hidden_size]
        return np.vstack(embeddings)
    
    def chinese_tokenize(self, text):
        """使用jieba对中文文本进行分词"""
        return " ".join(jieba.cut(text))

if __name__ == "__main__":
    model = BERT_CTM_Model('./bert_model')
    text = "这是一个测试文本"
    tokenized_text = model.chinese_tokenize(text)
    print(tokenized_text)
