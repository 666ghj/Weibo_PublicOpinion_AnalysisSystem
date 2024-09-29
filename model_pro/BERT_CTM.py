import os
from transformers.models.bert import BertTokenizer, BertModel
import torch

class BERT_CTM_Model:
    def __init__(self, bert_model_path):
        # 加载BERT模型和tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_path)
        self.model = BertModel.from_pretrained(bert_model_path)

    def get_bert_embeddings(self, text):
        """使用BERT模型生成文本的嵌入向量"""
        inputs = self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=80)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.cpu().numpy()  # [batch_size, sequence_length, hidden_size]

if __name__ == "__main__":
    model = BERT_CTM_Model('./bert_model')
    text = "这是一个测试文本"
    embedding = model.get_bert_embeddings(text)
    print(embedding.shape)
