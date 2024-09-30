import os
from transformers.models.bert import BertTokenizer, BertModel
import torch
from tqdm import tqdm
import numpy as np
import jieba
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
from contextualized_topic_models.models.ctm import CombinedTM

class BERT_CTM_Model:
    def __init__(self, bert_model_path, ctm_tokenizer_path, n_components=12, num_epochs=50):
        # 加载BERT模型和tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_path)
        self.model = BertModel.from_pretrained(bert_model_path)

        # 创建CTM数据预处理对象
        self.tp = TopicModelDataPreparation(ctm_tokenizer_path)
        self.n_components = n_components
        self.num_epochs = num_epochs

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

    def train_ctm(self, texts):
        """训练CTM模型"""
        bow_texts = [self.chinese_tokenize(text) for text in texts]
        training_dataset = self.tp.fit(text_for_contextual=texts, text_for_bow=bow_texts)

        # 训练CTM
        ctm = CombinedTM(bow_size=len(self.tp.vocab), contextual_size=768, n_components=self.n_components, num_epochs=self.num_epochs)
        ctm.fit(training_dataset)
        print("CTM模型训练完成")

if __name__ == "__main__":
    model = BERT_CTM_Model('./bert_model', './sentence_bert_model')
    texts = ["这是第一个文本", "这是第二个文本"]
    model.train_ctm(texts)
