import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pandas as pd
from tqdm import tqdm
from transformers.models.bert import BertTokenizer, BertModel
from contextualized_topic_models.models.ctm import CombinedTM
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
from contextualized_topic_models.utils.preprocessing import WhiteSpacePreprocessing
import numpy as np
import torch
import jieba
import pickle  # 用于保存和加载模型
from utils.logger import model_logger as logging

class BERT_CTM:
    def __init__(self, model_save_path='model_pro/saved_models/ctm_model.pkl'):
        self.model_save_path = model_save_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bert_model = None
        self.tokenizer = None
        self.ctm_model = None
        self.vocab = None
        self.vectorizer = None
        
    def save_model(self):
        """保存模型和词袋"""
        try:
            with open(self.model_save_path, 'wb') as f:
                pickle.dump({
                    'ctm_model': self.ctm_model,
                    'vocab': self.vocab,
                    'vectorizer': self.vectorizer
                }, f)
            logging.info(f"CTM模型和词袋保存到: {self.model_save_path}")
        except Exception as e:
            logging.error(f"保存模型时发生错误: {e}")
            
    def load_model(self):
        """加载模型和词袋"""
        try:
            with open(self.model_save_path, 'rb') as f:
                saved_data = pickle.load(f)
                self.ctm_model = saved_data['ctm_model']
                self.vocab = saved_data['vocab']
                self.vectorizer = saved_data['vectorizer']
            logging.info("CTM模型、词袋和vectorizer加载成功")
        except Exception as e:
            logging.error(f"加载模型时发生错误: {e}")
            raise
            
    def train(self, texts, num_topics=10, num_epochs=100):
        """训练CTM模型"""
        try:
            # 初始化BERT
            if not self.bert_model:
                self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
                self.bert_model = BertModel.from_pretrained('bert-base-chinese').to(self.device)
                
            # 提取BERT嵌入
            logging.info("正在提取BERT嵌入...")
            embeddings = self._get_bert_embeddings(texts)
            
            # 准备CTM数据
            logging.info("正在准备CTM训练数据...")
            preprocessor = WhiteSpacePreprocessing(texts)
            dataset = TopicModelDataPreparation(embeddings)
            
            # 训练CTM模型
            logging.info("正在训练CTM模型...")
            self.ctm_model = CombinedTM(
                bow_size=len(preprocessor.vocab),
                contextual_size=768,  # BERT输出维度
                n_components=num_topics,
                num_epochs=num_epochs
            )
            self.ctm_model.fit(dataset)
            
            # 保存词袋相关数据
            self.vocab = preprocessor.vocab
            self.vectorizer = preprocessor.vectorizer
            
            # 保存模型
            self.save_model()
            logging.info("模型训练完成并保存")
            
        except Exception as e:
            logging.error(f"训练模型时发生错误: {e}")
            raise
            
    def _get_bert_embeddings(self, texts):
        """获取文本的BERT嵌入"""
        embeddings = []
        try:
            for text in texts:
                inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.bert_model(**inputs)
                    # 使用[CLS]标记的输出作为文档表示
                    embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(embedding[0])
                
            return np.array(embeddings)
        except Exception as e:
            logging.error(f"获取BERT嵌入时发生错误: {e}")
            raise
            
    def get_topics(self, num_words=10):
        """获取主题词"""
        try:
            if not self.ctm_model or not self.vocab:
                raise ValueError("模型未训练或未加载")
                
            topics = []
            for topic_idx in range(self.ctm_model.n_components):
                topic = self.ctm_model.get_topic_lists(top_n=num_words)[topic_idx]
                topics.append(topic)
            return topics
        except Exception as e:
            logging.error(f"获取主题词时发生错误: {e}")
            raise

if __name__ == "__main__":
    # 创建BERT_CTM实例
    model = BERT_CTM(
        model_save_path='model_pro/saved_models/ctm_model.pkl',  # 保存路径
    )

    # 传入CSV文件路径进行训练
    model.train("./train.csv")
