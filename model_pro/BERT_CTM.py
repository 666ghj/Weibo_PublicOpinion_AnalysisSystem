import os
from transformers.models.bert import BertTokenizer, BertModel
import torch
from tqdm import tqdm
import numpy as np
import jieba
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
from contextualized_topic_models.models.ctm import CombinedTM

class BERT_CTM_Model:
    def __init__(self, bert_model_path, ctm_tokenizer_path, n_components=12, num_epochs=50, device=None):
        # 确定设备 (CPU/GPU)
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # 检查模型路径是否存在
        if not os.path.exists(bert_model_path):
            raise ValueError(f"BERT模型路径不存在: {bert_model_path}")
        if not os.path.exists(ctm_tokenizer_path):
            raise ValueError(f"CTM分词器路径不存在: {ctm_tokenizer_path}")

        # 加载BERT模型和tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_path)
        self.model = BertModel.from_pretrained(bert_model_path).to(self.device)

        # 创建CTM数据预处理对象
        self.tp = TopicModelDataPreparation(ctm_tokenizer_path)
        self.n_components = n_components
        self.num_epochs = num_epochs
        self.ctm_model = None

    def get_bert_embeddings(self, texts):
        """使用BERT模型批量生成文本的嵌入向量"""
        embeddings = []
        for text in tqdm(texts, desc="Processing texts with BERT"):
            inputs = self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=80).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            embeddings.append(outputs.last_hidden_state[:, 0, :].cpu().numpy())  # [batch_size, hidden_size]
        return np.vstack(embeddings)

    def chinese_tokenize(self, text):
        """使用jieba对中文文本进行分词"""
        return " ".join(jieba.cut(text))

    def train_ctm(self, texts):
        """训练CTM模型"""
        try:
            # 分词并准备BOW文本
            bow_texts = [self.chinese_tokenize(text) for text in texts]
            training_dataset = self.tp.fit(text_for_contextual=texts, text_for_bow=bow_texts)

            # 训练CTM
            self.ctm_model = CombinedTM(bow_size=len(self.tp.vocab), contextual_size=768, 
                                        n_components=self.n_components, num_epochs=self.num_epochs)
            self.ctm_model.fit(training_dataset)
            print("CTM模型训练完成")
        except Exception as e:
            print(f"训练CTM模型时发生错误: {e}")

    def predict(self, texts):
        """使用训练好的CTM模型预测新文本的主题分布"""
        if not self.ctm_model:
            raise ValueError("模型尚未训练或加载，无法进行预测")
        
        try:
            bow_texts = [self.chinese_tokenize(text) for text in texts]
            testing_dataset = self.tp.transform(text_for_contextual=texts, text_for_bow=bow_texts)
            topic_distributions = self.ctm_model.get_doc_topic_distribution(testing_dataset)
            return topic_distributions
        except Exception as e:
            print(f"预测主题时发生错误: {e}")
            return None

    def save_model(self, path):
        """保存训练后的CTM模型"""
        if self.ctm_model:
            self.ctm_model.save(path)
            print(f"CTM模型已保存至: {path}")
        else:
            print("未找到已训练的CTM模型，无法保存")

    def load_model(self, path):
        """加载已保存的CTM模型"""
        if os.path.exists(path):
            self.ctm_model = CombinedTM.load(path)
            print(f"CTM模型已加载自: {path}")
        else:
            print(f"无法加载模型，路径不存在: {path}")

if __name__ == "__main__":
    # 设定BERT和CTM模型的路径
    bert_model_path = './bert_model'
    ctm_tokenizer_path = './sentence_bert_model'
    
    # 初始化模型
    model = BERT_CTM_Model(bert_model_path, ctm_tokenizer_path)

    # 示例文本
    texts = ["这是第一个文本", "这是第二个文本"]
    
    # 训练CTM模型
    model.train_ctm(texts)

    # 保存CTM模型
    model.save_model('./trained_ctm_model')

    # 加载CTM模型
    model.load_model('./trained_ctm_model')

    # 预测新文本的主题分布
    new_texts = ["这是一个新的文本", "另外一个新文本"]
    topic_distributions = model.predict(new_texts)

    # 输出预测结果
    if topic_distributions is not None:
        for idx, distribution in enumerate(topic_distributions):
            print(f"文本 {idx+1} 的主题分布: {distribution}")
