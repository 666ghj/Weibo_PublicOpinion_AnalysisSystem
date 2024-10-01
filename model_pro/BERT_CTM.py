import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pandas as pd
from tqdm import tqdm
from transformers.models.bert import BertTokenizer, BertModel
from contextualized_topic_models.models.ctm import CombinedTM
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
import numpy as np
import torch
import jieba
import pickle  # 用于保存和加载模型

class BERT_CTM_Model:
    def __init__(self, bert_model_path, ctm_tokenizer_path, n_components=12, num_epochs=50, model_save_path='./ctm_model'):
        self.bert_model_path = bert_model_path
        self.ctm_tokenizer_path = ctm_tokenizer_path
        self.n_components = n_components
        self.num_epochs = num_epochs
        self.model_save_path = model_save_path
        # 加载BERT模型和tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_model_path)
        self.model = BertModel.from_pretrained(self.bert_model_path)

        # 创建CTM数据预处理对象
        self.tp = TopicModelDataPreparation(self.ctm_tokenizer_path)

    def chinese_tokenize(self, text):
        """使用jieba对中文文本进行分词"""
        return " ".join(jieba.cut(text))
    
    def get_bert_embeddings(self, texts):
        """使用BERT模型生成文本的嵌入向量"""
        embeddings = []
        for text in tqdm(texts, desc="Processing texts with BERT"):
            inputs = self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=80)
            with torch.no_grad():
                outputs = self.model(**inputs)
            embeddings.append(outputs.last_hidden_state.cpu().numpy())  # [batch_size, sequence_length, hidden_size]
        return np.vstack(embeddings)

    def save_model(self, ctm):
        """保存CTM模型、词袋和BoW的vectorizer"""
        os.makedirs(self.model_save_path, exist_ok=True)
        with open(f"{self.model_save_path}/ctm_model.pkl", 'wb') as f:
            pickle.dump(ctm, f)
        with open(f"{self.model_save_path}/vocab.pkl", 'wb') as f:
            pickle.dump(self.tp.vocab, f)
        with open(f"{self.model_save_path}/vectorizer.pkl", 'wb') as f:  # 保存BoW的vectorizer
            pickle.dump(self.tp.vectorizer, f)
        print(f"CTM模型和词袋保存到: {self.model_save_path}")

    def load_model(self):
        """加载CTM模型、词袋和BoW的vectorizer"""
        with open(f"{self.model_save_path}/ctm_model.pkl", 'rb') as f:
            ctm = pickle.load(f)
        with open(f"{self.model_save_path}/vocab.pkl", 'rb') as f:
            self.tp.vocab = pickle.load(f)
        with open(f"{self.model_save_path}/vectorizer.pkl", 'rb') as f:  # 加载BoW的vectorizer
            self.tp.vectorizer = pickle.load(f)
        print(f"CTM模型、词袋和vectorizer加载成功")
        return ctm

    def train(self, csv_file):
        """训练BERT + CTM模型并保存最终的特征向量和标签"""
        # 读取CSV文件中的文本和标签
        data = pd.read_csv(csv_file)
        texts = data['TEXT'].tolist()
        labels = data['label'].tolist()

        # Step 1: 获取BERT的嵌入向量
        print("Extracting BERT embeddings...")
        bert_embeddings = self.get_bert_embeddings(texts)  # [batch_size, sequence_length, hidden_size]

        # Step 2: 准备CTM数据
        print("Preparing data for CTM using training set...")
        bow_texts = [self.chinese_tokenize(text) for text in texts]
        training_dataset = self.tp.fit(text_for_contextual=texts, text_for_bow=bow_texts)

        # Step 3: 替换BERT嵌入
        training_dataset._X = bert_embeddings[:, 0, :]  # 只使用第一个token的向量用于CTM

        # Step 4: 训练CTM模型
        print("Training CTM model...")
        ctm = CombinedTM(bow_size=len(self.tp.vocab), contextual_size=768, n_components=self.n_components, num_epochs=self.num_epochs)
        ctm.fit(train_dataset=training_dataset, verbose=True)

        # Step 5: 保存CTM模型和词袋
        self.save_model(ctm)

        # Step 6: 获取CTM的特征向量
        print("Generating CTM features...")
        ctm_features = ctm.get_doc_topic_distribution(training_dataset)  # [batch_size, n_components]

        # Step 7: 将CTM特征扩展为与BERT的sequence长度一致
        sequence_length = bert_embeddings.shape[1]
        ctm_features_expanded = np.repeat(ctm_features[:, np.newaxis, :], sequence_length, axis=1)  # [batch_size, sequence_length, n_components]

        # Step 8: 拼接BERT嵌入和CTM特征
        final_embeddings = np.concatenate([bert_embeddings, ctm_features_expanded], axis=-1)  # [batch_size, sequence_length, hidden_size + n_components]

        return bert_embeddings

if __name__ == "__main__":
    # 创建BERT_CTM_Model实例
    model = BERT_CTM_Model(
        bert_model_path='./bert_model',  # BERT模型的路径
        ctm_tokenizer_path='./sentence_bert_model',  # CTM分词器的路径
        n_components=12,  # 主题数量
        num_epochs=50,  # 训练轮次
        model_save_path='./ctm_model',  # 保存路径
    )

    # 传入CSV文件路径进行训练
    model.train("./train.csv")
