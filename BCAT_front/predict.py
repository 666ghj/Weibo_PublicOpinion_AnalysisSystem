import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os
import sys
import json
import chardet

# 导入改进版模型的组件
from model_pro.MHA import MultiHeadAttentionLayer
from model_pro.classifier import FinalClassifier
from model_pro.BERT_CTM import BERT_CTM_Model

class ModelManager:
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.classifier_model = None
            self.attention_model = None
            self.bert_ctm_model = None
            self._initialized = True
    
    def load_models(self, model_save_path, bert_model_path, ctm_tokenizer_path):
        """加载所有需要的模型"""
        try:
            if self.classifier_model is None:
                self.classifier_model = torch.load(model_save_path, map_location=self.device)
                self.classifier_model.eval()
                
            if self.attention_model is None:
                self.attention_model = MultiHeadAttentionLayer(embed_size=768, num_heads=8)
                self.attention_model.to(self.device)
                self.attention_model.eval()
                
            if self.bert_ctm_model is None:
                self.bert_ctm_model = BERT_CTM_Model(
                    bert_model_path=bert_model_path,
                    ctm_tokenizer_path=ctm_tokenizer_path
                )
            return True
        except Exception as e:
            print(f"模型加载失败: {e}")
            return False
    
    def predict_batch(self, texts, batch_size=32):
        """批量预测文本情感"""
        try:
            all_predictions = []
            all_probabilities = []
            
            # 分批处理文本
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # 获取文本嵌入
                embeddings = self.bert_ctm_model.get_bert_embeddings(batch_texts)
                
                # 转换为tensor
                batch_x = torch.tensor(embeddings, dtype=torch.float32).to(self.device)
                batch_x = torch.mean(batch_x, dim=1)
                
                with torch.no_grad():
                    # 使用注意力机制
                    attention_output = self.attention_model(batch_x, batch_x, batch_x)
                    # 获取分类结果
                    outputs = self.classifier_model(attention_output)
                    outputs = torch.mean(outputs, dim=1)
                    # 获取预测概率
                    probabilities = torch.softmax(outputs, dim=1)
                    # 获取预测标签
                    _, predicted = torch.max(outputs, 1)
                    
                    all_predictions.extend(predicted.cpu().numpy())
                    all_probabilities.extend(probabilities.cpu().numpy())
                    
            return all_predictions, all_probabilities
        except Exception as e:
            print(f"预测过程中出现错误: {e}")
            return None, None

# 创建全局的模型管理器实例
model_manager = ModelManager()

def detect_file_encoding(file_path, num_bytes=10000):
    """
    使用 chardet 检测文件的编码。

    :param file_path: 文件路径
    :param num_bytes: 用于检测的字节数
    :return: 检测到的编码
    """
    with open(file_path, 'rb') as f:
        rawdata = f.read(num_bytes)
    result = chardet.detect(rawdata)
    encoding = result['encoding']
    confidence = result['confidence']
    print(f"检测到的编码: {encoding}, 置信度: {confidence}")
    return encoding


def get_bert_ctm_embeddings(texts, bert_model_path, ctm_tokenizer_path, n_components=12, num_epochs=20):
    # 创建BERT_CTM_Model实例
    bert_ctm_model = BERT_CTM_Model(
        bert_model_path=bert_model_path,
        ctm_tokenizer_path=ctm_tokenizer_path,
        n_components=n_components,
        num_epochs=num_epochs
    )
    # 获取嵌入
    embeddings = bert_ctm_model.get_bert_embeddings(texts)
    return embeddings


def prepare_dataloader(features, batch_size):
    tensor_x = torch.tensor(features, dtype=torch.float32)
    dataset = TensorDataset(tensor_x)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def predict(model_save_path, input_data_path, output_path, bert_model_path, ctm_tokenizer_path, stats_output_path,
            batch_size=128,
            num_classes=2):
    try:
        # 加载模型
        print("加载模型...")
        if not model_manager.load_models(model_save_path, bert_model_path, ctm_tokenizer_path):
            return False

        # 检测文件编码
        encoding = detect_file_encoding(input_data_path)

        # 读取输入数据
        print("读取输入数据...")
        data = pd.read_csv(input_data_path, encoding=encoding)
        texts = data['TEXT'].tolist()

        # 生成嵌入
        print("生成文本嵌入...")
        embeddings = get_bert_ctm_embeddings(texts, bert_model_path, ctm_tokenizer_path)

        # 准备DataLoader
        data_loader = prepare_dataloader(embeddings, batch_size)

        # 存储预测结果
        all_predictions = []
        all_probabilities = []

        print("开始预测...")
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="预测进度"):
                batch_x = batch[0].to(model_manager.device)
                batch_x = torch.mean(batch_x, dim=1)
                
                # 使用注意力机制
                attention_output = model_manager.attention_model(batch_x, batch_x, batch_x)
                
                # 获取分类结果
                outputs = model_manager.classifier_model(attention_output)
                outputs = torch.mean(outputs, dim=1)
                
                # 获取预测概率
                probabilities = torch.softmax(outputs, dim=1)
                
                # 获取预测标签
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

        # 添加预测结果和概率到数据框
        data['Predicted_Label'] = all_predictions
        data['Confidence'] = [prob[pred] for prob, pred in zip(all_probabilities, all_predictions)]

        # 保存预测结果
        data.to_csv(output_path, index=False, encoding='utf-8')
        print(f"预测结果已保存到 {output_path}")

        # 统计标签的个数和占比
        label_counts = data['Predicted_Label'].value_counts()
        total_count = len(data)
        stats = {
            '统计信息': {
                '总样本数': total_count,
                '各类别统计': {}
            }
        }
        
        for label, count in label_counts.items():
            label_name = "良好" if label == 0 else "不良"
            percentage = (count / total_count) * 100
            confidence_mean = data[data['Predicted_Label'] == label]['Confidence'].mean()
            
            stats['统计信息']['各类别统计'][label_name] = {
                '数量': int(count),
                '占比': f"{percentage:.2f}%",
                '平均置信度': f"{confidence_mean:.2f}"
            }
            print(f"标签: {label_name}, 数量: {count}, 占比: {percentage:.2f}%, 平均置信度: {confidence_mean:.2f}")

        # 将统计信息保存到 JSON 文件
        with open(stats_output_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=4)

        return True
    except Exception as e:
        print(f"预测过程中出现错误: {e}")
        return False


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("使用方法: python predict.py <input_data_path> <stats_output_path>")
        sys.exit(1)

    input_data_path = sys.argv[1]
    stats_output_path = sys.argv[2]
    
    # 定义路径
    model_save_path = 'model_pro/final_model.pt'
    output_path = 'model_pro/predictions.csv'
    bert_model_path = 'model_pro/bert_model'
    ctm_tokenizer_path = 'model_pro/sentence_bert_model'

    # 执行预测
    success = predict(model_save_path, input_data_path, output_path, bert_model_path, ctm_tokenizer_path,
                     stats_output_path)

    if success:
        sys.exit(0)
    else:
        sys.exit(1)
