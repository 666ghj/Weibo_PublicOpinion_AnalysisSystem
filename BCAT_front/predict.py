import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os
import sys
import json
import chardet  # 导入 chardet

# 导入您定义的模型和模块
from MHA import MultiHeadAttentionLayer
from classifier import FinalClassifier
from BERT_CTM import BERT_CTM_Model

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    print(f"Detected encoding: {encoding} with confidence {confidence}")
    return encoding


def get_bert_ctm_embeddings(texts, bert_model_path, ctm_tokenizer_path, n_components=12, num_epochs=20):
    # 创建BERT_CTM_Model实例
    bert_ctm_model = BERT_CTM_Model(
        bert_model_path=bert_model_path,
        ctm_tokenizer_path=ctm_tokenizer_path,
        n_components=n_components,
        num_epochs=num_epochs
    )
    # 加载已保存的CTM模型
    bert_ctm_model.load_model()
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
        # 修改这里，设置 weights_only=True 以消除 FutureWarning
        checkpoint = torch.load(model_save_path, map_location=device, weights_only=False)
        classifier_model = FinalClassifier(input_dim=768, num_classes=num_classes)
        classifier_model.load_state_dict(checkpoint['classifier_model_state_dict'])
        classifier_model.to(device)
        classifier_model.eval()

        attention_model = MultiHeadAttentionLayer(embed_size=768, num_heads=8)
        attention_model.load_state_dict(checkpoint['attention_model_state_dict'])
        attention_model.to(device)
        attention_model.eval()

        # 检测文件编码
        encoding = detect_file_encoding(input_data_path)

        # 读取输入数据
        data = pd.read_csv(input_data_path, encoding=encoding)
        texts = data['TEXT'].tolist()

        # 生成嵌入
        print("Generating embeddings...")
        embeddings = get_bert_ctm_embeddings(texts, bert_model_path, ctm_tokenizer_path)

        # 准备DataLoader
        data_loader = prepare_dataloader(embeddings, batch_size)

        # 存储预测结果
        all_predictions = []

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Predicting"):
                batch_x = batch[0].to(device)
                batch_x = torch.mean(batch_x, dim=1)
                attention_output = attention_model(batch_x, batch_x, batch_x)
                outputs = classifier_model(attention_output)
                outputs = torch.mean(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                all_predictions.extend(predicted.cpu().numpy())

        # 保存预测结果
        data['Predicted_Label'] = all_predictions
        data.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Predictions saved to {output_path}")

        # 统计标签的个数和占比
        label_counts = data['Predicted_Label'].value_counts()
        total_count = len(data)
        stats = {}
        for label, count in label_counts.items():
            label_name = "良好" if label == 0 else "不良"
            percentage = (count / total_count) * 100
            stats[label_name] = {
                'count': count,
                'percentage': f"{percentage:.2f}%"
            }
            print(f"Label: {label_name}, Count: {count}, Percentage: {percentage:.2f}%")

        # 将统计信息保存到 JSON 文件
        with open(stats_output_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False)

        return True  # 成功执行
    except Exception as e:
        print(f"Error during prediction: {e}")
        return False  # 执行失败


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python using_example.py <input_data_path> <stats_output_path>")
        sys.exit(1)

    input_data_path = sys.argv[1]
    stats_output_path = sys.argv[2]
    # 定义路径
    model_save_path = 'BCAT/final_model.pt'
    output_path = 'BCAT/predictions.csv'  # 保存预测结果的文件
    bert_model_path = 'BCAT/bert_model'
    ctm_tokenizer_path = 'BCAT/sentence_bert_model'

    # 执行预测
    success = predict(model_save_path, input_data_path, output_path, bert_model_path, ctm_tokenizer_path,
                      stats_output_path)

    if success:
        sys.exit(0)  # 成功
    else:
        sys.exit(1)  # 失败
