import torch
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
from CNN import extract_CNN_features
from MHA import MultiHeadAttentionLayer
from classifier import FinalClassifier
from BERT_CTM import BERT_CTM_Model
import os
from tqdm import tqdm  # 导入 tqdm 库用于进度条
from sklearn.metrics import confusion_matrix


# BERT_CTM 嵌入生成和加载函数
def get_bert_ctm_embeddings(texts, bert_model_path, ctm_tokenizer_path, n_components=12, num_epochs=20, save_path=None):
    # 检查是否已经存在保存的嵌入文件
    if save_path and os.path.exists(save_path):
        print(f"从文件 {save_path} 加载嵌入...")
        embeddings = np.load(save_path)
    else:
        print("生成 BERT+CTM 嵌入...")
        bert_ctm_model = BERT_CTM_Model(
            bert_model_path=bert_model_path,
            ctm_tokenizer_path=ctm_tokenizer_path,
            n_components=n_components,
            num_epochs=num_epochs
        )
        embeddings = bert_ctm_model.train(texts)  # 生成嵌入

        # 保存嵌入到文件
        if save_path:
            print(f"保存嵌入到文件 {save_path}...")
            np.save(save_path, embeddings)

    return embeddings


# 数据加载和准备函数
def prepare_dataloader(features, labels, batch_size):
    """创建 DataLoader 用于训练、验证和测试"""
    tensor_x = torch.tensor(features, dtype=torch.float32)
    tensor_y = torch.tensor(labels, dtype=torch.long)
    dataset = TensorDataset(tensor_x, tensor_y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# 训练模型函数
def train_model(train_data_path, valid_data_path, test_data_path, train_labels, valid_labels, test_labels,
                bert_model_path, ctm_tokenizer_path, num_heads=8, num_classes=2, epochs=10, batch_size=128,
                learning_rate=5e-3, model_save_path='./final_model.pt'):
    # Step 1: 获取 BERT+CTM 嵌入
    print("Step 1: 获取 BERT+CTM 嵌入...")
    valid_features = get_bert_ctm_embeddings(valid_data_path, bert_model_path, ctm_tokenizer_path,
                                             save_path='valid_embeddings.npy')
    test_features = get_bert_ctm_embeddings(test_data_path, bert_model_path, ctm_tokenizer_path,
                                            save_path='test_embeddings.npy')
    train_features = get_bert_ctm_embeddings(train_data_path, bert_model_path, ctm_tokenizer_path,
                                             save_path='train_embeddings.npy')

    # 保存标签到 .npy 文件
    print("保存标签到 labels.npy 文件...")
    np.save('train_labels.npy', train_labels)
    np.save('valid_labels.npy', valid_labels)
    np.save('test_labels.npy', test_labels)

    # Step 2: 检查标签的合理性
    print("Step 2: 检查标签的合理性...")
    unique_labels_train = np.unique(train_labels)
    unique_labels_valid = np.unique(valid_labels)
    unique_labels_test = np.unique(test_labels)
    print(f"训练标签的唯一值: {unique_labels_train}")
    print(f"训练集类别分布: {np.bincount(train_labels)}")
    print(f"验证标签的唯一值: {unique_labels_valid}")
    print(f"验证集类别分布: {np.bincount(valid_labels)}")
    print(f"测试标签的唯一值: {unique_labels_test}")
    print(f"测试集类别分布: {np.bincount(test_labels)}")

    if len(unique_labels_train) != num_classes or len(unique_labels_valid) != num_classes or len(
            unique_labels_test) != num_classes:
        raise ValueError(f"标签中的类别数量与期望的不符: 期望 {num_classes}, 但训练集、验证集或测试集中发现了其他类别")

    # Step 3: 创建 DataLoader
    print("Step 3: 创建 DataLoader...")
    train_loader = prepare_dataloader(train_features, train_labels, batch_size)
    valid_loader = prepare_dataloader(valid_features, valid_labels, batch_size)
    test_loader = prepare_dataloader(test_features, test_labels, batch_size)

    # Step 4: 初始化CNN
    print("Step 4: 初始化CNN...")
    num_filters = 256  # 使用256个卷积输出通道
    kernel_sizes = [2, 3, 4]  # 卷积核大小
    k = 3 * len(kernel_sizes)
    cnn_output_dim = num_filters * (k + 1)  # 计算CNN输出的特征维度

    # Step 5: 初始化注意力机制
    print("Step 5: 初始化多头注意力机制...")
    attention_model = MultiHeadAttentionLayer(embed_size=768, num_heads=8)

    # Step 6: 初始化分类器
    print("Step 6: 初始化分类器...")
    classifier_model = FinalClassifier(input_dim=768, num_classes=num_classes)
    optimizer = torch.optim.Adam(classifier_model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    # Step 7: 开始训练
    print("开始训练...")
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(epochs):
        classifier_model.train()
        epoch_loss = 0
        y_true = []
        y_pred = []

        # 使用 tqdm 为 CNN 特征提取添加进度条
        for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} - Training"):
            optimizer.zero_grad()
            batch_x = torch.mean(batch_x, dim=1)
            # 从CNN提取特征
            # cnn_output = extract_CNN_features(batch_x)
            # batch_x = torch.mean(batch_x, dim=1)
            # cnn_output = torch.cat((batch_x,cnn_output), dim=-1)
            attention_output = attention_model(batch_x, batch_x, batch_x)
            outputs = classifier_model(attention_output)
            outputs = torch.mean(outputs, dim=1)
            loss = criterion(outputs, batch_y)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 优化

            epoch_loss += loss.item()

            _, predicted = torch.max(outputs, 1)  # 获取预测类别
            y_true.extend(batch_y.tolist())
            y_pred.extend(predicted.tolist())

        # 计算训练准确率、精确率、召回率和F1分数
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')

        print(
            f"Epoch [{epoch + 1}/{epochs}] Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        print(confusion_matrix(y_true, y_pred))

    # 保存模型
    torch.save(classifier_model, model_save_path)
    print(f"训练好的模型已经保存到 {model_save_path}")

    # 验证集评估
    classifier_model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch_x, batch_y in valid_loader:
            batch_x = torch.mean(batch_x, dim=1)
            # cnn_output = extract_CNN_features(batch_x)
            # batch_x = torch.mean(batch_x, dim=1)
            # cnn_output = torch.cat((batch_x,cnn_output), dim=-1)
            attention_output = attention_model(batch_x, batch_x, batch_x)
            outputs = classifier_model(attention_output)
            outputs = torch.mean(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(batch_y.tolist())
            y_pred.extend(predicted.tolist())

    # 验证集准确率、精确率、召回率和F1分数
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    print(f"\nValidation - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    print(confusion_matrix(y_true, y_pred))

    # 测试集评估
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = torch.mean(batch_x, dim=1)
            # cnn_output = extract_CNN_features(batch_x)
            # batch_x = torch.mean(batch_x, dim=1)
            # cnn_output = torch.cat((batch_x,cnn_output), dim=-1)
            attention_output = attention_model(batch_x, batch_x, batch_x)
            outputs = classifier_model(attention_output)
            outputs = torch.mean(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(batch_y.tolist())
            y_pred.extend(predicted.tolist())
    # 测试集准确率、精确率、召回率和F1分数
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    print(f"\nTest - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    # 加载和准备数据
    train_data_path = './train.csv'
    valid_data_path = './dev.csv'
    test_data_path = './test.csv'

    train_data = pd.read_csv(train_data_path)
    valid_data = pd.read_csv(valid_data_path)
    test_data = pd.read_csv(test_data_path)

    train_labels = train_data['label'].values
    valid_labels = valid_data['label'].values
    test_labels = test_data['label'].values

    # 训练模型
    bert_model_path = './bert_model'
    ctm_tokenizer_path = './sentence_bert_model'

    # 训练模型
    train_model(train_data_path, valid_data_path, test_data_path, train_labels, valid_labels, test_labels,
                bert_model_path, ctm_tokenizer_path, num_heads=12, num_classes=2, model_save_path='./final_model.pt')
