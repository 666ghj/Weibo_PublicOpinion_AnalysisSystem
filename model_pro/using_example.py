import torch
import numpy as np
from transformers.models.bert import BertTokenizer, BertModel
from MHA import MultiHeadAttentionLayer
from classifier import FinalClassifier


# 加载BERT模型并生成嵌入
def get_sentence_embeddings(sentences, bert_model_path, max_length=80):
    """使用BERT生成多个句子的嵌入"""
    tokenizer = BertTokenizer.from_pretrained(bert_model_path)
    model = BertModel.from_pretrained(bert_model_path)

    embeddings = []
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length)
        with torch.no_grad():
            outputs = model(**inputs)
        embedding = outputs.last_hidden_state.cpu().numpy()
        embeddings.append(embedding)

    return np.vstack(embeddings)  # 保持多句子输出格式一致


# 加载已经训练好的模型
def load_model(model_path):
    print(f"加载模型 {model_path}...")
    model = torch.load(model_path)
    model.eval()  # 设置为评估模式
    return model


# 多句子的预测函数
def predict_sentences(sentences, model, bert_model_path, max_length=80):
    # 检查是否为单个句子输入，如果是，将其包装为列表
    if isinstance(sentences, str):
        sentences = [sentences]

    # 生成句子的BERT嵌入
    embeddings = get_sentence_embeddings(sentences, bert_model_path, max_length)

    # 转换为Tensor
    embedding_tensors = torch.tensor(embeddings, dtype=torch.float32).squeeze(1)  # 修改squeeze以适应多个句子

    # 检查嵌入维度是否符合注意力层要求
    embed_size = embedding_tensors.size(-1)
    num_heads = 12
    if embed_size % num_heads != 0:
        raise ValueError(f"嵌入维度 {embed_size} 无法被注意力头数量 {num_heads} 整除")

    # 加载多头注意力机制
    attention_model = MultiHeadAttentionLayer(embed_size=embed_size, num_heads=num_heads)

    predictions = []
    with torch.no_grad():
        for embedding_tensor in embedding_tensors:
            attention_output = attention_model(embedding_tensor.unsqueeze(0), embedding_tensor.unsqueeze(0),
                                               embedding_tensor.unsqueeze(0))
            outputs = model(attention_output)
            outputs = torch.mean(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)  # 获取预测的类别
            predictions.append(predicted.item())

    return predictions


if __name__ == "__main__":
    # 加载已经训练好的模型
    model_path = './final_model.pt'
    model = load_model(model_path)

    # 需要预测的句子，可以输入单个句子或多个句子
    sentences = ["这是一条待预测的句子",
                 "他在你面前骂黑鬼 印度屎屁尿背后就会根人家骂你中国猴子，这可能不是种族歧视这是素质太低",
                 "完美女朋友",
                 "在美国的亚裔就是一盘散沙。日裔看不起韩裔 韩裔仇视日裔 港澳台裔看不起大陆裔，大陆裔里面又歧视福建裔"]  # 可以替换为单个句子或多个句子

    # BERT模型路径
    bert_model_path = './bert_model'

    # 对句子进行预测
    predicted_labels = predict_sentences(sentences, model, bert_model_path)

    # 根据预测的label输出对应的文本
    for i, label in enumerate(predicted_labels):
        if label == 1:
            print(f"句子: '{sentences[i]}' 预测结果: 不良言论")
        elif label == 0:
            print(f"句子: '{sentences[i]}' 预测结果: 正常言论")
        else:
            print(f"句子: '{sentences[i]}' 未知标签: {label}")
