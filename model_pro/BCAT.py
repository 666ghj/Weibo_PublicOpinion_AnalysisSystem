import os
import numpy as np
from BERT_CTM import BERT_CTM_Model  # 假设BERT_CTM模型在这个文件中

# BERT_CTM 嵌入生成和加载函数
def get_bert_ctm_embeddings(texts, bert_model_path, ctm_tokenizer_path, n_components=12, num_epochs=20, save_path=None):
    """
    获取或生成 BERT+CTM 嵌入，并保存到文件。
    
    :param texts: 需要嵌入的文本
    :param bert_model_path: BERT 模型的路径
    :param ctm_tokenizer_path: CTM tokenizer 的路径
    :param n_components: 生成的主题数量
    :param num_epochs: 训练的epoch数
    :param save_path: 嵌入保存路径
    :return: 生成或加载的嵌入
    """
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


if __name__ == "__main__":
    # 示例调用
    sample_texts = ["This is a test text.", "Another example of text data."]
    bert_model_path = './bert_model'
    ctm_tokenizer_path = './sentence_bert_model'
    save_path = 'sample_embeddings.npy'

    # 生成或加载 BERT+CTM 嵌入
    embeddings = get_bert_ctm_embeddings(sample_texts, bert_model_path, ctm_tokenizer_path, save_path=save_path)

    # 打印嵌入形状
    print(f"嵌入形状: {embeddings.shape}")
