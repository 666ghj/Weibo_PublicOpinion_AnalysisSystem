import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, embed_size, num_heads, dropout_rate=0.1):
        super(MultiHeadAttentionLayer, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        
        assert (self.head_dim * num_heads == embed_size), "Embedding size needs to be divisible by num_heads"
        
        # 定义线性变换层，分别用于 Q, K, V
        self.q_linear = nn.Linear(embed_size, embed_size)
        self.k_linear = nn.Linear(embed_size, embed_size)
        self.v_linear = nn.Linear(embed_size, embed_size)
        
        # 最终的线性层
        self.fc_out = nn.Linear(embed_size, embed_size)
        
        # 增加 Dropout 和 LayerNorm
        self.dropout = nn.Dropout(p=dropout_rate)
        self.layer_norm = nn.LayerNorm(embed_size)

    def forward(self, values, keys, query, mask=None):
        N = query.shape[0]  # batch_size
        
        # 将输入变换为 Q, K, V
        Q = self.q_linear(query)  # shape: (N, seq_len, embed_size)
        K = self.k_linear(keys)   # shape: (N, seq_len, embed_size)
        V = self.v_linear(values) # shape: (N, seq_len, embed_size)
        
        # 将 Q, K, V 分成多个头
        Q = Q.reshape(N, -1, self.num_heads, self.head_dim)  # shape: (N, seq_len, num_heads, head_dim)
        K = K.reshape(N, -1, self.num_heads, self.head_dim)  # shape: (N, seq_len, num_heads, head_dim)
        V = V.reshape(N, -1, self.num_heads, self.head_dim)  # shape: (N, seq_len, num_heads, head_dim)
        
        # 计算缩放点积注意力
        attention_scores = torch.einsum("nqhd,nkhd->nhqk", [Q, K])  # (N, num_heads, seq_len_q, seq_len_k)
        attention_scores = attention_scores / (self.head_dim ** (1 / 2))  # 缩放
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float("-1e20"))
        
        attention = torch.softmax(attention_scores, dim=-1)  # 归一化
        
        # 根据注意力分布加权 V
        out = torch.einsum("nhql,nlhd->nqhd", [attention, V])  # (N, num_heads, seq_len_q, head_dim)
        out = out.reshape(N, -1, self.embed_size)  # 将多头输出拼接回原始嵌入大小
        
        # 通过线性层
        out = self.fc_out(out)
        
        # 使用残差连接并应用 LayerNorm
        out = self.layer_norm(out + query)
        
        # 应用 Dropout
        out = self.dropout(out)
        
        return out

