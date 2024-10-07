import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadAttentionLayer, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        
        assert (self.head_dim * num_heads == embed_size), "Embedding size needs to be divisible by num_heads"
        
        # Define linear layers for Q, K, V
        self.q_linear = nn.Linear(embed_size, embed_size)
        self.k_linear = nn.Linear(embed_size, embed_size)
        self.v_linear = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query):
        N = query.shape[0]  # batch_size
        
        # Linear transformations for Q, K, V
        Q = self.q_linear(query)
        K = self.k_linear(keys)
        V = self.v_linear(values)
        
        # Reshape into multiple heads
        Q = Q.reshape(N, -1, self.num_heads, self.head_dim)
        K = K.reshape(N, -1, self.num_heads, self.head_dim)
        V = V.reshape(N, -1, self.num_heads, self.head_dim)
        
        # Compute scaled dot-product attention scores
        attention_scores = torch.einsum("nqhd,nkhd->nhqk", [Q, K])
        attention_scores = attention_scores / (self.head_dim ** 0.5)
        attention = torch.softmax(attention_scores, dim=-1)  # Normalize
        
        return attention


if __name__ == "__main__":
    embed_size = 512
    num_heads = 8
    mha_layer = MultiHeadAttentionLayer(embed_size, num_heads)
    
    values = torch.randn(2, 10, embed_size)
    keys = torch.randn(2, 10, embed_size)
    query = torch.randn(2, 10, embed_size)
    
    attention = mha_layer(values, keys, query)
    print(f"Attention shape: {attention.shape}")
