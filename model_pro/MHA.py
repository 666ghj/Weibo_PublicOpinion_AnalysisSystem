import torch
import torch.nn as nn

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


if __name__ == "__main__":
    embed_size = 512
    num_heads = 8
    mha_layer = MultiHeadAttentionLayer(embed_size, num_heads)
    print("Linear layers for Q, K, V initialized.")
