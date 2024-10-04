import torch
import torch.nn as nn

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadAttentionLayer, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        
        assert (self.head_dim * num_heads == embed_size), "Embedding size needs to be divisible by num_heads"


if __name__ == "__main__":
    embed_size = 512
    num_heads = 8
    mha_layer = MultiHeadAttentionLayer(embed_size, num_heads)
    print("Model initialized successfully.")
