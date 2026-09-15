import torch 
import torch.nn as nn 
from Attention.MultiHeadAttention import MultiHeadedAttention
from residual_connect import ResidualConnect

class SingleEncodeBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ffo_neurons = 1024):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ffo_neurons = ffo_neurons
        
        
        self.attention = MultiHeadedAttention(embed_dim, num_heads)
        self.residual_attention = ResidualConnect(
            self.attention
        )
        self.layer_norm_attention = nn.LayerNorm(embed_dim)
        
        
        self.ffo = nn.Sequential(
            nn.Linear(embed_dim, ffo_neurons),
            nn.ReLU(),
            nn.Linear(ffo_neurons, embed_dim)
        )
        self.residual_ffo = ResidualConnect(
            self.ffo
        )
        self.layer_norm_ffo = nn.LayerNorm(embed_dim)
        
        
    def forward(self, input):
        B,T,E = input.shape   
        H = self.num_heads
        assert E == self.embed_dim
        assert E%H == 0
        
        self.attention.to(input.device)
        self.residual_attention.to(input.device)
        self.layer_norm_attention.to(input.device)
        self.ffo.to(input.device)
        self.residual_ffo.to(input.device)
        self.layer_norm_ffo.to(input.device)
        
        attention = self.residual_attention(input)
        attention = self.layer_norm_attention(attention)
        
        ff_output = self.residual_ffo(attention) 
        ff_output = self.layer_norm_ffo(ff_output) 
        
        return ff_output