import torch 
import torch.nn as nn

from Attention.crossAttention import CrossAttention
from Attention.MultiHeadAttention import MultiHeadedAttention
from Components.residual_connect import ResidualConnect

class SingleDecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ffo_neurons = 1024):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        attention = MultiHeadedAttention(embed_dim, num_heads) 
        cross_attention = CrossAttention(embed_dim, num_heads) 
            
        self.ffo = nn.Sequential(
            nn.Linear(embed_dim,ffo_neurons),
            nn.ReLU(),
            nn.Linear(ffo_neurons,embed_dim)
        )

        self.residual_attention = ResidualConnect(
            attention
        )
        self.residual_cross_attention = ResidualConnect(
            cross_attention
        )
        self.residual_ffo = ResidualConnect(
            self.ffo
        )

        self.attention_norm = nn.LayerNorm(embed_dim)
        self.cross_attention_norm = nn.LayerNorm(embed_dim)
        self.ffo_norm = nn.LayerNorm(embed_dim)


    def forward(self, encoder_output : torch.Tensor, decoder_input : torch.Tensor) -> torch.Tensor:
        B,T_e,E = encoder_output.shape
        _,T_d,_ = decoder_input.shape

        self.residual_attention = self.residual_attention.to(decoder_input.device)
        self.residual_cross_attention = self.residual_cross_attention.to(decoder_input.device)
        self.ffo = self.ffo.to(decoder_input.device)
        

        attention = self.residual_attention(residual_input = decoder_input, input = decoder_input, mask = 'casual' )
        attention = self.attention_norm(attention)

        cross_attention = self.residual_cross_attention(residual_input = attention,decoder_input = attention, encoder_output = encoder_output)
        cross_attention = self.cross_attention_norm(cross_attention)

        ffo = self.residual_ffo(cross_attention, cross_attention)
        ffo = self.ffo_norm(ffo)

        return ffo 




        