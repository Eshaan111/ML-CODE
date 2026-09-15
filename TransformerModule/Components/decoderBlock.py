import torch 
import torch.nn as nn

from Attention.crossAttention import CrossAttention
from Attention.MultiHeadAttention import MultiHeadedAttention
from Components.residual_connect import ResidualConnect

class SingleDecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ffo_neurons = 1024, is_cross_attention = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.is_cross_attention = is_cross_attention
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
        if is_cross_attention :
            self.residual_cross_attention = ResidualConnect(
                cross_attention
            )
        self.residual_ffo = ResidualConnect(
            self.ffo
        )

        self.attention_norm = nn.LayerNorm(embed_dim)
        if is_cross_attention : self.cross_attention_norm = nn.LayerNorm(embed_dim)
        self.ffo_norm = nn.LayerNorm(embed_dim)


    def forward(self, decoder_input : torch.Tensor, encoder_output : torch.Tensor = None) -> torch.Tensor:
        if self.is_cross_attention : B,T_e,E = encoder_output.shape
        _,T_d,_ = decoder_input.shape

        attention = self.residual_attention(residual_input = decoder_input, input = decoder_input, mask = 'casual' )
        attention = self.attention_norm(attention)
        if self.is_cross_attention :
            cross_attention = self.residual_cross_attention(residual_input = attention,decoder_input = attention, encoder_output = encoder_output)
            cross_attention = self.cross_attention_norm(cross_attention)
            attention = cross_attention
        
        ffo = self.residual_ffo(attention, attention)
        ffo = self.ffo_norm(ffo)

        return ffo 




        