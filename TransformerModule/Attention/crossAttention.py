import torch
import torch.nn as nn
from math import sqrt


class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)

        self.wo = nn.Linear(embed_dim,embed_dim)


    def forward(self, decoder_input, encoder_output):
        B,T_d,E = decoder_input.shape
        _,T_e,_ = encoder_output.shape
        assert E % self.num_heads == 0
        assert E == self.embed_dim

        self.q = self.q.to(decoder_input.device)
        self.k = self.k.to(encoder_output.device)
        self.v = self.v.to(encoder_output.device)
        self.wo = self.wo.to(encoder_output.device)

        H = self.num_heads
        Eh = self.embed_dim//H 

        q = self.q(decoder_input)  # B,T_d,E
        k = self.k(encoder_output) # B,T_e,E
        v = self.v(encoder_output) # B,T_e,E

        q = q.reshape(B,T_d,H,Eh).transpose(1,2) # B,H,T_d,Eh
        k = k.reshape(B,T_e,H,Eh).transpose(1,2) # B,H,T_e,Eh
        v = v.reshape(B,T_e,H,Eh).transpose(1,2) # B,H,T_e,Eh

        sim_scores = q @ k.transpose(-1,-2) # (B,H,T_d,Eh) x (B,H,Eh,T_e) = (B,H,T_d,T_e)
        sim_scores = sim_scores / sqrt(Eh)
        attention_weights = torch.softmax(sim_scores, dim = -1)

        attention = attention_weights @ v # (B,H,T_d,T_e) x (B,H,T_e,Eh) = (B,H,T_d,Eh)
        attention = attention.transpose(-2,-3).reshape(B,T_d,E) # (B,T_d,E)

        return self.wo(attention)