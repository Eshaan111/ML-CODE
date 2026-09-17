import torch
import torch.nn as nn
import math

class MultiHeadedAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)

        self.wo = nn.Linear(embed_dim,embed_dim)
        

    def forward(self, input, mask = None):
        B,T,E = input.shape

        assert E % self.num_heads == 0
        assert E == self.embed_dim

        H = self.num_heads
        Eh = self.embed_dim//H 

        q = self.q(input)  # B,T,E
        k = self.k(input) # B,T,E
        v = self.v(input) # B,T,E

        q = q.reshape(B,T,H,Eh).transpose(1,2) # B,H,T,Eh
        k = k.reshape(B,T,H,Eh).transpose(1,2) # B,H,T,Eh
        v = v.reshape(B,T,H,Eh).transpose(1,2) # B,H,T,Eh

        sim_scores = q @ k.transpose(-1,-2) # (B,H,T,Eh) x (B,H,Eh,T) = (B,H,T_q,T_k)
        sim_scores = sim_scores / math.sqrt(Eh)

        if mask == 'causal':
            t_q = torch.arange(0,T).view(T,1).to(input.device)
            t_k = torch.arange(0,T).view(1,T).to(input.device)
            # mask_0 = t_q >= t_k
            mask_inf = t_q < t_k
            sim_scores = sim_scores.masked_fill(
                mask_inf,
                float('-inf')
            )

        attention_weights = torch.softmax(sim_scores, dim = -1)

        attention = attention_weights @ v # (B,H,T,T) x (B,H,T,Eh) = (B,H,T,Eh)
        attention = attention.transpose(-2,-3).reshape(B,T,E) # (B,T,E)

        return self.wo(attention)
