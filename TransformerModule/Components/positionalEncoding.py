import torch 
import torch.nn as nn 
import math

def create_pos_tensor(start_T, end_T,embed_dim, device = 'cpu'):
    seq_len = end_T - start_T
    position = torch.arange(start=start_T, end=end_T, dtype=torch.float32).unsqueeze(1).to(device)
    div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)).to(device)
    embeddings = torch.zeros(seq_len, embed_dim).to(device)


    embeddings[:, 0::2] = torch.sin(position * div_term) # Even indices
    embeddings[:, 1::2] = torch.cos(position * div_term) # Odd indices

    return embeddings


class PositionalEncoder(nn.Module):
    def __init__(self, embed_dim, initial_max_len = 512):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_len = initial_max_len

        self.positional_embeds = create_pos_tensor(0, self.max_len, embed_dim)
        self.positional_embeds.requires_grad = False

    def forward(self, input, return_added = False):
        B,T_B,E = input.shape
        assert E == self.embed_dim
        self.positional_embeds = self.positional_embeds.to(input.device).detach()

        T_p,_ = self.positional_embeds.shape

        if T_B > T_p :
            with torch.no_grad():
                new_max_len = max(T_B, self.max_len*2)
                new_embeds = create_pos_tensor(self.max_len,new_max_len, self.embed_dim, device=input.device)
                new_embed_tensor = torch.cat([self.positional_embeds, new_embeds], dim=0)
                self.positional_embeds = new_embed_tensor
                self.max_len = new_max_len

        if not return_added :
            return input + self.positional_embeds[:T_B, :]

        return self.positional_embeds[:T_B, :]
    