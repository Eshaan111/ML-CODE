import torch 
import torch.nn as nn 


class ResidualConnect(nn.Module):
    def __init__(self, sublayer):
        super().__init__()
        self.sublayer = sublayer

    def forward(self, residual_input, *args, **kwargs):
        return residual_input + self.sublayer(*args, **kwargs) 