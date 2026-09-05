import torch
from math import sqrt
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import os

print(os.getcwd(), device)

from math import sin, cos, pow

def PositionalEncoder(input_embedding, device):
    batch_count, timesteps, embedding_dim = input_embedding.shape
    output_tensor = torch.zeros((batch_count, timesteps, embedding_dim), device=device)

    for i in range(0,timesteps):
        for j in range(0,embedding_dim,2):
            output_tensor[:,i,j] = sin(i/(pow(10000,j/embedding_dim)))
            output_tensor[:,i,j+1] = cos(i/(pow(10000,j/embedding_dim)))
    return output_tensor



