import torch
import torch.nn as nn 
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch.optim import Adam
import pandas as pd 
import numpy as np 

import re

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# nltk.download("punkt")
# nltk.download("punkt_tab")
# nltk.download("stopwords")

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(
    log_dir="runs/Attention"
)

from math import sqrt
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import os

print(os.getcwd(), device)



class MultiHeadedAttention(nn.Module):

    def __init__(self,num_heads,input_embedding_dim, device, masked=False, ):
        super().__init__()

        if(input_embedding_dim%num_heads != 0):
            raise ValueError(f"CANT DIVIDE Embedding-dim{input_embedding_dim} into Number-Heads{num_heads}")

        embedding_dim = input_embedding_dim//num_heads # Eh
        self.isMasked = masked
        self.device = device
        self.num_heads = num_heads
        self.expected_dim = input_embedding_dim 
        self.tensor_all_query = nn.Parameter(torch.rand((num_heads, input_embedding_dim, embedding_dim), device=self.device))#H,E,Eh
        self.tensor_all_key = nn.Parameter(torch.rand((num_heads, input_embedding_dim, embedding_dim), device=self.device))#H,E,Eh
        self.tensor_all_value = nn.Parameter(torch.rand((num_heads, input_embedding_dim, embedding_dim), device=self.device))#H,E,Eh
        self.output_weights = nn.Parameter(torch.rand((input_embedding_dim,input_embedding_dim), device=self.device)) #E,E



    def forward(self, batch):
        batch_count, timestep_count, original_embedding_dim = batch.shape ## BxTxE
        num_heads = self.num_heads
        if(self.expected_dim  != original_embedding_dim):
            raise ValueError(f"EXPECTED embed dim={self.expected_dim} but given {original_embedding_dim}")

        embedding_dim = original_embedding_dim//num_heads #Eh
        head_outputs = torch.zeros(size=(num_heads,batch_count, timestep_count, embedding_dim), device=self.device) #H,B,T,Eh

        for i in range(num_heads):
            query_tensor = self.tensor_all_query[i,:,:] #ExEh
            key_tensor = self.tensor_all_key[i,:,:] #ExEh
            value_tensor = self.tensor_all_value[i,:,:] #ExEh

            query_embeds = batch @ query_tensor 
            key_embeds = batch @ key_tensor 
            value_embeds = batch @ value_tensor

            temp = query_embeds @ key_embeds.permute(0,2,1) # BxTxEh . BxEhxT = BxTqueyxTkey
            scaled = temp/sqrt(embedding_dim)

            if self.isMasked:
                idx_T_row = torch.arange(timestep_count, device=device).view(timestep_count,1)
                idx_T_col = torch.arange(timestep_count, device=device).view(1,timestep_count)
                mask_0 = idx_T_col <= idx_T_row
                mask_inf = idx_T_col > idx_T_row
                scaled[:,mask_0] += 0
                scaled[:,mask_inf] = float("-inf")
                
            head_output_embeds = torch.softmax(scaled, dim=2 ) @ value_embeds

            
            head_outputs[i,:,:,:] = head_output_embeds

        head_outputs = head_outputs.permute(1, 2, 0, 3) #new shape = (B,T,Head_count,Eh)
        head_outputs = head_outputs.flatten(start_dim=2)

        output_embeds = head_outputs @ self.output_weights # B,T,E . E,E

        return output_embeds # B,T,E


