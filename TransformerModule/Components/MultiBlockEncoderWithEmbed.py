import torch 
import torch.nn as nn 
from encoderBlock import SingleEncodeBlock
from positionalEncoding import PositionalEncoder


class CompleteEncoderBlock(nn.Module):
    def __init__(self, n_blocks, num_heads, vocab_count, embed_dim, ffo_neurons=1024):
        super().__init__()
        self.n_blocks = n_blocks
        self.num_heads = num_heads
        self.vocab_count = vocab_count
        self.embed_dim = embed_dim
        
        self.emebedding_layer = nn.Embedding(vocab_count, embed_dim)
        self.positional_layer = PositionalEncoder(embed_dim)
        
        self.encoders = nn.ModuleList(
            [SingleEncodeBlock(embed_dim, num_heads, 1024) for _ in range(n_blocks)]
        ) 
        
    def forward(self, input):
        B,T_e = input.shape
        
        input = self.emebedding_layer(input) # B,T_e,E
        input = self.positional_layer.forward(input, return_added= True) # B,T_e,E
        
        encoder_input = input    
        for single_encoder in self.encoders :
            encoder_input = single_encoder(encoder_input)

        final_output =  encoder_input
        return final_output
            
            