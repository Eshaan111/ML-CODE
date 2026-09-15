import torch 
import torch.nn as nn 
from Components.decoderBlockCrossAttention import SingleDecoderBlock
from positionalEncoding import PositionalEncoder


class CompleteDecoderBlock(nn.Module):
    def __init__(self, n_blocks, num_heads, vocab_count, embed_dim, ffo_neurons=1024):
        super().__init__()
        self.n_blocks = n_blocks
        self.num_heads = num_heads
        self.vocab_count = vocab_count
        self.embed_dim = embed_dim
        
        self.emebedding_layer = nn.Embedding(vocab_count, embed_dim)
        self.positional_layer = PositionalEncoder(embed_dim)
        
        self.decoders = nn.ModuleList(
            [SingleDecoderBlock(embed_dim, num_heads, 1024) for _ in range(n_blocks)]
        ) 
        self.ffo = nn.Linear(embed_dim, ffo_neurons)
        
    def forward(self, decoder_input,encoder_output, return_post_softmax = False):
        B_d,T_d = decoder_input.shape
        B_e,T_e,E_e = encoder_output.shape
        assert (B_e, E_e) == (B_d, self.embed_dim)
        
        input = self.emebedding_layer(decoder_input) # B,T_d,E
        input = self.positional_layer.forward(input, return_added= True) # B,T_d,E
        
        decoder_input = input    
        for single_decoder in self.decoders :
            decoder_input = single_decoder(encoder_output = encoder_output , decoder_input = decoder_input)
            
        ffo = self.ffo(decoder_input) # B,T_d,V
        if return_post_softmax :
            return torch.softmax(ffo, dim = -1)
        
        return ffo
            
            