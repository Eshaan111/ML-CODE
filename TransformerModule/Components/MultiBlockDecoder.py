import torch 
import torch.nn as nn 
from .decoderBlock import SingleDecoderBlock
from .positionalEncoding import PositionalEncoder


class CompleteDecoderBlock(nn.Module):
    def __init__(self, n_blocks, num_heads, vocab_count, embed_dim, ffo_neurons=1024, is_cross_attention = True):
        super().__init__()
        self.n_blocks = n_blocks
        self.num_heads = num_heads
        self.vocab_count = vocab_count
        self.embed_dim = embed_dim
        self.is_cross_attention = is_cross_attention
        self.embedding_layer = nn.Embedding(vocab_count, embed_dim)
        self.positional_layer = PositionalEncoder(embed_dim)
        
        self.decoders = nn.ModuleList(
            [SingleDecoderBlock(embed_dim, num_heads, ffo_neurons, is_cross_attention ) for _ in range(n_blocks)]
        ) 
        self.output_head = nn.Linear(embed_dim, vocab_count)
        
    def forward(self, decoder_input, encoder_output = None, return_post_softmax = False):
        B_d,T_d = decoder_input.shape
        if self.is_cross_attention : 
            assert encoder_output is not None, \
                "encoder_output is required when cross-attention is enabled"
            B_e,T_e,E_e = encoder_output.shape
            assert (B_e, E_e) == (B_d, self.embed_dim)
        
        input = self.embedding_layer(decoder_input) # B,T_d,E
        input = self.positional_layer(input, return_added= True) # B,T_d,E
        # print(input.shape)
        decoder_input = input    
        for single_decoder in self.decoders :
            if self.is_cross_attention:
                decoder_input = single_decoder(decoder_input = decoder_input, encoder_output = encoder_output)
            else:
                decoder_input = single_decoder(decoder_input = decoder_input)
                
        output = self.output_head(decoder_input) # B,T_d,V
        if return_post_softmax :
            return torch.softmax(output, dim = -1)
        
        return output
            
            