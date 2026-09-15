import torch 
import torch.nn as nn
from Components.decoderBlock import SingleDecoderBlock
from Components.residual_connect import ResidualConnect
import pytest


@pytest.fixture(
    params=[
        (1, 2, 3, 8, 1),
        (1, 5, 4, 32, 4),
        (4, 10, 10, 64, 8),
        (8, 20, 14, 128, 8),
    ]
)
def decoder_config(request):
    B,T_e,T_d,E,H = request.param
    return B,T_e,T_d,E,H


def test_input_output_shape(decoder_config):
    B,T_e,T_d,E,H = decoder_config

    output_e_tensor = torch.randn(B, T_e, E)
    input_d_tensor = torch.randn(B, T_d, E)

    decoderBlock = SingleDecoderBlock(
        embed_dim=E,
        num_heads=H        
    )

    output = decoderBlock(output_e_tensor, input_d_tensor)

    assert output.shape == (B,T_d,E)


def test_residual_connects(decoder_config):
    B,T_e,T_d,E,H = decoder_config
    input_d_tensor = torch.randn(B, T_d, E)
    sublayer = nn.Sequential(
        nn.Linear(E,1024),
        nn.ReLU(),
        nn.Linear(1024,E)
    )

    residual_connect = ResidualConnect(sublayer)
    output = residual_connect(input_d_tensor, input_d_tensor)

    assert torch.allclose(
        output,
        input_d_tensor + sublayer(input_d_tensor)
    )