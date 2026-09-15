import torch
import pytest 
from Attention.crossAttention import CrossAttention


@pytest.fixture(
    params=[
        (1, 2, 3, 8, 1),
        (1, 5, 4, 32, 4),
        (4, 10, 10, 64, 8),
        (8, 20, 14, 128, 8),
    ]
)
def attention_config(request):
    B, T_e, T_d, E, H = request.param
    return B, T_e, T_d, E, H


def test_cross_attention_io_shape(attention_config):
    B, T_e, T_d, E, H = attention_config

    attention = CrossAttention(
        embed_dim=E,
        num_heads=H
    )

    output_e_tensor = torch.randn(B, T_e, E)
    input_d_tensor = torch.randn(B, T_d, E)

    cross_attention = attention(input_d_tensor, output_e_tensor)

    assert cross_attention.shape == (B, T_d, E)


def test_cross_attention_backward():
    B, T_e, T_d, E, H = 3,5,4,32,4

    attention = CrossAttention(
        embed_dim=E,
        num_heads=H
    )

    output_e_tensor = torch.randn(B, T_e, E , requires_grad=True)
    input_d_tensor = torch.randn(B, T_d, E, requires_grad=True)

    cross_attention = attention(input_d_tensor, output_e_tensor)

    loss = cross_attention.mean()
    loss.backward()

    assert input_d_tensor.grad is not None
    assert output_e_tensor.grad is not None

    for name, parameter in attention.named_parameters():

        assert parameter.grad is not None, \
            f"{name} has no gradient"



def test_cross_attention_encoder_significance(attention_config):
    B, T_e, T_d, E, H = attention_config

    attention = CrossAttention(
        embed_dim=E,
        num_heads=H
    )

    output_e_tensor1 = torch.randn(B, T_e, E)
    output_e_tensor2 = torch.randn(B, T_e, E)
    input_d_tensor = torch.randn(B, T_d, E)

    cross_attention1 = attention(input_d_tensor, output_e_tensor1)
    cross_attention2 = attention(input_d_tensor, output_e_tensor2)

    assert not torch.allclose(
        cross_attention1,
        cross_attention2
    )


def test_cross_attention_decoder_significance(attention_config):
    B, T_e, T_d, E, H = attention_config

    attention = CrossAttention(
        embed_dim=E,
        num_heads=H
    )

    output_e_tensor = torch.randn(B, T_e, E)
    input_d_tensor1 = torch.randn(B, T_d, E)
    input_d_tensor2 = torch.randn(B, T_d, E)

    cross_attention1 = attention(input_d_tensor1, output_e_tensor)
    cross_attention2 = attention(input_d_tensor2, output_e_tensor)

    assert not torch.allclose(
        cross_attention1,
        cross_attention2
    )
