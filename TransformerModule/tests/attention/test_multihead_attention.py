import torch
from Attention.MultiHeadAttention import MultiHeadedAttention
import pytest

def test_multi_head_attention_io_shape():
    B, T, E = 4,10,64
    H = 4 
    test_tensor = torch.randn(size=(B, T, E))

    attention = MultiHeadedAttention(
        embed_dim=E,
        num_heads=H
    )
    output = attention(test_tensor, mask = 'causal')

    assert output.shape == (B,T,E)




def test_multi_head_attention_causalilty_for_mask():
    B, T, E = 4,10,64
    H = 4
    test_tensor1 = torch.randn(size=(B, T, E))
    test_tensor2 = test_tensor1.clone()
    test_tensor2[:,2:,:] += 100

    attention = MultiHeadedAttention(
        embed_dim=E,
        num_heads=H
    )
    output1 = attention(test_tensor1, mask = 'causal')
    output2 = attention(test_tensor2, mask = 'causal')

    assert torch.allclose(
        output1[:,:2,:],
        output2[:,:2:,:]
    )


def test_multi_head_attention_non_casuality_for_mask():
    B, T, E = 4,10,64
    H = 4 
    test_tensor1 = torch.randn(size=(B, T, E))
    test_tensor2 = torch.clone(test_tensor1)
    test_tensor2[:,2:,:] += 100

    attention = MultiHeadedAttention(
        embed_dim=E,
        num_heads=H,
    )
    output1 = attention(test_tensor1, mask = 'causal')
    output2 = attention(test_tensor2, mask = None)

    assert not torch.allclose(
        output1[:,1:,:],
        output2[:,1:,:]
    )

@pytest.mark.parametrize(
    "B,T,E,H",
    [
        (1, 1, 8, 1),
        (1, 5, 32, 4),
        (4, 10, 64, 8),
        (8, 20, 128, 8),
    ]
)
def test_multihead_attention_backward(B, T, E, H):
    
    attention = MultiHeadedAttention(
        embed_dim=E,
        num_heads=H
    )

    test_tensor = torch.randn(
        B, T, E,
        requires_grad=True
    )

    output = attention(test_tensor)

    loss = output.mean()
    loss.backward()

    assert test_tensor.grad is not None

    for name, parameter in attention.named_parameters():
        assert parameter.grad is not None, \
            f"{name} has no gradient"