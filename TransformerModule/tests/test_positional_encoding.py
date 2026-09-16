import torch 
import pytest 
from Components.positionalEncoding import PositionalEncoder 

@pytest.fixture(
    params = [
        (1,12,64),
        (1,3,128),
        (1,3,4),
        (1,8,8),
        (1,32,1024)
    ]
)
def test_config(request):
    B,T,E = request.param
    return B,T,E


def test_positinal_i_o_shape(test_config):
    B,T,E = test_config
    pos_encoder = PositionalEncoder(E)
    
    test_tensor = torch.rand(size=(B,T,E))
    output1 = pos_encoder(test_tensor , True)
    output2 = pos_encoder(test_tensor , False)
    
    assert output1.shape == test_tensor.shape
    assert output2.shape == test_tensor.shape