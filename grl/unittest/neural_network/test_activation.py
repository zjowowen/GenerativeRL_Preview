# Test grl/neural_network/activation.py

def test_activation():
    import torch
    from torch import nn
    from grl.neural_network.activation import Swish, Lambda, ACTIVATIONS, get_activation

    assert ACTIVATIONS["mish"] == nn.Mish()
    assert ACTIVATIONS["tanh"] == nn.Tanh()
    assert ACTIVATIONS["relu"] == nn.ReLU()
    assert ACTIVATIONS["softplus"] == nn.Softplus()
    assert ACTIVATIONS["elu"] == nn.ELU()
    assert ACTIVATIONS["silu"] == nn.SiLU()
    assert ACTIVATIONS["swish"] == Swish()
    assert ACTIVATIONS["square"] == Lambda(lambda x: x**2)
    assert ACTIVATIONS["identity"] == Lambda(lambda x: x)

    assert get_activation("mish") == nn.Mish()
    assert get_activation("tanh") == nn.Tanh()
    assert get_activation("relu") == nn.ReLU()
    assert get_activation("softplus") == nn.Softplus()
    assert get_activation("elu") == nn.ELU()
    assert get_activation("silu") == nn.SiLU()
    assert get_activation("swish") == Swish()
    assert get_activation("square") == Lambda(lambda x: x**2)
    assert get_activation("identity") == Lambda(lambda x: x)

    try:
        get_activation("unknown")
    except ValueError as e:
        assert str(e) == "Unknown activation function unknown"
