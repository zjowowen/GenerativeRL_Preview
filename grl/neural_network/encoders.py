import numpy as np
import torch
import torch.nn as nn


class GaussianFourierProjectionTimeEncoder(nn.Module):
    """
    Overview:
        Gaussian random features for encoding time steps.
        This module is used as the encoder of time in generative models such as diffusion model.
    Interfaces:
        ``__init__``, ``forward``.
    """

    def __init__(self, embed_dim, scale=30.):
        """
        Overview:
            Initialize the Gaussian Fourier Projection Time Encoder according to arguments.
        Arguments:
            - embed_dim (:obj:`int`): The dimension of the output embedding vector.
            - scale (:obj:`float`): The scale of the Gaussian random features.
        """
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale * 2 * np.pi, requires_grad=False)

    def forward(self, x):
        """
        Overview:
            Return the output embedding vector of the input time step.
        Arguments:
            - x (:obj:`torch.Tensor`): Input time step tensor.
        Returns:
            - output (:obj:`torch.Tensor`): Output embedding vector.
        Shapes:
            - x (:obj:`torch.Tensor`): :math:`(B,)`, where B is batch size.
            - output (:obj:`torch.Tensor`): :math:`(B, embed_dim)`, where B is batch size, embed_dim is the \
                dimension of the output embedding vector.
        Examples:
            >>> encoder = GaussianFourierProjectionTimeEncoder(128)
            >>> x = torch.randn(100)
            >>> output = encoder(x)
        """
        x_proj = x[..., None] * self.W[None, :]
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class GaussianFourierProjectionEncoder(nn.Module):
    """
    Overview:
        Gaussian random features for encoding time steps.
        This module is used as the encoder of time in generative models such as diffusion model.
    Interfaces:
        ``__init__``, ``forward``.
    """

    def __init__(self, embed_dim, x_shape, scale=30.):
        """
        Overview:
            Initialize the Gaussian Fourier Projection Time Encoder according to arguments.
        Arguments:
            - embed_dim (:obj:`int`): The dimension of the output embedding vector.
            - x_shape (:obj:`tuple`): The shape of the input tensor.
            - scale (:obj:`float`): The scale of the Gaussian random features.
        """
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale * 2 * np.pi, requires_grad=False)
        self.x_shape = x_shape

    def forward(self, x):
        """
        Overview:
            Return the output embedding vector of the input time step.
        Arguments:
            - x (:obj:`torch.Tensor`): Input time step tensor.
        Returns:
            - output (:obj:`torch.Tensor`): Output embedding vector.
        Shapes:
            - x (:obj:`torch.Tensor`): :math:`(B,)`, where B is batch size.
            - output (:obj:`torch.Tensor`): :math:`(B, embed_dim)`, where B is batch size, embed_dim is the \
                dimension of the output embedding vector.
        Examples:
            >>> encoder = GaussianFourierProjectionTimeEncoder(128)
            >>> x = torch.randn(100)
            >>> output = encoder(x)
        """
        x_proj = x[..., None] * self.W[None, :]
        x_proj = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

        # if x shape is (B1, ..., Bn, **x_shape), then the output shape is (B1, ..., Bn, np.prod(x_shape) * embed_dim)
        x_proj = torch.flatten(x_proj, start_dim=-1-self.x_shape.__len__())
        return x_proj


ENCODERS={
    "GaussianFourierProjectionTimeEncoder".lower(): GaussianFourierProjectionTimeEncoder,
    "GaussianFourierProjectionEncoder".lower(): GaussianFourierProjectionEncoder,
}

def get_encoder(type: str):

    if type.lower() in ENCODERS:
        return ENCODERS[type.lower()]
    else:
        raise ValueError(f"Unknown encoder type: {type}")
