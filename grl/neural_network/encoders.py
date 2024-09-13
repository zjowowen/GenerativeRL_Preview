import math

import numpy as np
import torch
import torch.nn as nn

def register_encoder(module: nn.Module, name: str):
    """
    Overview:
        Register the encoder to the module dictionary.
    Arguments:
        - module (:obj:`nn.Module`): The module to be registered.
        - name (:obj:`str`): The name of the module.
    """
    global ENCODERS
    if name.lower() in ENCODERS:
        raise KeyError(f"Encoder {name} is already registered.")
    ENCODERS[name.lower()] = module

def get_encoder(type: str):
    """
    Overview:
        Get the encoder module by the encoder type.
    Arguments:
        type (:obj:`str`): The encoder type.
    """

    if type.lower() in ENCODERS:
        return ENCODERS[type.lower()]
    else:
        raise ValueError(f"Unknown encoder type: {type}")


class GaussianFourierProjectionTimeEncoder(nn.Module):
    r"""
    Overview:
        Gaussian random features for encoding time variable.
        This module is used as the encoder of time in generative models such as diffusion model.
        It transforms the time :math:`t` to a high-dimensional embedding vector :math:`\phi(t)`.
        The output embedding vector is computed as follows:

        .. math::

            \phi(t) = [ \sin(t \cdot w_1), \cos(t \cdot w_1), \sin(t \cdot w_2), \cos(t \cdot w_2), \ldots, \sin(t \cdot w_{\text{embed\_dim} / 2}), \cos(t \cdot w_{\text{embed\_dim} / 2}) ]

        where :math:`w_i` is a random scalar sampled from the Gaussian distribution.
    Interfaces:
        ``__init__``, ``forward``.
    """

    def __init__(self, embed_dim, scale=30.0):
        """
        Overview:
            Initialize the Gaussian Fourier Projection Time Encoder according to arguments.
        Arguments:
            embed_dim (:obj:`int`): The dimension of the output embedding vector.
            scale (:obj:`float`): The scale of the Gaussian random features.
        """
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(
            torch.randn(embed_dim // 2) * scale * 2 * np.pi, requires_grad=False
        )

    def forward(self, x):
        """
        Overview:
            Return the output embedding vector of the input time step.
        Arguments:
            x (:obj:`torch.Tensor`): Input time step tensor.
        Returns:
            output (:obj:`torch.Tensor`): Output embedding vector.
        Shapes:
            x (:obj:`torch.Tensor`): :math:`(B,)`, where B is batch size.
            output (:obj:`torch.Tensor`): :math:`(B, embed_dim)`, where B is batch size, embed_dim is the \
                dimension of the output embedding vector.
        Examples:
            >>> encoder = GaussianFourierProjectionTimeEncoder(128)
            >>> x = torch.randn(100)
            >>> output = encoder(x)
        """
        x_proj = x[..., None] * self.W[None, :]
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class GaussianFourierProjectionEncoder(nn.Module):
    r"""
    Overview:
        Gaussian random features for encoding variables.
        This module can be seen as a generalization of GaussianFourierProjectionTimeEncoder for encoding multi-dimensional variables.
        It transforms the input tensor :math:`x` to a high-dimensional embedding vector :math:`\phi(x)`.
        The output embedding vector is computed as follows:

        .. math::

                \phi(x) = [ \sin(x \cdot w_1), \cos(x \cdot w_1), \sin(x \cdot w_2), \cos(x \cdot w_2), \ldots, \sin(x \cdot w_{\text{embed\_dim} / 2}), \cos(x \cdot w_{\text{embed\_dim} / 2}) ]

        where :math:`w_i` is a random scalar sampled from the Gaussian distribution.
    Interfaces:
        ``__init__``, ``forward``.
    """

    def __init__(self, embed_dim, x_shape, flatten=True, scale=30.0):
        """
        Overview:
            Initialize the Gaussian Fourier Projection Time Encoder according to arguments.
        Arguments:
            embed_dim (:obj:`int`): The dimension of the output embedding vector.
            x_shape (:obj:`tuple`): The shape of the input tensor.
            flatten (:obj:`bool`): Whether to flatten the output tensor afyer applying the encoder.
            scale (:obj:`float`): The scale of the Gaussian random features.
        """
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(
            torch.randn(embed_dim // 2) * scale * 2 * np.pi, requires_grad=False
        )
        self.x_shape = x_shape
        self.flatten = flatten

    def forward(self, x):
        """
        Overview:
            Return the output embedding vector of the input time step.
        Arguments:
            x (:obj:`torch.Tensor`): Input time step tensor.
        Returns:
            output (:obj:`torch.Tensor`): Output embedding vector.
        Shapes:
            x (:obj:`torch.Tensor`): :math:`(B, D)`, where B is batch size.
            output (:obj:`torch.Tensor`): :math:`(B, D * embed_dim)` if flatten is True, otherwise :math:`(B, D, embed_dim)`.
                where B is batch size, embed_dim is the dimension of the output embedding vector, D is the shape of the input tensor.
        Examples:
            >>> encoder = GaussianFourierProjectionTimeEncoder(128)
            >>> x = torch.randn(torch.Size([100, 10]))
            >>> output = encoder(x)
        """
        x_proj = x[..., None] * self.W[None, :]
        x_proj = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

        # if x shape is (B1, ..., Bn, **x_shape), then the output shape is (B1, ..., Bn, np.prod(x_shape) * embed_dim)
        if self.flatten:
            x_proj = torch.flatten(x_proj, start_dim=-1 - self.x_shape.__len__())

        return x_proj


class ExponentialFourierProjectionTimeEncoder(nn.Module):
    r"""
    Overview:
        Expoential Fourier Projection Time Encoder.
        It transforms the time :math:`t` to a high-dimensional embedding vector :math:`\phi(t)`.
        The output embedding vector is computed as follows:

        .. math::

                \phi(t) = [ \sin(t \cdot w_1), \cos(t \cdot w_1), \sin(t \cdot w_2), \cos(t \cdot w_2), \ldots, \sin(t \cdot w_{\text{embed\_dim} / 2}), \cos(t \cdot w_{\text{embed\_dim} / 2}) ]

            where :math:`w_i` is a random scalar sampled from a uniform distribution, then transformed by exponential function.
        There is an additional MLP layer to transform the frequency embedding:

        .. math::

            \text{MLP}(\phi(t)) = \text{SiLU}(\text{Linear}(\text{SiLU}(\text{Linear}(\phi(t)))))

    Interfaces:
        ``__init__``, ``forward``
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        """
        Overview:
            Initialize the timestep embedder.
        Arguments:
            hidden_size (:obj:`int`): The hidden size.
            frequency_embedding_size (:obj:`int`): The size of the frequency embedding.
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    # TODO: simplify this function
    @staticmethod
    def timestep_embedding(t, embed_dim, max_period=10000):
        """
        Overview:
            Create sinusoidal timestep embeddings.
        Arguments:
            t (:obj:`torch.Tensor`): a 1-D Tensor of N indices, one per batch element. These may be fractional.
            embed_dim (:obj:`int`): the dimension of the output.
            max_period (:obj:`int`): controls the minimum frequency of the embeddings.
        """

        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = embed_dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        if len(t.shape) == 0:
            t = t.unsqueeze(0)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if embed_dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t: torch.Tensor):
        """
        Overview:
            Return the output embedding vector of the input time step.
        Arguments:
            t (:obj:`torch.Tensor`): Input time step tensor.
        """
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class  TensorDictencoder(torch.nn.Module):
    def __init__(self):
        super(TensorDictencoder, self).__init__()

    def forward(self, x: dict) -> torch.Tensor:
        tensors = []
        for v in x.values():
            if v.dim() == 3 and v.shape[0] == 1:
                v = v.view(1, -1)
            tensors.append(v)
        x = torch.cat(tensors, dim=1)
        return x

class  TensorDictencoder(torch.nn.Module):
    def __init__(self,usePixel=False,useRichData=True):
        super(TensorDictencoder, self).__init__()
        self.usePixel=usePixel
        self.useRichData=useRichData
        if self.usePixel ==False:
            self.useRichData=True
        else:
            from torchvision.transforms import v2
            self.image_transform = transforms = v2.Compose([
                # v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
            self.conv_block = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                
                nn.Flatten()
            )
            self.mlp_block = nn.Sequential(
                nn.Linear(64 * 16 * 16, 512),
                nn.ReLU(),
                nn.Linear(512, 128),
            )
            
    def forward(self, x: dict) -> torch.Tensor:
        tensors = []
        for v in x.values():
            if v.dim() == 1 and self.useRichData == True:
                v = v.unsqueeze(-1)
            if v.dim() == 3 and v.shape[0] == 1:
                import ipdb
                ipdb.set_trace()
                v = v.view(1, -1)
            if v.dim() == 2 and self.useRichData == True:
                tensors.append(v)
            if v.dim() == 4 and self.usePixel== True:
                v = v.permute(0, 3, 1, 2)/255.0
                v = self.image_transform(v)
                v = self.conv_block(v)
                v = self.mlp_block(v)
                tensors.append(v)
        new = torch.cat(tensors, dim=1)
        return new

class  walker_encoder(torch.nn.Module):
    def __init__(self,mean,std,min_val,max_val):
        super(walker_encoder, self).__init__()
        self.mean=mean
        self.std=std
        self.min_val=min_val
        self.max_val=max_val        
        self.orientation_mlp = nn.Sequential(
            nn.Linear(14, 28),
            nn.ReLU(),
            nn.Linear(28, 28)
        )
        
        self.velocity_mlp = nn.Sequential(
            nn.Linear(9, 18),
            nn.ReLU(),
            nn.Linear(18, 18),
        )
        
        self.height_mlp = nn.Sequential(
            nn.Linear(1, 2),
            nn.ReLU(),
            nn.Linear(2, 2),
        )
        
            
    def forward(self, x: dict) -> torch.Tensor:
        orientation_output = self.orientation_mlp(x['orientations'])
        if x['velocity'].shape[0]==1:
            device=x['velocity'].device
            self.mean = torch.tensor( self.mean, device=device)
            self.std = torch.tensor(self.std, device=device)
            self.min_val = torch.tensor(self.min_val, device=device)
            self.max_val = torch.tensor(self.max_val, device=device)
            standardized_data = (x['velocity'] - self.mean) / self.std
            data = (standardized_data - self.min_val) / (self.max_val - self.min_val)
            x['velocity'] = data*2-1
        velocity_output = self.velocity_mlp(x['velocity'])
        height=x["height"]
        if height.dim() == 1:  # Check if it's [b]
            height = height.unsqueeze(-1)  # Expand to [b, 1]
        height_output = self.height_mlp(height)        
        combined_output = torch.cat([orientation_output, velocity_output, height_output], dim=-1)
        return combined_output


ENCODERS = {
    "GaussianFourierProjectionTimeEncoder".lower(): GaussianFourierProjectionTimeEncoder,
    "GaussianFourierProjectionEncoder".lower(): GaussianFourierProjectionEncoder,
    "ExponentialFourierProjectionTimeEncoder".lower(): ExponentialFourierProjectionTimeEncoder,
    "SinusoidalPosEmb".lower(): SinusoidalPosEmb,
    "TensorDictencoder".lower(): TensorDictencoder,
    "walker_encoder".lower(): walker_encoder,
}
