from typing import Union
from easydict import EasyDict
import torch
from torch import nn
from tensordict import TensorDict

class base_deterministic_modules(nn.Module):
    def __init__(self, config: EasyDict):
        super().__init__()
        pass

    def forward(self, x: Union[torch.Tensor, TensorDict]) -> Union[torch.Tensor, TensorDict]:
        pass

