from typing import Union
from easydict import EasyDict
import torch
from torch import nn
from torch.distributions import Distribution
from tensordict import TensorDict

from torch.distributions.constraints import Constraint

class base_stochastic_modules(nn.Module):
    def __init__(self, config: EasyDict):
        super().__init__()
        pass

    def forward(self, x: Union[torch.Tensor, TensorDict]) -> Distribution:
        pass

