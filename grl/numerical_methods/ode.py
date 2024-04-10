from typing import Union, Callable
import torch
from torch import nn
from tensordict import TensorDict

class ODE:
    """
    Overview:
        Base class for ordinary differential equations.
        The ODE is defined as:
        .. math::
            dx = f(x, t)dt
        where f(x, t) is the drift term.

    Interfaces:
        ``__init__``
    """

    def __init__(
            self,
            drift: Union[nn.Module, Callable] = None,
        ):
        self.drift = drift
