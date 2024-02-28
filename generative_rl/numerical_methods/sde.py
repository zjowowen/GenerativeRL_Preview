from typing import Union, Callable
from easydict import EasyDict
import torch
from torch import nn
from torch.distributions import Distribution
from tensordict import TensorDict

from generative_rl.numerical_methods.numerical_solvers.sde_solver import SDESolver

class SDE:
    """
    Overview:
        Base class for stochastic differential equations.
        The SDE is defined as:
            dx = f(x, t)dt + g(x, t)dW
        where f(x, t) is the drift term, g(x, t) is the diffusion term, and dW is the Wiener process.

    Interfaces:
        ``__init__``, ``sample``
    """

    def __init__(
            self,
            drift: Union[nn.Module, Callable] = None,
            diffusion: Union[nn.Module, Callable] = None,
        ):
        self.drift = drift
        self.diffusion = diffusion

    def sample(
            self,
            t: torch.Tensor,
            x: Union[torch.Tensor, TensorDict],
            solver_kwargs: EasyDict = None,
        ) -> Union[torch.Tensor, TensorDict]:
        """
        Overview:
            Return the output of the SDE model by integrating the SDE.
        Arguments:
            - delta_t (:obj:`torch.Tensor`): The time step.
            - t (:obj:`torch.Tensor`): The time at which to evaluate the SDE.
            - x (:obj:`Union[torch.Tensor, TensorDict]`): The input state.
            - solver_kwargs (:obj:`EasyDict`): The keyword arguments for the SDE solver.
        """
        sde_solver = SDESolver(
            drift=self.drift,
            diffusion=self.diffusion,
            data_size=x.size(),
            **solver_kwargs,
        )

        _, x_t = sde_solver.integrate(
            x0=x,
            t=t,
        )

        return x_t
