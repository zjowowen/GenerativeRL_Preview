from typing import Union, Callable
from easydict import EasyDict
import torch
from torch import nn
from torch.distributions import Distribution
from tensordict import TensorDict

from generative_rl.numerical_methods.numerical_solvers.ode_solver import ODESolver

class ODE:
    """
    Overview:
        Base class for ordinary differential equations.
        The ODE is defined as:
            dx = f(x, t)dt
        where f(x, t) is the drift term.

    Interfaces:
        ``__init__``, ``sample``
    """

    def __init__(
            self,
            drift: Union[nn.Module, Callable] = None,
        ):
        self.drift = drift

    def sample(
            self,
            t: torch.Tensor,
            x: Union[torch.Tensor, TensorDict],
            solver_kwargs: EasyDict = None,
        ) -> Union[torch.Tensor, TensorDict]:
        """
        Overview:
            Return the output of the ODE model by integrating the ODE.
        Arguments:
            - t (:obj:`torch.Tensor`): The time at which to evaluate the ODE.
            - x (:obj:`Union[torch.Tensor, TensorDict]`): The input state.
            - solver_kwargs (:obj:`EasyDict`): The keyword arguments for the ODE solver.
        """
        ode_solver = ODESolver(
            drift=self.drift,
            data_size=x.size(),
            **solver_kwargs,
        )

        _, x_t = ode_solver.integrate(
            x0=x,
            t=t,
        )

        return x_t
