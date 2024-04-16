from typing import Any, Callable, Dict, List, Tuple, Union

import torch
import torchode
from tensordict import TensorDict
from torch import nn
from torchdyn.core import NeuralODE
from torchdyn.numerics import Euler
from torchdyn.numerics import odeint as torchdyn_odeint


class ODESolver:
    """
    Overview:
        The ODE solver class.
    Interfaces:
        ``__init__``, ``integrate``
    """

    def __init__(
        self,
        ode_solver="euler",
        dt=0.01,
        atol=1e-5,
        rtol=1e-5,
        library="torchdyn",
        **kwargs,
    ):
        """
        Overview:
            Initialize the ODE solver using torchode or torchdyn library.
        Arguments:
            ode_solver (:obj:`str`): The ODE solver to use.
            dt (:obj:`float`): The time step.
            atol (:obj:`float`): The absolute tolerance.
            rtol (:obj:`float`): The relative tolerance.
            library (:obj:`str`): The library to use for the ODE solver. Currently, it supports 'torchdyn' and 'torchode'.
            **kwargs: Additional arguments for the ODE solver.
        """
        self.ode_solver = ode_solver
        self.dt = dt
        self.atol = atol
        self.rtol = rtol
        self.nfe = 0
        self.kwargs = kwargs
        self.library = library

    def integrate(
            self,
            drift: Union[nn.Module, Callable],
            x0: Union[torch.Tensor, TensorDict],
            t_span: torch.Tensor,
        ):
        """
        Overview:
            Integrate the ODE.
        Arguments:
            drift (:obj:`Union[nn.Module, Callable]`): The drift term of the ODE.
            x0 (:obj:`Union[torch.Tensor, TensorDict]`): The input initial state.
            t_span (:obj:`torch.Tensor`): The time at which to evaluate the ODE. The first element is the initial time, and the last element is the final time. For example, t = torch.tensor([0.0, 1.0]).
        Returns:
            trajectory (:obj:`Union[torch.Tensor, TensorDict]`): The output trajectory of the ODE, which has the same data type as x0 and the shape of (len(t_span), *x0.shape).
        """

        self.nfe = 0
        if self.library == "torchdyn":
            return self.odeint_by_torchdyn(drift, x0, t_span)
        elif self.library == "torchdyn_NeuralODE":
            return self.odeint_by_torchdyn_NeuralODE(drift, x0, t_span)
        elif self.library == "torchode":
            return self.odeint_by_torchode(drift, x0, t_span)
        else:
            raise ValueError(f"library {self.library} is not supported")

    def forward_ode_drift_by_torchdyn(self, t, x):
        self.nfe += 1
        # broadcasting t to match the batch size of x
        t = t.repeat(x.shape[0])
        return self.drift(t, x)

    def forward_ode_drift_by_torchdyn_NeuralODE(self, t, x, args):
        self.nfe += 1
        # broadcasting t to match the batch size of x
        t = t.repeat(x.shape[0])
        return self.drift(t, x)

    def forward_ode_drift_by_torchode(self, t, x):
        self.nfe += 1
        #TODO: implement forward_ode_drift_by_torchode
        pass

    def odeint_by_torchdyn(self, drift, x0, t_span):
        
        def forward_ode_drift_by_torchdyn(t, x):
            self.nfe += 1
            # broadcasting t to match the batch size of x
            t = t.repeat(x.shape[0])
            return drift(t, x)

        t_eval, trajectory = torchdyn_odeint(
            f=forward_ode_drift_by_torchdyn,
            x=x0,
            t_span=t_span,
            solver=self.ode_solver,
            atol=self.atol,
            rtol=self.rtol,
            **self.kwargs,
        )
        return trajectory

    def odeint_by_torchdyn_NeuralODE(self, drift, x0, t_span):

        def forward_ode_drift_by_torchdyn_NeuralODE(t, x, args):
            self.nfe += 1
            # broadcasting t to match the batch size of x
            t = t.repeat(x.shape[0])
            return drift(t, x)

        neural_ode = NeuralODE(
            vector_field=forward_ode_drift_by_torchdyn_NeuralODE,
            solver=self.ode_solver,
            atol=self.atol,
            rtol=self.rtol,
            return_t_eval=False,
            **self.kwargs,
        )
        trajectory = neural_ode(drift, x0, t_span)
        return trajectory

    def odeint_by_torchode(self, x0, t_span):
        pass
