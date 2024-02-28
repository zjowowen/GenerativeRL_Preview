
from typing import Union, Tuple, List, Dict, Any
import torch
from torch import nn
import torchode
from torchdyn.core import NeuralODE

class ODESolver(nn.Module):

    def __init__(
        self,
        drift,
        data_size: Union[torch.Size, int, Tuple[int], List[int], Dict[Any, Any]],
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
            - drift (:obj:`nn.Module`): The function that defines the ODE.
            - data_size (:obj:`int` or :obj:`tuple`): The dimension of the variable.
            - ode_solver (:obj:`str`): The ODE solver to use.
            - dt (:obj:`float`): The time step.
            - atol (:obj:`float`): The absolute tolerance.
            - rtol (:obj:`float`): The relative tolerance.
            - library (:obj:`str`): The library to use for the ODE solver.
            - **kwargs: Additional arguments for the ODE solver.
        """
        super().__init__()
        self.drift = drift
        self.data_size = data_size
        self.ode_solver = ode_solver
        self.dt = dt
        self.atol = atol
        self.rtol = rtol
        self.nfe = 0
        self.kwargs = kwargs
        self.library = library

    def forward_ode_drift(self, t, x):
        self.nfe += 1
        return self.drift(t, x)

    def integrate(self, x0, t_span):
        self.nfe = 0
        if self.library == "torchdyn":
            return self.odeint_by_torchdyn(x0, t_span)
        else:
            return self.odeint_by_torchode(x0, t_span)


    def odeint_by_torchdyn(self, x0, t_span):

        neural_ode = NeuralODE(
            self.forward_ode_drift,
            solver=self.ode_solver,
            atol=self.atol,
            rtol=self.rtol,
            return_t_eval=False,
        )
        t, trajectory = neural_ode(x0, t_span)
        return trajectory

    def odeint_by_torchode(self, x0, t_span):
        pass


