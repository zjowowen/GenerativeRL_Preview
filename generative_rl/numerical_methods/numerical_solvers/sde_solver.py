
from typing import Union, Tuple, List, Dict, Any
import torch
from torch import nn
import torchsde

class TorchSDE(torch.nn.Module):
    """
    Overview:
        The SDE class for torchsde library, wich is an object with methods `f` and `g` representing the drift and diffusion.
        The output of `g` should be a single tensor of size (batch_size, d) for diagonal noise SDEs or (batch_size, d, m) for SDEs of other noise types,
        where d is the dimensionality of state and m is the dimensionality of Brownian motion.
    """
    def __init__(
        self,
        drift,
        diffusion,
        noise_type,
        sde_type,
    ):
        """
        Overview:
            Initialize the SDE object.
        Arguments:
            - drift (:obj:`nn.Module`): The function that defines the drift of the SDE.
            - diffusion (:obj:`nn.Module`): The function that defines the diffusion of the SDE.
            - noise_type (:obj:`str`): The type of noise of the SDE. It can be 'diagonal', 'general', 'scalar' or 'additive'.
            - sde_type (:obj:`str`): The type of the SDE. It can be 'ito' or 'stratonovich'.
        """
        super().__init__()
        self.drift = drift
        self.diffusion = diffusion
        
        self.noise_type = noise_type
        self.sde_type = sde_type

    def f(self, t, y):
        """
        Overview:
            The drift function of the SDE.
        """
        return self.drift(t, y)

    def g(self, t, y):
        """
        Overview:
            The diffusion function of the SDE.
        """
        return self.diffusion(t, y)


class SDESolver(nn.Module):

    def __init__(
        self,
        drift,
        diffusion,
        data_size: Union[torch.Size, int, Tuple[int], List[int], Dict[Any, Any]],
        sde_solver="euler",
        sde_noise_type="general",
        sde_type="ito",
        dt=0.01,
        atol=1e-5,
        rtol=1e-5,
        library="torchsde",
        **kwargs,
    ):
        """
        Overview:
            Initialize the SDE solver using torchsde library.
        Arguments:
            - drift (:obj:`nn.Module`): The function that defines the ODE.
            - diffusion (:obj:`nn.Module`): The function that defines the ODE.
            - data_size (:obj:`int` or :obj:`tuple`): The dimension of the variable.
            - sde_solver (:obj:`str`): The SDE solver to use.
            - sde_noise_type (:obj:`str`): The type of noise of the SDE. It can be 'diagonal', 'general', 'scalar' or 'additive'.
            - sde_type (:obj:`str`): The type of the SDE. It can be 'ito' or 'stratonovich'.
            - dt (:obj:`float`): The time step.
            - atol (:obj:`float`): The absolute tolerance.
            - rtol (:obj:`float`): The relative tolerance.
            - library (:obj:`str`): The library to use for the ODE solver.
            - **kwargs: Additional arguments for the ODE solver.
        """
        super().__init__()
        self.drift = drift
        self.diffusion = diffusion
        self.data_size = data_size
        self.sde_solver = sde_solver
        self.sde_noise_type = sde_noise_type
        self.sde_type = sde_type
        self.dt = dt
        self.atol = atol
        self.rtol = rtol
        self.nfe_drift = 0
        self.nfe_diffusion = 0
        self.kwargs = kwargs
        self.library = library

    def forward_drift(self, t, x):
        self.nfe_drift += 1
        return self.drift(t, x)
    
    def forward_diffusion(self, t, x):
        self.nfe_diffusion += 1
        return self.diffusion(t, x)

    def integrate(self, x0, t_span, logqp=False, adaptive=False):
        self.nfe_drift = 0
        self.nfe_diffusion = 0

        sde = TorchSDE(
            drift=self.forward_drift,
            diffusion=self.forward_diffusion,
            noise_type=self.sde_noise_type,
            sde_type=self.sde_type
        )

        trajectory = torchsde.sdeint(
            sde,
            x0,
            t_span,
            method=self.sde_solver,
            dt=self.dt,
            rtol=self.rtol,
            atol=self.atol,
            logqp=logqp,
            adaptive=adaptive,
            **self.kwargs,
        )

        return trajectory
    
