from typing import Union, Callable
from easydict import EasyDict
import torch
import torch.nn as nn
from torch.distributions import Distribution
from tensordict import TensorDict

from generative_rl.numerical_methods.probability_path import GaussianConditionalProbabilityPath
from generative_rl.numerical_methods.sde import SDE
from generative_rl.numerical_methods.ode import ODE

class VPSDE:
    """
    Overview:
        Model of Variance-Preserving Stochastic Differential Equations.
        .. math::
            \mathrm{d}x=-\frac{1}{2}\beta(t)x\mathrm{d}t+\sqrt{\beta(t)}\mathrm{d}w_{t}
    """

    def __init__(
            self,
            gaussian_conditional_probability_path: GaussianConditionalProbabilityPath
        ):
        super().__init__()
        self.gaussian_conditional_probability_path = gaussian_conditional_probability_path
        self.t_max = torch.tensor(1.)

    def drift(
            self,
            t: torch.Tensor,
            x: Union[torch.Tensor, TensorDict],
            condition: Union[torch.Tensor, TensorDict] = None,
        ) -> Union[torch.Tensor, TensorDict]:
        """
        Overview:
            Return the drift term of the VP SDE.
            The drift term is given by the following:
            .. math::
                f(x,t)=-\frac{1}{2}\beta(t)x
        Arguments:
            - t (:obj:`torch.Tensor`): The input time.
            - x (:obj:`Union[torch.Tensor, TensorDict]`): The input state.            
            - condition (:obj:`Union[torch.Tensor, TensorDict]`): The input condition.
        Returns:
            - drift (:obj:`Union[torch.Tensor, TensorDict]`): The output drift term.
        """

        return self.gaussian_conditional_probability_path.drift(t, x)
    
    def diffusion(
            self,
            t: torch.Tensor,
            x: Union[torch.Tensor, TensorDict],
            condition: Union[torch.Tensor, TensorDict] = None,
        ) -> Union[torch.Tensor, TensorDict]:
        """
        Overview:
            Return the diffusion term of the VP SDE.
            The diffusion term is given by the following:
            .. math::
                g(x,t)=\sqrt{\beta(t)}
        Arguments:
            - t (:obj:`torch.Tensor`): The input time.
            - x (:obj:`Union[torch.Tensor, TensorDict]`): The input state.
            - condition (:obj:`Union[torch.Tensor, TensorDict]`): The input condition.
        Returns:
            - diffusion (:obj:`Union[torch.Tensor, TensorDict]`): The output diffusion term.
        """

        return self.gaussian_conditional_probability_path.diffusion(t, x)

    def diffusion_squared(
            self,
            t: torch.Tensor,
            x: Union[torch.Tensor, TensorDict],
            condition: Union[torch.Tensor, TensorDict] = None,
        ) -> Union[torch.Tensor, TensorDict]:
        """
        Overview:
            Return the square of the diffusion term of the VP SDE.
            The square of the diffusion term is given by the following:
            .. math::
                g^2(x,t)=\beta(t)
        Arguments:
            - t (:obj:`torch.Tensor`): The input time.
            - x (:obj:`Union[torch.Tensor, TensorDict]`): The input state.
            - condition (:obj:`Union[torch.Tensor, TensorDict]`): The input condition.
        Returns:
            - diffusion_squared (:obj:`Union[torch.Tensor, TensorDict]`): The output square of the diffusion term.
        """

        return self.gaussian_conditional_probability_path.diffusion_squared(t, x)

    def sde(self, condition: Union[torch.Tensor, TensorDict] = None) -> SDE:
        """
        Overview:
            Return the SDE of VP SDE with the input condition.
        Arguments:
            - condition (:obj:`Union[torch.Tensor, TensorDict]`): The input condition.
        Returns:
            - sde (:obj:`SDE`): The SDE of VP SDE.
        """
        
        def sde_drift(
                t: torch.Tensor,
                x: Union[torch.Tensor, TensorDict],
        ) -> Union[torch.Tensor, TensorDict]:
            return self.drift(t, x, condition)
        
        def sde_diffusion(
                t: torch.Tensor,
                x: Union[torch.Tensor, TensorDict],
        ) -> Union[torch.Tensor, TensorDict]:
            return self.diffusion(t, x, condition)

        return SDE(drift=sde_drift, diffusion=sde_diffusion)

    def reverse_sde(
            self,
            score_function: Union[Callable, nn.Module],
            condition: Union[torch.Tensor, TensorDict] = None,
            T: torch.Tensor = torch.tensor(1.),
            ) -> SDE:
        """
        Overview:
            Return the reversed time SDE of the VP SDE with the input condition.
        Arguments:
            - T (:obj:`torch.Tensor`): The time at which the SDE is reversed.
            - condition (:obj:`Union[torch.Tensor, TensorDict]`): The input condition.
        Returns:
            - reverse_sde (:obj:`SDE`): The reverse VP SDE.
        """

        def reverse_sde_drift(
                t: torch.Tensor,
                x: Union[torch.Tensor, TensorDict],
        ) -> Union[torch.Tensor, TensorDict]:
            return - self.drift(T - t, x, condition) + self.diffusion_squared(T - t, x, condition) * score_function(T - t, x, condition)

        def reverse_sde_diffusion(
                t: torch.Tensor,
                x: Union[torch.Tensor, TensorDict],
        ) -> Union[torch.Tensor, TensorDict]:
            return self.diffusion(T - t, x, condition)

        return SDE(drift=reverse_sde_drift, diffusion=reverse_sde_diffusion)

    def reverse_ode(
            self,
            score_function: Union[Callable, nn.Module],
            condition: Union[torch.Tensor, TensorDict] = None,
            T: torch.Tensor = torch.tensor(1.),
            ) -> ODE:
        """
        Overview:
            Return the reversed time ODE of the VP SDE with the input condition.
        """
        
        def reverse_ode_drift(
                t: torch.Tensor,
                x: Union[torch.Tensor, TensorDict],
        ) -> Union[torch.Tensor, TensorDict]:
            return - self.drift(T - t, x, condition) + 0.5 * self.diffusion_squared(T - t, x, condition) * score_function(T - t, x, condition)
        
        return ODE(drift=reverse_ode_drift)
        
    def forward(
            self,
            t: torch.Tensor,
            x: Union[torch.Tensor, TensorDict],
            condition: Union[torch.Tensor, TensorDict] = None,
        ) -> Distribution:
        """
        Overview:
            Return the distribution of the state at time given the initial state.
            VP SDE is a linear stochastic differential equation of the general form:
            .. math::
                \mathrm{d}x=f(t)x\mathrm{d}t+u(t)\mathrm{d}t+L(t)\mathrm{d}w
            The solution is a linear transformation of Brownian motion, which is a Gaussian process.
            For VP SDE, the solution at time t is a Gaussian distribution with mean and variance given by the following:
            .. math::
                p(x(t)|x(0))=\mathcal{N}(x(t);x(0)e^{-\frac{1}{2}\int_{0}^{t}{\beta(s)\mathrm{d}s}},(1-e^{-\int_{0}^{t}{\beta(s)\mathrm{d}s}})I)
        Arguments:
            - t (:obj:`torch.Tensor`): The input time.
            - x (:obj:`Union[torch.Tensor, TensorDict]`): The input state at time 0.
            - condition (:obj:`Union[torch.Tensor, TensorDict]`): The input condition.
        Returns:
            - distribution (:obj:`Distribution`): The output distribution of the state at time t.
        """
        pass

    def direct_forward(
            self,
            t: torch.Tensor,
            x: Union[torch.Tensor, TensorDict],
            condition: Union[torch.Tensor, TensorDict] = None, 
        ) -> Distribution:
        pass

    def sample(
            self,
            t: torch.Tensor,
            x: Union[torch.Tensor, TensorDict],
            condition: Union[torch.Tensor, TensorDict] = None,
        ) -> Union[torch.Tensor, TensorDict]:
        """
        Overview:
            Return the sample of the state at time t given the initial state.
        Arguments:
            - t (:obj:`torch.Tensor`): The input time.
            - x (:obj:`Union[torch.Tensor, TensorDict]`): The input state at time 0.
            - condition (:obj:`Union[torch.Tensor, TensorDict]`): The input condition.
        Returns:
            - sample (:obj:`Union[torch.Tensor, TensorDict]`): The output sample.
        """
        return self.forward(x, t, condition).sample()

    def direct_sample(
            self,
            t: torch.Tensor,
            x: Union[torch.Tensor, TensorDict],
            condition: Union[torch.Tensor, TensorDict] = None,
        ) -> Union[torch.Tensor, TensorDict]:
        """
        Overview:
            Return the sample of the state at time t given the initial state by using gaussian conditional probability path of VP SDE.
        Arguments:
            - t (:obj:`torch.Tensor`): The input time.
            - x (:obj:`Union[torch.Tensor, TensorDict]`): The input state at time 0.
            - condition (:obj:`Union[torch.Tensor, TensorDict]`): The input condition.
        """

        return x * self.gaussian_conditional_probability_path.scale(t) + self.gaussian_conditional_probability_path.std(t) * torch.randn_like(x)

    def direct_sample_and_return_noise(
            self,
            t: torch.Tensor,
            x: Union[torch.Tensor, TensorDict],
            condition: Union[torch.Tensor, TensorDict] = None,
        ) -> Union[torch.Tensor, TensorDict]:
        """
        Overview:
            Return the sample of the state at time t given the initial state by using gaussian conditional probability path of VP SDE and the noise.
        Arguments:
            - t (:obj:`torch.Tensor`): The input time.
            - x (:obj:`Union[torch.Tensor, TensorDict]`): The input state at time 0.
            - condition (:obj:`Union[torch.Tensor, TensorDict]`): The input condition.
        Returns:
            - sample (:obj:`Union[torch.Tensor, TensorDict]`): The output sample.
            - noise (:obj:`Union[torch.Tensor, TensorDict]`): The output noise.
        """

        noise = torch.randn_like(x).to(x.device)

        return x * self.gaussian_conditional_probability_path.scale(t) + self.gaussian_conditional_probability_path.std(t) * noise, noise
