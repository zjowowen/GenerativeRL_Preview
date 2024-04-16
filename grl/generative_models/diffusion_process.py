from typing import Callable, Union

import torch
import torch.nn as nn
from easydict import EasyDict
from tensordict import TensorDict
from torch.distributions import Distribution

from grl.numerical_methods.ode import ODE
from grl.numerical_methods.probability_path import \
    GaussianConditionalProbabilityPath
from grl.numerical_methods.sde import SDE


class DiffusionProcess:
    """
    Overview:
        Common methods of diffusion process.
    """

    def __init__(
            self,
            path: GaussianConditionalProbabilityPath,
            t_max: float = 1.0
        ):
        """
        Overview:
            Initialize the diffusion process.
        Arguments:
            path (:obj:`GaussianConditionalProbabilityPath`): The Gaussian conditional probability path.
            t_max (:obj:`float`): The maximum time.
        """
        super().__init__()
        self.path = path
        self.t_max = t_max

    def drift(
            self,
            t: torch.Tensor,
            x: Union[torch.Tensor, TensorDict],
            condition: Union[torch.Tensor, TensorDict] = None,
        ) -> Union[torch.Tensor, TensorDict]:
        """
        Overview:
            Return the drift term of the diffusion process.
            The drift term is given by the following:

            .. math::
                f(x,t)

        Arguments:
            t (:obj:`torch.Tensor`): The input time.
            x (:obj:`Union[torch.Tensor, TensorDict]`): The input state.            
            condition (:obj:`Union[torch.Tensor, TensorDict]`): The input condition.
        Returns:
            drift (:obj:`Union[torch.Tensor, TensorDict]`): The output drift term.
        """

        if len(x.shape) > len(t.shape):
            return x * self.path.drift_coefficient(t)[(..., ) + (None, ) * (len(x.shape)-len(t.shape))].expand(x.shape)
        else:
            return x * self.path.drift_coefficient(t)
    
    def drift_coefficient(
            self,
            t: torch.Tensor,
            x: Union[torch.Tensor, TensorDict],
            condition: Union[torch.Tensor, TensorDict] = None,
        ) -> Union[torch.Tensor, TensorDict]:
        """
        Overview:
            Return the drift coefficient of the diffusion process.
            The drift coefficient is given by the following:

            .. math::
                f(t)

        Arguments:
            t (:obj:`torch.Tensor`): The input time.
            x (:obj:`Union[torch.Tensor, TensorDict]`): The input state.
            condition (:obj:`Union[torch.Tensor, TensorDict]`): The input condition.
        Returns:
            drift_coefficient (:obj:`Union[torch.Tensor, TensorDict]`): The output drift coefficient.
        """

        if x is not None and len(x.shape) > len(t.shape):
            return self.path.drift_coefficient(t)[(..., ) + (None, ) * (len(x.shape)-len(t.shape))].expand(x.shape)
        else:
            return self.path.drift_coefficient(t)

    def diffusion(
            self,
            t: torch.Tensor,
            x: Union[torch.Tensor, TensorDict],
            condition: Union[torch.Tensor, TensorDict] = None,
        ) -> Union[torch.Tensor, TensorDict]:
        """
        Overview:
            Return the diffusion term of the diffusion process.
            The diffusion term is given by the following:

            .. math::
                g(x,t)

        Arguments:
            t (:obj:`torch.Tensor`): The input time.
            x (:obj:`Union[torch.Tensor, TensorDict]`): The input state.
            condition (:obj:`Union[torch.Tensor, TensorDict]`): The input condition.
        Returns:
            diffusion (:obj:`Union[torch.Tensor, TensorDict]`): The output diffusion term.
        """

        if x is not None and len(x.shape) > len(t.shape):
            return self.path.diffusion(t)[(..., ) + (None, ) * (len(x.shape)-len(t.shape))].expand(x.shape)
        else:
            return self.path.diffusion(t)

    def diffusion_squared(
            self,
            t: torch.Tensor,
            x: Union[torch.Tensor, TensorDict],
            condition: Union[torch.Tensor, TensorDict] = None,
        ) -> Union[torch.Tensor, TensorDict]:
        """
        Overview:
            Return the square of the diffusion term of the diffusion process.
            The square of the diffusion term is given by the following:

            .. math::
                g^2(x,t)

        Arguments:
            t (:obj:`torch.Tensor`): The input time.
            x (:obj:`Union[torch.Tensor, TensorDict]`): The input state.
            condition (:obj:`Union[torch.Tensor, TensorDict]`): The input condition.
        Returns:
            diffusion_squared (:obj:`Union[torch.Tensor, TensorDict]`): The output square of the diffusion term.
        """

        if x is not None and len(x.shape) > len(t.shape):
            return self.path.diffusion_squared(t)[(..., ) + (None, ) * (len(x.shape)-len(t.shape))].expand(x.shape)
        else:
            return self.path.diffusion_squared(t)

    def scale(
            self,
            t: torch.Tensor,
            x: Union[torch.Tensor, TensorDict],
            condition: Union[torch.Tensor, TensorDict] = None,
        ) -> Union[torch.Tensor, TensorDict]:
        """
        Overview:
            Return the scale of the diffusion process.
            The scale is given by the following:

            .. math::
                s(t)

        Arguments:
            t (:obj:`torch.Tensor`): The input time.
            x (:obj:`Union[torch.Tensor, TensorDict]`): The input state.
            condition (:obj:`Union[torch.Tensor, TensorDict]`): The input condition.
        Returns:
            scale (:obj:`Union[torch.Tensor, TensorDict]`): The output scale.
        """

        if x is not None and len(x.shape) > len(t.shape):
            return self.path.scale(t)[(..., ) + (None, ) * (len(x.shape)-len(t.shape))].expand(x.shape)
        else:
            return self.path.scale(t)

    def log_scale(
            self,
            t: torch.Tensor,
            x: Union[torch.Tensor, TensorDict],
            condition: Union[torch.Tensor, TensorDict] = None,
        ) -> Union[torch.Tensor, TensorDict]:
        """
        Overview:
            Return the log scale of the diffusion process.
            The log scale is given by the following:

            .. math::
                \log(s(t))

        Arguments:
            t (:obj:`torch.Tensor`): The input time.
            x (:obj:`Union[torch.Tensor, TensorDict]`): The input state.
            condition (:obj:`Union[torch.Tensor, TensorDict]`): The input condition.
        Returns:
            log_scale (:obj:`Union[torch.Tensor, TensorDict]`): The output log scale.
        """

        if x is not None and len(x.shape) > len(t.shape):
            return self.path.log_scale(t)[(..., ) + (None, ) * (len(x.shape)-len(t.shape))].expand(x.shape)
        else:
            return self.path.log_scale(t)

    def std(
            self,
            t: torch.Tensor,
            x: Union[torch.Tensor, TensorDict],
            condition: Union[torch.Tensor, TensorDict] = None,
        ) -> Union[torch.Tensor, TensorDict]:
        """
        Overview:
            Return the standard deviation of the diffusion process.
            The standard deviation is given by the following:

            .. math::
                \sigma(t)

        Arguments:
            t (:obj:`torch.Tensor`): The input time.
            x (:obj:`Union[torch.Tensor, TensorDict]`): The input state.
            condition (:obj:`Union[torch.Tensor, TensorDict]`): The input condition.
        Returns:
            std (:obj:`Union[torch.Tensor, TensorDict]`): The output standard deviation.
        """

        if x is not None and len(x.shape) > len(t.shape):
            return self.path.std(t)[(..., ) + (None, ) * (len(x.shape)-len(t.shape))].expand(x.shape)
        else:
            return self.path.std(t)

    def covariance(
            self,
            t: torch.Tensor,
            x: Union[torch.Tensor, TensorDict],
            condition: Union[torch.Tensor, TensorDict] = None,
        ) -> Union[torch.Tensor, TensorDict]:
        """
        Overview:
            Return the covariance of the diffusion process.
            The covariance is given by the following:

            .. math::
                \Sigma(t)

        Arguments:
            t (:obj:`torch.Tensor`): The input time.
            x (:obj:`Union[torch.Tensor, TensorDict]`): The input state.
            condition (:obj:`Union[torch.Tensor, TensorDict]`): The input condition.
        Returns:
            covariance (:obj:`Union[torch.Tensor, TensorDict]`): The output covariance.
        """

        if x is not None and len(x.shape) > len(t.shape):
            return self.path.covariance(t)[(..., ) + (None, ) * (len(x.shape)-len(t.shape))].expand(x.shape)
        else:
            return self.path.covariance(t)

    def velocity(
            self,
            t: torch.Tensor,
            x: Union[torch.Tensor, TensorDict],
            condition: Union[torch.Tensor, TensorDict] = None,
            noise: Union[torch.Tensor, TensorDict] = None,
        ) -> Union[torch.Tensor, TensorDict]:
        """
        Overview:
            Return the velocity of the diffusion process.
            The velocity is given by the following:

            .. math::
                v(t,x):=\frac{\mathrm{d}(x_t|x_0)}{\mathrm{d}t}

        Arguments:
            t (:obj:`torch.Tensor`): The input time.
            x (:obj:`Union[torch.Tensor, TensorDict]`): The input state.
            condition (:obj:`Union[torch.Tensor, TensorDict]`): The input condition.
            noise (:obj:`Union[torch.Tensor, TensorDict]`): The input noise.
        """
        if noise is None:
            noise = torch.randn_like(x).to(x.device)
        if len(x.shape) > len(t.shape):
            d_scale_dt = self.path.d_scale_dt(t)[(..., ) + (None, ) * (len(x.shape)-len(t.shape))].expand(x.shape)
            d_std_dt = self.path.d_std_dt(t)[(..., ) + (None, ) * (len(x.shape)-len(t.shape))].expand(x.shape)
            return d_scale_dt * x + d_std_dt * noise
        else:
            return self.path.d_scale_dt(t) * x + self.path.d_std_dt(t) * noise

    def HalfLogSNR(
            self,
            t: torch.Tensor,
            x: Union[torch.Tensor, TensorDict] = None,
            condition: Union[torch.Tensor, TensorDict] = None,
        ) -> Union[torch.Tensor, TensorDict]:
        """
        Overview:
            Return the half log signal-to-noise ratio of the diffusion process.
            The half log signal-to-noise ratio is given by the following:

            .. math::
                \log(s(t))-\log(\sigma(t))

        Arguments:
            t (:obj:`torch.Tensor`): The input time.
            x (:obj:`Union[torch.Tensor, TensorDict]`): The input state.
            condition (:obj:`Union[torch.Tensor, TensorDict]`): The input condition.
        Returns:
            HalfLogSNR (:obj:`torch.Tensor`): The half-logSNR.
        """

        if x is not None and len(x.shape) > len(t.shape):
            return self.path.HalfLogSNR(t)[(..., ) + (None, ) * (len(x.shape)-len(t.shape))].expand(x.shape)
        else:
            return self.path.HalfLogSNR(t)
    
    def InverseHalfLogSNR(
            self,
            HalfLogSNR: torch.Tensor,
            x: Union[torch.Tensor, TensorDict] = None,
            condition: Union[torch.Tensor, TensorDict] = None,
        ) -> Union[torch.Tensor, TensorDict]:
        """
        Overview:
            Return the inverse function of half log signal-to-noise ratio of the diffusion process, \
            which is the time at which the half log signal-to-noise ratio is given.
        Arguments:
            HalfLogSNR (:obj:`torch.Tensor`): The input half-logSNR.
            x (:obj:`Union[torch.Tensor, TensorDict]`): The input state.
            condition (:obj:`Union[torch.Tensor, TensorDict]`): The input condition.
        Returns:
            t (:obj:`torch.Tensor`): The output time.
        """

        if x is not None:
            return self.path.InverseHalfLogSNR(HalfLogSNR).to(x.device)
        else:
            return self.path.InverseHalfLogSNR(HalfLogSNR)

    def sde(self, condition: Union[torch.Tensor, TensorDict] = None) -> SDE:
        """
        Overview:
            Return the SDE of diffusion process with the input condition.
        Arguments:
            condition (:obj:`Union[torch.Tensor, TensorDict]`): The input condition.
        Returns:
            sde (:obj:`SDE`): The SDE of diffusion process.
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

    def forward_reversed_sde(
            self,
            function: Union[Callable, nn.Module],
            function_type: str,
            forward_diffusion_function: Union[Callable, nn.Module] = None,
            forward_diffusion_squared_function: Union[Callable, nn.Module] = None,
            condition: Union[torch.Tensor, TensorDict] = None,
            T: torch.Tensor = torch.tensor(1.),
        ) -> SDE:
        """
        Overview:
            Return the forward of reversed time SDE of the diffusion process with the input condition.
        Arguments:
            function (:obj:`Union[Callable, nn.Module]`): The input function.
            function_type (:obj:`str`): The type of the function.
            reverse_diffusion_function (:obj:`Union[Callable, nn.Module]`): The input reverse diffusion function.
            condition (:obj:`Union[torch.Tensor, TensorDict]`): The input condition.
            T (:obj:`torch.Tensor`): The maximum time.
        Returns:
            reverse_sde (:obj:`SDE`): The reverse diffusion process.
        """

        #TODO: validate these functions

        if function_type == "score_function":

            def sde_drift(
                    t: torch.Tensor,
                    x: Union[torch.Tensor, TensorDict],
            ) -> Union[torch.Tensor, TensorDict]:
                return self.drift(t, x, condition) - 0.5 * (self.diffusion_squared(t, x, condition) - forward_diffusion_squared_function(t, x, condition)) * function(t, x, condition)
            
            def sde_diffusion(
                    t: torch.Tensor,
                    x: Union[torch.Tensor, TensorDict],
            ) -> Union[torch.Tensor, TensorDict]:
                return forward_diffusion_function(t, x, condition)
            
            return SDE(drift=sde_drift, diffusion=sde_diffusion)

        elif function_type == "noise_function":

            def sde_drift(
                    t: torch.Tensor,
                    x: Union[torch.Tensor, TensorDict],
            ) -> Union[torch.Tensor, TensorDict]:
                return self.drift(t, x, condition) + 0.5 * (self.diffusion_squared(t, x, condition) - forward_diffusion_squared_function(t, x, condition)) * function(t, x, condition) / self.std(t, x, condition)
            
            def sde_diffusion(
                    t: torch.Tensor,
                    x: Union[torch.Tensor, TensorDict],
            ) -> Union[torch.Tensor, TensorDict]:
                return forward_diffusion_function(t, x, condition)
            
            return SDE(drift=sde_drift, diffusion=sde_diffusion)

        elif function_type == "velocity_function":

            def sde_drift(
                    t: torch.Tensor,
                    x: Union[torch.Tensor, TensorDict],
            ) -> Union[torch.Tensor, TensorDict]:
                v = function(t, x, condition)
                r = forward_diffusion_squared_function(t, x, condition) / (self.diffusion_squared(t, x, condition) + 1e-8)
                return v - (v - self.drift(t, x, condition)) * r
            
            def sde_diffusion(
                    t: torch.Tensor,
                    x: Union[torch.Tensor, TensorDict],
            ) -> Union[torch.Tensor, TensorDict]:
                return forward_diffusion_function(t, x, condition)
            
            return SDE(drift=sde_drift, diffusion=sde_diffusion)

        elif function_type == "data_prediction_function":

            def sde_drift(
                    t: torch.Tensor,
                    x: Union[torch.Tensor, TensorDict],
            ) -> Union[torch.Tensor, TensorDict]:
                D = 0.5 * (self.diffusion_squared(t, x, condition) - forward_diffusion_squared_function(t, x, condition)) / self.covariance(t, x, condition)
                return (self.drift_coefficient(t, x) + D) * x - self.scale(t, x, condition) * D * function(t, x, condition)
            
            def sde_diffusion(
                    t: torch.Tensor,
                    x: Union[torch.Tensor, TensorDict],
            ) -> Union[torch.Tensor, TensorDict]:
                return forward_diffusion_function(t, x, condition)
            
            return SDE(drift=sde_drift, diffusion=sde_diffusion)

        else:
            raise NotImplementedError("Unknown type of function {}".format(function_type))

    def forward_reversed_ode(
            self,
            function: Union[Callable, nn.Module],
            function_type: str,
            condition: Union[torch.Tensor, TensorDict] = None,
            T: torch.Tensor = torch.tensor(1.),
        ) -> ODE:
        """
        Overview:
            Return the forward of reversed time ODE of the diffusion process with the input condition.
        """

        #TODO: validate these functions

        if function_type == "score_function":

            def ode_drift(
                    t: torch.Tensor,
                    x: Union[torch.Tensor, TensorDict],
            ) -> Union[torch.Tensor, TensorDict]:
                return self.drift(t, x, condition) - 0.5 * self.diffusion_squared(t, x, condition) * function(t, x, condition)
            
            return ODE(drift=ode_drift)

        elif function_type == "noise_function":

            def ode_drift(
                    t: torch.Tensor,
                    x: Union[torch.Tensor, TensorDict],
            ) -> Union[torch.Tensor, TensorDict]:
                return self.drift(t, x, condition) + 0.5 * self.diffusion_squared(t, x, condition) * function(t, x, condition) / self.std(t, x, condition)
            
            return ODE(drift=ode_drift)

        elif function_type == "velocity_function":

            def ode_drift(
                    t: torch.Tensor,
                    x: Union[torch.Tensor, TensorDict],
            ) -> Union[torch.Tensor, TensorDict]:
                return function(t, x, condition)
            
            return ODE(drift=ode_drift)

        elif function_type == "data_prediction_function":

            def ode_drift(
                    t: torch.Tensor,
                    x: Union[torch.Tensor, TensorDict],
            ) -> Union[torch.Tensor, TensorDict]:
                D = 0.5 * self.diffusion_squared(t, x, condition) / self.covariance(t, x, condition)
                return (self.drift_coefficient(t, x) + D) * x - self.scale(t, x, condition) * D * function(t, x, condition)
            
            return ODE(drift=ode_drift)
        
        else:
            raise NotImplementedError("Unknown type of function {}".format(function_type))

    def reverse_sde(
            self,
            function: Union[Callable, nn.Module],
            function_type: str,
            reverse_diffusion_function: Union[Callable, nn.Module] = None,
            reverse_diffusion_squared_function: Union[Callable, nn.Module] = None,
            condition: Union[torch.Tensor, TensorDict] = None,
            T: torch.Tensor = torch.tensor(1.),
        ) -> SDE:
        """
        Overview:
            Return the reversed time SDE of the diffusion process with the input condition.
        Arguments:
            function (:obj:`Union[Callable, nn.Module]`): The input function.
            function_type (:obj:`str`): The type of the function.
            reverse_diffusion_function (:obj:`Union[Callable, nn.Module]`): The input reverse diffusion function.
            condition (:obj:`Union[torch.Tensor, TensorDict]`): The input condition.
            T (:obj:`torch.Tensor`): The maximum time.
        Returns:
            reverse_sde (:obj:`SDE`): The reverse diffusion process.
        """

        if function_type == "score_function":

            def reverse_sde_drift(
                    t: torch.Tensor,
                    x: Union[torch.Tensor, TensorDict],
            ) -> Union[torch.Tensor, TensorDict]:
                return - self.drift(T - t, x, condition) + 0.5 * (self.diffusion_squared(T - t, x, condition) + reverse_diffusion_squared_function(T - t, x, condition)) * function(T - t, x, condition)
            
            def reverse_sde_diffusion(
                    t: torch.Tensor,
                    x: Union[torch.Tensor, TensorDict],
            ) -> Union[torch.Tensor, TensorDict]:
                return reverse_diffusion_function(T - t, x, condition)
            
            return SDE(drift=reverse_sde_drift, diffusion=reverse_sde_diffusion)

        elif function_type == "noise_function":

            def reverse_sde_drift(
                    t: torch.Tensor,
                    x: Union[torch.Tensor, TensorDict],
            ) -> Union[torch.Tensor, TensorDict]:
                return - self.drift(T - t, x, condition) - 0.5 * (self.diffusion_squared(T - t, x, condition) + reverse_diffusion_squared_function(T - t, x, condition)) * function(T - t, x, condition) / self.std(T - t, x, condition)
            
            def reverse_sde_diffusion(
                    t: torch.Tensor,
                    x: Union[torch.Tensor, TensorDict],
            ) -> Union[torch.Tensor, TensorDict]:
                return reverse_diffusion_function(T - t, x, condition)
            
            return SDE(drift=reverse_sde_drift, diffusion=reverse_sde_diffusion)

        elif function_type == "velocity_function":

            def reverse_sde_drift(
                    t: torch.Tensor,
                    x: Union[torch.Tensor, TensorDict],
            ) -> Union[torch.Tensor, TensorDict]:
                v = function(T - t, x, condition)
                r = reverse_diffusion_squared_function(T - t, x, condition) / (self.diffusion_squared(T - t, x, condition) + 1e-8)
                return - v - (v - self.drift(T - t, x, condition)) * r
            
            def reverse_sde_diffusion(
                    t: torch.Tensor,
                    x: Union[torch.Tensor, TensorDict],
            ) -> Union[torch.Tensor, TensorDict]:
                return reverse_diffusion_function(T - t, x, condition)
            
            return SDE(drift=reverse_sde_drift, diffusion=reverse_sde_diffusion)

        elif function_type == "data_prediction_function":

            def reverse_sde_drift(
                    t: torch.Tensor,
                    x: Union[torch.Tensor, TensorDict],
            ) -> Union[torch.Tensor, TensorDict]:
                D = 0.5 * (self.diffusion_squared(T - t, x, condition) + reverse_diffusion_squared_function(T - t, x, condition)) / self.covariance(T - t, x, condition)
                return - (self.drift_coefficient(T - t, x) + D) * x + self.scale(T - t, x, condition) * D * function(T - t, x, condition)
            
            def reverse_sde_diffusion(
                    t: torch.Tensor,
                    x: Union[torch.Tensor, TensorDict],
            ) -> Union[torch.Tensor, TensorDict]:
                return reverse_diffusion_function(T - t, x, condition)
            
            return SDE(drift=reverse_sde_drift, diffusion=reverse_sde_diffusion)

        else:
            raise NotImplementedError("Unknown type of function {}".format(function_type))

    def reverse_ode(
            self,
            function: Union[Callable, nn.Module],
            function_type: str,
            condition: Union[torch.Tensor, TensorDict] = None,
            T: torch.Tensor = torch.tensor(1.),
        ) -> ODE:
        """
        Overview:
            Return the reversed time ODE of the diffusion process with the input condition.
        """
    
        if function_type == "score_function":

            def reverse_ode_drift(
                    t: torch.Tensor,
                    x: Union[torch.Tensor, TensorDict],
            ) -> Union[torch.Tensor, TensorDict]:
                return - self.drift(T - t, x, condition) + 0.5 * self.diffusion_squared(T - t, x, condition) * function(T - t, x, condition)
            
            return ODE(drift=reverse_ode_drift)

        elif function_type == "noise_function":

            def reverse_ode_drift(
                    t: torch.Tensor,
                    x: Union[torch.Tensor, TensorDict],
            ) -> Union[torch.Tensor, TensorDict]:
                return - self.drift(T - t, x, condition) - 0.5 * self.diffusion_squared(T - t, x, condition) * function(T - t, x, condition) / self.std(T - t, x, condition)
            
            return ODE(drift=reverse_ode_drift)

        elif function_type == "velocity_function":

            def reverse_ode_drift(
                    t: torch.Tensor,
                    x: Union[torch.Tensor, TensorDict],
            ) -> Union[torch.Tensor, TensorDict]:
                return - function(T - t, x, condition)
            
            return ODE(drift=reverse_ode_drift)

        elif function_type == "data_prediction_function":

            def reverse_ode_drift(
                    t: torch.Tensor,
                    x: Union[torch.Tensor, TensorDict],
            ) -> Union[torch.Tensor, TensorDict]:
                D = 0.5 * self.diffusion_squared(T - t, x, condition) / self.covariance(T - t, x, condition)
                return - (self.drift_coefficient(T - t, x) + D) * x + self.scale(T - t, x, condition) * D * function(T - t, x, condition)
            
            return ODE(drift=reverse_ode_drift)
        
        else:
            raise NotImplementedError("Unknown type of function {}".format(function_type))

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
            t (:obj:`torch.Tensor`): The input time.
            x (:obj:`Union[torch.Tensor, TensorDict]`): The input state at time 0.
            condition (:obj:`Union[torch.Tensor, TensorDict]`): The input condition.
        Returns:
            sample (:obj:`Union[torch.Tensor, TensorDict]`): The output sample.
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
            Return the sample of the state at time t given the initial state by using gaussian conditional probability path of diffusion process.
        Arguments:
            t (:obj:`torch.Tensor`): The input time.
            x (:obj:`Union[torch.Tensor, TensorDict]`): The input state at time 0.
            condition (:obj:`Union[torch.Tensor, TensorDict]`): The input condition.
        """

        #TODO: make it compatible with TensorDict

        return self.scale(t, x) * x + self.std(t, x) * torch.randn_like(x).to(x.device)

    def direct_sample_and_return_noise(
            self,
            t: torch.Tensor,
            x: Union[torch.Tensor, TensorDict],
            condition: Union[torch.Tensor, TensorDict] = None,
        ) -> Union[torch.Tensor, TensorDict]:
        """
        Overview:
            Return the sample of the state at time t given the initial state by using gaussian conditional probability path of diffusion process and the noise.
        Arguments:
            t (:obj:`torch.Tensor`): The input time.
            x (:obj:`Union[torch.Tensor, TensorDict]`): The input state at time 0.
            condition (:obj:`Union[torch.Tensor, TensorDict]`): The input condition.
        Returns:
            sample (:obj:`Union[torch.Tensor, TensorDict]`): The output sample.
            noise (:obj:`Union[torch.Tensor, TensorDict]`): The output noise.
        """

        noise = torch.randn_like(x).to(x.device)

        return self.scale(t, x) * x + self.std(t, x) * noise, noise
