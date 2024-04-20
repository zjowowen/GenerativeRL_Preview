from typing import Callable, Union

import torch
import torch.nn as nn
from easydict import EasyDict
from tensordict import TensorDict

from grl.generative_models.diffusion_process import DiffusionProcess


class VelocityFunction:
    """
    Overview:
        Velocity function Class.
    Interfaces:
        ``__init__``, ``forward``, ``flow_matching_loss``
    """

    def __init__(
            self,
            model_type: str,
            process: DiffusionProcess,
        ):
        """
        Overview:
            Initialize the velocity function.
        Arguments:
            - model_type (:obj:`str`): The type of the model.
            - process (:obj:`DiffusionProcess`): The process.
        """
        self.model_type = model_type
        self.process = process

    def forward(
            self,
            model: Union[Callable, nn.Module],
            t: torch.Tensor,
            x: Union[torch.Tensor, TensorDict],
            condition: Union[torch.Tensor, TensorDict] = None,
        ) -> torch.Tensor:
        """
        Overview:
            Return velocity of the model at time t given the initial state.
            .. math::
                v_{\theta}(t, x)
        Arguments:
            - model (:obj:`Union[Callable, nn.Module]`): The model.
            - t (:obj:`torch.Tensor`): The input time.
            - x (:obj:`Union[torch.Tensor, TensorDict]`): The input state at time t.
            - condition (:obj:`Union[torch.Tensor, TensorDict]`): The input condition.
        """

        if self.model_type == "noise_function":
            #TODO: check if this is correct
            return self.process.drift(t, x) + 0.5 * self.process.diffusion_squared(t, x) / self.process.std(t, x) * model(t, x, condition)
        elif self.model_type == "score_function":
            #TODO: check if this is correct
            return self.process.drift(t, x) - 0.5 * self.process.diffusion_squared(t, x) * model(t, x, condition)
        elif self.model_type == "velocity_function":
            return model(t, x, condition)
        elif self.model_type == "data_prediction_function":
            #TODO: check if this is correct
            D = 0.5 * self.process.diffusion_squared(t, x) / self.process.covariance(t, x)
            return (self.process.drift_coefficient(t) + D) - D * self.process.scale(t) * model(t, x, condition)
        else:
            raise NotImplementedError("Unknown type of Velocity Function {}".format(type))


    def flow_matching_loss(
            self,
            model: Union[Callable, nn.Module],
            x: Union[torch.Tensor, TensorDict],
            condition: Union[torch.Tensor, TensorDict] = None,
        ) -> torch.Tensor:

        #TODO: make it compatible with TensorDict
        if self.model_type == "noise_function":
            eps = 1e-5
            t_random = torch.rand(x.shape[0], device=x.device) * (self.process.t_max - eps) + eps
            noise = torch.randn_like(x).to(x.device)
            std = self.process.std(t_random, x)
            x_t = self.process.scale(t_random, x) * x + std * noise
            velocity_value = self.process.drift(t_random, x_t) + 0.5 * self.process.diffusion_squared(t_random, x_t) * model(t_random, x_t, condition=condition) / std
            velocity = self.process.velocity(t_random, x, noise=noise)
            loss = torch.mean(torch.sum((velocity_value - velocity) ** 2, dim=(1, )))
            return loss
        elif self.model_type == "score_function":
            eps = 1e-5
            t_random = torch.rand(x.shape[0], device=x.device) * (self.process.t_max - eps) + eps
            noise = torch.randn_like(x).to(x.device)
            std = self.process.std(t_random, x)
            x_t = self.process.scale(t_random, x) * x + std * noise
            velocity_value = self.process.drift(t_random, x_t) - 0.5 * self.process.diffusion_squared(t_random, x_t) * model(t_random, x_t, condition=condition)
            velocity = self.process.velocity(t_random, x, noise=noise)
            loss = torch.mean(torch.sum((velocity_value - velocity) ** 2, dim=(1, )))
            return loss
        elif self.model_type == "velocity_function":
            eps = 1e-5
            t_random = torch.rand(x.shape[0], device=x.device) * (self.process.t_max - eps) + eps
            noise = torch.randn_like(x).to(x.device)
            std = self.process.std(t_random, x)
            x_t = self.process.scale(t_random, x) * x + std * noise
            velocity_value = model(t_random, x_t, condition=condition)
            velocity = self.process.velocity(t_random, x, noise=noise)
            loss = torch.mean(torch.sum((velocity_value - velocity) ** 2, dim=(1, )))
            return loss
        elif self.model_type == "data_prediction_function":
            #TODO: check if this is correct
            eps = 1e-5
            t_random = torch.rand(x.shape[0], device=x.device) * (self.process.t_max - eps) + eps
            noise = torch.randn_like(x).to(x.device)
            std = self.process.std(t_random, x)
            x_t = self.process.scale(t_random, x) * x + std * noise
            D = 0.5 * self.process.diffusion_squared(t_random, x) / self.process.covariance(t_random, x)
            velocity_value = (self.process.drift_coefficient(t_random, x_t) + D) * x_t - D * self.process.scale(t_random, x_t) * model(t_random, x_t, condition=condition)
            velocity = self.process.velocity(t_random, x, noise=noise)
            loss = torch.mean(torch.sum((velocity_value - velocity) ** 2, dim=(1, )))
            return loss
        else:
            raise NotImplementedError("Unknown type of velocity function {}".format(type))


    def flow_matching_loss_2(
            self,
            model: Union[Callable, nn.Module],
            x: Union[torch.Tensor, TensorDict],
            condition: Union[torch.Tensor, TensorDict] = None,
        ) -> torch.Tensor:

        #TODO: make it compatible with TensorDict
        if self.model_type == "noise_function":
            eps = 1e-5
            t_random = torch.rand(x.shape[0], device=x.device) * (self.process.t_max - eps) + eps
            noise = torch.randn_like(x).to(x.device)
            std = self.process.std(t_random, x)
            x_t = self.process.scale(t_random, x) * x + std * noise
            velocity_value = self.process.drift(t_random, x_t) + 0.5 * self.process.diffusion_squared(t_random, x_t) * model(t_random, x_t, condition=condition) / std
            velocity = self.process.velocity(t_random, x, noise=noise)
            loss = torch.mean(torch.sum((velocity_value * torch.abs(velocity_value) - velocity * torch.abs(velocity)) ** 2, dim=(1, )))
            return loss
        elif self.model_type == "score_function":
            eps = 1e-5
            t_random = torch.rand(x.shape[0], device=x.device) * (self.process.t_max - eps) + eps
            noise = torch.randn_like(x).to(x.device)
            std = self.process.std(t_random, x)
            x_t = self.process.scale(t_random, x) * x + std * noise
            velocity_value = self.process.drift(t_random, x_t) - 0.5 * self.process.diffusion_squared(t_random, x_t) * model(t_random, x_t, condition=condition)
            velocity = self.process.velocity(t_random, x, noise=noise)
            loss = torch.mean(torch.sum((velocity_value * torch.abs(velocity_value) - velocity * torch.abs(velocity)) ** 2, dim=(1, )))
            return loss
        elif self.model_type == "velocity_function":
            eps = 1e-5
            t_random = torch.rand(x.shape[0], device=x.device) * (self.process.t_max - eps) + eps
            noise = torch.randn_like(x).to(x.device)
            std = self.process.std(t_random, x)
            x_t = self.process.scale(t_random, x) * x + std * noise
            velocity_value = model(t_random, x_t, condition=condition)
            velocity = self.process.velocity(t_random, x, noise=noise)
            loss = torch.mean(torch.sum((velocity_value * torch.abs(velocity_value) - velocity * torch.abs(velocity)) ** 2, dim=(1, )))
            return loss
        elif self.model_type == "data_prediction_function":
            #TODO: check if this is correct
            eps = 1e-5
            t_random = torch.rand(x.shape[0], device=x.device) * (self.process.t_max - eps) + eps
            noise = torch.randn_like(x).to(x.device)
            std = self.process.std(t_random, x)
            x_t = self.process.scale(t_random, x) * x + std * noise
            D = 0.5 * self.process.diffusion_squared(t_random, x) / self.process.covariance(t_random, x)
            velocity_value = (self.process.drift_coefficient(t_random, x_t) + D) * x_t - D * self.process.scale(t_random, x_t) * model(t_random, x_t, condition=condition)
            velocity = self.process.velocity(t_random, x, noise=noise)
            loss = torch.mean(torch.sum((velocity_value * torch.abs(velocity_value) - velocity * torch.abs(velocity)) ** 2, dim=(1, )))
            return loss
        else:
            raise NotImplementedError("Unknown type of velocity function {}".format(type))

    def flow_matching_loss_hybrid(
            self,
            model: Union[Callable, nn.Module],
            x: Union[torch.Tensor, TensorDict],
            condition: Union[torch.Tensor, TensorDict] = None,
        ) -> torch.Tensor:

        #TODO: make it compatible with TensorDict
        if self.model_type == "noise_function":
            eps = 1e-5
            t_random = torch.rand(x.shape[0], device=x.device) * (self.process.t_max - eps) + eps
            noise = torch.randn_like(x).to(x.device)
            std = self.process.std(t_random, x)
            x_t = self.process.scale(t_random, x) * x + std * noise
            velocity_value = self.process.drift(t_random, x_t) + 0.5 * self.process.diffusion_squared(t_random, x_t) * model(t_random, x_t, condition=condition) / std
            velocity = self.process.velocity(t_random, x, noise=noise)
            loss_1 = torch.mean(torch.sum((velocity_value - velocity) ** 2, dim=(1, )))
            loss_2 = torch.mean(torch.sum((velocity_value * torch.abs(velocity_value) - velocity * torch.abs(velocity)) ** 2, dim=(1, )))
            return loss_1 + loss_2 / 4.0
        elif self.model_type == "score_function":
            eps = 1e-5
            t_random = torch.rand(x.shape[0], device=x.device) * (self.process.t_max - eps) + eps
            noise = torch.randn_like(x).to(x.device)
            std = self.process.std(t_random, x)
            x_t = self.process.scale(t_random, x) * x + std * noise
            velocity_value = self.process.drift(t_random, x_t) - 0.5 * self.process.diffusion_squared(t_random, x_t) * model(t_random, x_t, condition=condition)
            velocity = self.process.velocity(t_random, x, noise=noise)
            loss_1 = torch.mean(torch.sum((velocity_value - velocity) ** 2, dim=(1, )))
            loss_2 = torch.mean(torch.sum((velocity_value * torch.abs(velocity_value) - velocity * torch.abs(velocity)) ** 2, dim=(1, )))
            return loss_1 + loss_2 / 4.0
        elif self.model_type == "velocity_function":
            eps = 1e-5
            t_random = torch.rand(x.shape[0], device=x.device) * (self.process.t_max - eps) + eps
            noise = torch.randn_like(x).to(x.device)
            std = self.process.std(t_random, x)
            x_t = self.process.scale(t_random, x) * x + std * noise
            velocity_value = model(t_random, x_t, condition=condition)
            velocity = self.process.velocity(t_random, x, noise=noise)
            loss_1 = torch.mean(torch.sum((velocity_value - velocity) ** 2, dim=(1, )))
            loss_2 = torch.mean(torch.sum((velocity_value * torch.abs(velocity_value) - velocity * torch.abs(velocity)) ** 2, dim=(1, )))
            return loss_1 + loss_2 / 4.0
        elif self.model_type == "data_prediction_function":
            #TODO: check if this is correct
            eps = 1e-5
            t_random = torch.rand(x.shape[0], device=x.device) * (self.process.t_max - eps) + eps
            noise = torch.randn_like(x).to(x.device)
            std = self.process.std(t_random, x)
            x_t = self.process.scale(t_random, x) * x + std * noise
            D = 0.5 * self.process.diffusion_squared(t_random, x) / self.process.covariance(t_random, x)
            velocity_value = (self.process.drift_coefficient(t_random, x_t) + D) * x_t - D * self.process.scale(t_random, x_t) * model(t_random, x_t, condition=condition)
            velocity = self.process.velocity(t_random, x, noise=noise)
            loss_1 = torch.mean(torch.sum((velocity_value - velocity) ** 2, dim=(1, )))
            loss_2 = torch.mean(torch.sum((velocity_value * torch.abs(velocity_value) - velocity * torch.abs(velocity)) ** 2, dim=(1, )))
            return loss_1 + loss_2 / 4.0
        else:
            raise NotImplementedError("Unknown type of velocity function {}".format(type))
