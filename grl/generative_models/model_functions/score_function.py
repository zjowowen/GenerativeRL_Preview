from typing import Callable, Union

import torch
import torch.nn as nn
import treetensor
from easydict import EasyDict
from tensordict import TensorDict


class ScoreFunction:
    """
    Overview:
        Model of Score function in Score-based generative model.
    Interfaces:
        ``__init__``, ``forward``, ``score_matching_loss``
    """

    def __init__(
            self,
            model_type: str,
            process: object,
        ):
        """
        Overview:
            Initialize the score function.
        Arguments:
            - model_type (:obj:`str`): The type of the model.
            - process (:obj:`object`): The process.
        """

        self.model_type = model_type
        self.process = process
        #TODO: add more types
        assert self.model_type in ["data_prediction_function", "noise_function", "score_function", "velocity_function", "denoiser_function"], \
            "Unknown type of ScoreFunction {}".format(type)

    def forward(
            self,
            model: Union[Callable, nn.Module],
            t: torch.Tensor,
            x: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
            condition: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor] = None,
        ) -> torch.Tensor:
        """
        Overview:
            Return score function of the model at time t given the initial state, which is the gradient of the log-likelihood.
            .. math::
                \nabla_{x_t} \log p_{\theta}(x_t)

        Arguments:
            - model (:obj:`Union[Callable, nn.Module]`): The model.
            - t (:obj:`torch.Tensor`): The input time.
            - x (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`): The input state.
            - condition (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`): The input condition.
        """

        if self.model_type == "noise_function":
            return - model(t, x, condition) / self.process.std(t, x)
        elif self.model_type == "score_function":
            return model(t, x, condition)
        elif self.model_type == "velocity_function":
            #TODO: check if is correct
            return - (model(t, x, condition) - self.process.drift(t, x)) * 2.0 / self.process.diffusion_squared(t, x)
        elif self.model_type == "data_prediction_function":
            #TODO: check if is correct
            return - (x - self.process.scale(t, x) * model(t, x, condition)) / self.process.covariance(t, x)
        else:
            raise NotImplementedError("Unknown type of ScoreFunction {}".format(type))

    def score_matching_loss(
            self,
            model: Union[Callable, nn.Module],
            x: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
            condition: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor] = None,
            gaussian_generator: Callable = None,
        ) -> torch.Tensor:
        """
        Overview:
            Return the score matching loss function of the model given the initial state and the condition.
        Arguments:
            - x (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`): The input state.
            - condition (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`): The input condition.
        """

        def get_batch_size_and_device(x):
            if isinstance(x, torch.Tensor):
                return x.shape[0], x.device
            elif isinstance(x, TensorDict):
                return x.shape, x.device
            elif isinstance(x, treetensor.torch.Tensor):
                return list(x.values())[0].shape[0], list(x.values())[0].device
            else:
                raise NotImplementedError("Unknown type of x {}".format(type))

        def get_loss(noise_value, noise):
            if isinstance(noise_value, torch.Tensor):
                return torch.mean(torch.sum((noise_value - noise) ** 2, dim=(1, )))
            elif isinstance(noise_value, TensorDict):
                raise NotImplementedError("Not implemented yet")
            elif isinstance(noise_value, treetensor.torch.Tensor):
                return treetensor.torch.mean(treetensor.torch.sum((noise_value - noise) * (noise_value - noise), dim=(1, )))
            else:
                raise NotImplementedError("Unknown type of noise_value {}".format(type))

        #TODO: make it compatible with TensorDict
        if self.model_type == "noise_function":
            #TODO: test esp
            eps = 1e-5
            batch_size, device = get_batch_size_and_device(x)
            t_random = torch.rand(batch_size, device=device) * (self.process.t_max - eps) + eps
            if gaussian_generator is None:
                noise = torch.randn_like(x).to(device)
            else:
                noise = gaussian_generator(batch_size)
            x_t = self.process.scale(t_random, x) * x + self.process.std(t_random, x) * noise
            noise_value = model(t_random, x_t, condition=condition)
            loss = get_loss(noise_value, noise)
            return loss
        elif self.model_type == "score_function":
            #TODO: test esp
            eps = 1e-5
            batch_size, device = get_batch_size_and_device(x)
            t_random = torch.rand(batch_size, device=device) * (self.process.t_max - eps) + eps
            if gaussian_generator is None:
                noise = torch.randn_like(x).to(device)
            else:
                noise = gaussian_generator(batch_size)
            std = self.process.std(t_random, x)
            x_t = self.process.scale(t_random, x) * x + std * noise
            score_value = model(t_random, x_t, condition=condition)
            loss = get_loss(score_value * std, noise)
            return loss
        elif self.model_type == "velocity_function":
            #TODO: test esp
            eps = 1e-5
            batch_size, device = get_batch_size_and_device(x)
            t_random = torch.rand(batch_size, device=device) * (self.process.t_max - eps) + eps
            if gaussian_generator is None:
                noise = torch.randn_like(x).to(device)
            else:
                noise = gaussian_generator(batch_size)
            std = self.process.std(t_random, x)
            x_t = self.process.scale(t_random, x) * x + std * noise
            velocity_value = model(t_random, x_t, condition=condition)
            noise_value = (velocity_value - self.process.drift(t_random, x_t)) * 2.0 * std / self.process.diffusion_squared(t_random, x_t)
            loss = get_loss(noise_value, noise)
            return loss
        elif self.model_type == "data_prediction_function":
            #TODO: test esp
            eps = 1e-5
            batch_size, device = get_batch_size_and_device(x)
            t_random = torch.rand(batch_size, device=device) * (self.process.t_max - eps) + eps
            if gaussian_generator is None:
                noise = torch.randn_like(x).to(device)
            else:
                noise = gaussian_generator(batch_size)
            std = self.process.std(t_random, x)
            scale = self.process.scale(t_random, x)
            x_t = scale * x + std * noise
            data_predicted = model(t_random, x_t, condition=condition)
            noise_value = (x_t - scale * data_predicted) / std
            loss = get_loss(noise_value, noise)
            return loss
        else:
            raise NotImplementedError("Unknown type of score function {}".format(type))
