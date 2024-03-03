from typing import Union
from easydict import EasyDict
import torch
import torch.nn as nn
from tensordict import TensorDict
from generative_rl.machine_learning.encoders import get_encoder
from generative_rl.machine_learning.modules import get_module
from generative_rl.numerical_methods.probability_path import GaussianConditionalProbabilityPath


class ScoreFunctionModel(nn.Module):
    """
    Overview:
        Model of Score function in Score-based generative model.
    Interfaces:
        ``__init__``, ``forward``
    """

    def __init__(self, config: EasyDict):
        #TODO
        
        super().__init__()

        self.config = config
        assert hasattr(config, "backbone"), "backbone must be specified in config"

        self.model = torch.nn.ModuleDict()
        if hasattr(config, "t_encoder"):
            self.model["t_encoder"] = get_encoder(config.t_encoder.type)(**config.t_encoder.args)
        else:
            self.model["t_encoder"] = torch.nn.Identity()
        if hasattr(config, "x_encoder"):
            self.model["x_encoder"] = get_encoder(config.x_encoder.type)(**config.x_encoder.args)
        else:
            self.model["x_encoder"] = torch.nn.Identity()
        if hasattr(config, "condition_encoder"):
            self.model["condition_encoder"] = get_encoder(config.condition_encoder.type)(**config.condition_encoder.args)
        else:
            self.model["condition_encoder"] = torch.nn.Identity()
        
        #TODO
        # specific backbone network
        self.model["backbone"] = get_module(config.backbone.type)(**config.backbone.args)

    def forward(
            self,
            t: torch.Tensor,
            x: Union[torch.Tensor, TensorDict],
            condition: Union[torch.Tensor, TensorDict] = None,
        ) -> torch.Tensor:
        """
        Overview:
            Return the output of the model at time t given the initial state.
        """

        if condition is not None:
            t = self.model["t_encoder"](t)
            x = self.model["x_encoder"](x)
            condition = self.model["condition_encoder"](condition)
            output = self.model["backbone"](t, x, condition)
        else:
            t = self.model["t_encoder"](t)
            x = self.model["x_encoder"](x)
            output = self.model["backbone"](t, x)

        return output
    

class ScoreFunction(nn.Module):
    """
    Overview:
        Model of Score function in Score-based generative model.
    Interfaces:
        ``__init__``, ``forward``
    """

    def __init__(
            self,
            config: EasyDict,
            gaussian_conditional_probability_path: GaussianConditionalProbabilityPath
        ):
        """
        Overview:
            Initialize the ScoreFunction.
        Arguments:
            - config (:obj:`EasyDict`): The configuration of the ScoreFunction.
            - gaussian_conditional_probability_path (:obj:`GaussianConditionalProbabilityPath`): The Gaussian conditional probability path.
        """
        
        super().__init__()

        self.config = config
        self.type = config.type
        self.gaussian_conditional_probability_path = gaussian_conditional_probability_path
        #TODO: add more types
        assert self.type in ["noise_function", "score_function", "denoiser_function"], \
            "Unknown type of ScoreFunction {}".format(type)

        self.model = ScoreFunctionModel(config.model)


    def forward(
            self,
            t: torch.Tensor,
            x: Union[torch.Tensor, TensorDict],
            condition: Union[torch.Tensor, TensorDict] = None,
        ) -> torch.Tensor:
        """
        Overview:
            Return score function of the model at time t given the initial state, which is the gradient of the log-likelihood.
            .. math::
                \nabla_{x_t} \log p_{\theta}(x_t)

        Arguments:
            - t (:obj:`torch.Tensor`): The input time.
            - x (:obj:`Union[torch.Tensor, TensorDict]`): The input state.
            - condition (:obj:`Union[torch.Tensor, TensorDict]`): The input condition.
        """

        if self.type == "noise_function":
            return - self.model(t, x, condition) / self.gaussian_conditional_probability_path.std(t)[..., None]
        elif self.type == "score_function":
            return self.model(t, x, condition)
        else:
            raise NotImplementedError("Unknown type of ScoreFunction {}".format(type))

    def score_matching_loss(
            self,
            x: Union[torch.Tensor, TensorDict],
            condition: Union[torch.Tensor, TensorDict] = None,
        ) -> torch.Tensor:
        """
        Overview:
            Return the score matching loss function of the model given the initial state and the condition.
        Arguments:
            - x (:obj:`Union[torch.Tensor, TensorDict]`): The input state.
            - condition (:obj:`Union[torch.Tensor, TensorDict]`): The input condition.
        """

        #TODO: make it compatible with TensorDict
        if self.type == "noise_function":
            #TODO: test esp
            eps = 1e-3
            t_random = torch.rand(x.shape[0], device=x.device) * (self.gaussian_conditional_probability_path.t_max - eps) + eps
            noise = torch.randn_like(x).to(x.device)
            x_t = torch.einsum('i,i...->i...', self.gaussian_conditional_probability_path.scale(t_random), x) + torch.einsum('i,i...->i...', self.gaussian_conditional_probability_path.std(t_random), noise)
            noise_value = self.model(t_random, x_t, condition=condition)
            loss = torch.mean(torch.sum((noise_value - noise) ** 2, dim=(1, )))
            return loss
        elif self.type == "score_function":
            #TODO: test esp
            eps = 1e-3
            t_random = torch.rand(x.shape[0], device=x.device) * (self.gaussian_conditional_probability_path.t_max - eps) + eps
            noise = torch.randn_like(x).to(x.device)
            std = self.gaussian_conditional_probability_path.std(t_random)
            x_t = torch.einsum('i,i...->i...', self.gaussian_conditional_probability_path.scale(t_random), x) + torch.einsum('i,i...->i...', std, noise)
            score_value = self.model(t_random, x_t, condition=condition)
            loss = torch.mean(torch.sum((score_value * std + noise) ** 2, dim=(1, )))
            return loss
        else:
            raise NotImplementedError("Unknown type of ScoreFunction {}".format(type))
