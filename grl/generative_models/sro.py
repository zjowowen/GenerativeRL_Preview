from typing import Any, Callable, Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict
from tensordict import TensorDict
from torch.distributions import Distribution

from grl.generative_models.diffusion_model.diffusion_model import \
    DiffusionModel


class SRPOConditionalDiffusionModel(nn.Module):
    """
    Overview:
        Energy Conditional Diffusion Model.
    Interfaces:
        ``__init__``, ``sample``, ``sample_without_energy_guidance``, ``sample_forward_process``, ``score_function``,
        ``score_function_with_energy_guidance``, ``score_matching_loss``, ``velocity_function``, ``flow_matching_loss``,
        ``energy_guidance_loss``
    """

    def __init__(
        self,
        config: EasyDict,
        value_model: Union[torch.nn.Module, torch.nn.ModuleDict],
        distribution_model,
    ) -> None:
        """
        Overview:
            Initialization of Energy Conditional Diffusion Model.
        Arguments:
            config (:obj:`EasyDict`): The configuration.
            energy_model (:obj:`Union[torch.nn.Module, torch.nn.ModuleDict]`): The energy model.
        """

        super().__init__()
        self.config = config
        self.diffusion_model = DiffusionModel(config)
        self.value_model = value_model
        self.distribution_model = distribution_model
        self.env_beta = config.beta

    def score_matching_loss(
        self,
        x: Union[torch.Tensor, TensorDict],
        condition: Union[torch.Tensor, TensorDict] = None,
    ) -> torch.Tensor:
        """
        Overview:
            The loss function for training unconditional diffusion model.
        Arguments:
            x (:obj:`Union[torch.Tensor, TensorDict]`): The input.
            condition (:obj:`Union[torch.Tensor, TensorDict]`): The input condition.
        """

        return self.diffusion_model.score_matching_loss(x, condition)

    def srpo_loss(
        self,
        condition: Union[torch.Tensor, TensorDict],  # state
    ):
        x = self.distribution_model(condition)
        t_random = torch.rand(x.shape[0], device=x.device) * 0.96 + 0.02  # [256]
        x_t = self.diffusion_model.diffusion_process.direct_sample(
            t_random, x
        )  # [256,6x]
        wt = self.diffusion_model.diffusion_process.std(t_random, x) ** 2
        with torch.no_grad():
            episilon = self.diffusion_model.noise_function(
                t_random, x_t, condition
            ).detach()
        detach_x = x.detach().requires_grad_(True)
        qs = self.value_model.q0_target.compute_double_q(detach_x, condition)
        q = (qs[0].squeeze() + qs[1].squeeze()) / 2.0
        guidance = torch.autograd.grad(torch.sum(q), detach_x)[0].detach()
        loss = (episilon * x) * wt - (guidance * x) * self.env_beta
        return loss, torch.mean(q)
