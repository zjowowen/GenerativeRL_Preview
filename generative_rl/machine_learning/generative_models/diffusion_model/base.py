from typing import Union
from easydict import EasyDict
import torch
import torch.nn as nn
from tensordict import TensorDict

from generative_rl.numerical_methods.probability_path import GaussianConditionalProbabilityPath
from generative_rl.machine_learning.generative_models.diffusion_model.random_init import gaussian_random_variable
from generative_rl.machine_learning.generative_models.diffusion_model.score_model import ScoreFunction
from generative_rl.numerical_methods.diffusion_process import get_diffusion_process
from generative_rl.numerical_methods.numerical_solvers import get_solver

class DiffusionModel(nn.Module):
    """
    Overview:
        Diffusion Model.
    Interfaces:
        ``__init__``, ``sample``, ``score_matching_loss``
    """

    def __init__(
            self,
            config: EasyDict
            ) -> None:
        """
        Overview:
            Initialization of Diffusion Model.
        Arguments:
            - config (:obj:`EasyDict`): The configuration.
        """

        super().__init__()
        self.config = config
        self.x_size = config.x_size
        self.device = config.device

        self.solver = get_solver(config.solver.type)(config.solver.args)
        self.gaussian_generator = gaussian_random_variable(config.x_size, config.device)

        self.gaussian_conditional_probability_path = GaussianConditionalProbabilityPath(config.gaussian_conditional_probability_path.type)(config.gaussian_conditional_probability_path.args)
        self.diffusion_process = get_diffusion_process(config.diffusion_process)(self.gaussian_conditional_probability_path)
        self.score_function = ScoreFunction(config.score_function, self.gaussian_conditional_probability_path)

    def sample(
            self,
            condition: Union[torch.Tensor, TensorDict],
            **solver_kwargs,
        ):
        """
        Overview:
            Sample from the diffusion model.
        Arguments:
            - condition (:obj:`Union[torch.Tensor, TensorDict]`): The input condition.
            - solver_kwargs (:obj:`EasyDict`): The keyword arguments for the SDE solver or ODE solver.
        Returns:
            - x (:obj:`Union[torch.Tensor, TensorDict]`): The sampled result.
        """

        if self.config.solver.type.lower() == "dpmsolver":
            #TODO
            pass
        elif self.config.solver.type.lower() == "odesolver":
            x = self.gaussian_generator()
            self.diffusion_process.reverse_ode(score_function=self.score_function, condition=condition).sample(t=self.diffusion_process.t_max, x=x, **solver_kwargs)
        elif self.config.solver.type.lower() == "sdesolver":
            x = self.gaussian_generator()
            self.diffusion_process.reverse_sde(score_function=self.score_function, condition=condition).sample(t=self.diffusion_process.t_max, x=x, **solver_kwargs)
        else:
            raise NotImplementedError("Solver type {} is not implemented".format(self.config.solver.type))

    def score_matching_loss(
            self,
            x: Union[torch.Tensor, TensorDict],
            condition: Union[torch.Tensor, TensorDict],
        ) -> torch.Tensor:
        """
        Overview:
            The loss function for training unconditional diffusion model.
        Arguments:
            - x (:obj:`Union[torch.Tensor, TensorDict]`): The input.
            - condition (:obj:`Union[torch.Tensor, TensorDict]`): The input condition.
        """

        return self.score_function.score_matching_loss(x, condition)
