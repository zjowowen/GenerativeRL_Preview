from typing import Union
from easydict import EasyDict
import torch
import torch.nn as nn
from tensordict import TensorDict
from generative_rl.machine_learning.encoders import get_encoder
from generative_rl.machine_learning.modules import get_module
from generative_rl.machine_learning.generative_models.diffusion_model.random_init import gaussian_random_variable
from generative_rl.numerical_methods.numerical_solvers import get_solver
from generative_rl.numerical_methods.numerical_solvers.dpm_solver import DPMSolver
from generative_rl.numerical_methods.numerical_solvers.ode_solver import ODESolver
from generative_rl.numerical_methods.numerical_solvers.sde_solver import SDESolver
from generative_rl.numerical_methods.probability_path import GaussianConditionalProbabilityPath
from generative_rl.machine_learning.generative_models.diffusion_model.score_model import ScoreFunction
from generative_rl.numerical_methods.diffusion_process import get_diffusion_process


import gc
import time

class EnergyGuidance(nn.Module):
    """
    Overview:
        Energy Guidance for Energy Conditional Diffusion Model.
    Interfaces:
        ``__init__``, ``forward``, ``calculate_energy_guidance``
    """

    def __init__(self, config: EasyDict):
        """
        Overview:
            Initialization of Energy Guidance.
        Arguments:
            - config (:obj:`EasyDict`): The configuration.
        """
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
        self.model["backbone"] = get_module(config.backbone.type)(**config.backbone.args)

    def forward(
            self,
            t: torch.Tensor,
            x: Union[torch.Tensor, TensorDict],
            condition: Union[torch.Tensor, TensorDict] = None,
        ) -> torch.Tensor:
        """
        Overview:
            Return output of Energy Guidance.
        Arguments:
            - t (:obj:`torch.Tensor`): The input time.
            - x (:obj:`Union[torch.Tensor, TensorDict]`): The input.        
            - condition (:obj:`Union[torch.Tensor, TensorDict]`): The input condition.
        """
        
        if condition is not None:
            t = self.model["t_encoder"](t)
            x = self.model["x_encoder"](x)
            condition = self.model["condition_encoder"](condition)
            return self.model["backbone"](t, x, condition)
        else:
            t = self.model["t_encoder"](t)
            x = self.model["x_encoder"](x)
            return self.model["backbone"](t, x)

    def calculate_energy_guidance(
            self,
            t: torch.Tensor,
            x: Union[torch.Tensor, TensorDict],
            condition: Union[torch.Tensor, TensorDict] = None,
            guidance_scale: float = 1.0
        ) -> Union[torch.Tensor, TensorDict]:
        """
        Overview:
            Calculate the guidance for sampling.
        Arguments:
            - t (:obj:`torch.Tensor`): The input time.
            - x (:obj:`Union[torch.Tensor, TensorDict]`): The input.        
            - condition (:obj:`Union[torch.Tensor, TensorDict]`): The input condition.
            - guidance_scale (:obj:`float`): The scale of guidance.
        Returns:
            - guidance (:obj:`Union[torch.Tensor, TensorDict]`): The guidance for sampling.
        """

        # TODO: make it compatible with TensorDict
        with torch.enable_grad():
            x.requires_grad_(True)
            x_t = self.forward(t, x, condition)
            guidance = guidance_scale * torch.autograd.grad(torch.sum(x_t), x)[0]
        return guidance.detach()


class EnergyConditionalDiffusionModel(nn.Module):
    """
    Overview:
        Energy Conditional Diffusion Model.
    Interfaces:
        ``__init__``, ``sample``, ``energy_guidance_loss``
    """

    def __init__(
            self,
            config: EasyDict,
            energy_model: Union[torch.nn.Module, torch.nn.ModuleDict],
            ) -> None:
        """
        Overview:
            Initialization of Energy Conditional Diffusion Model.
        Arguments:
            - config (:obj:`EasyDict`): The configuration.
            - energy_model (:obj:`Union[torch.nn.Module, torch.nn.ModuleDict]`): The energy model.
        """

        super().__init__()
        self.config = config
        self.x_size = config.x_size
        self.device = config.device
        self.alpha = config.alpha

        self.gaussian_generator = gaussian_random_variable(config.x_size, config.device)

        self.gaussian_conditional_probability_path = GaussianConditionalProbabilityPath(config.gaussian_conditional_probability_path)
        self.diffusion_process = get_diffusion_process(config.diffusion_process)(self.gaussian_conditional_probability_path)
        self.score_function = ScoreFunction(config.score_function, self.gaussian_conditional_probability_path)

        self.energy_model = energy_model
        self.energy_guidance = EnergyGuidance(self.config.energy_guidance)

        if hasattr(config, "solver"):
            self.solver=get_solver(config.solver.type)(**config.solver.args)

    def score_function_with_energy_guidance(
            self,
            t: torch.Tensor,
            x: Union[torch.Tensor, TensorDict],
            condition: Union[torch.Tensor, TensorDict] = None,
            guidance_scale: float = 1.0
        ) -> torch.Tensor:
        """
        Overview:
            The score function for energy guidance.
        Arguments:
            - t (:obj:`torch.Tensor`): The input time.
            - x (:obj:`Union[torch.Tensor, TensorDict]`): The input state.
            - condition (:obj:`Union[torch.Tensor, TensorDict]`): The input condition.
            - guidance_scale (:obj:`float`): The scale of guidance.
        Returns:
            - score (:obj:`torch.Tensor`): The score function.
        """

        return self.score_function(t, x, condition) + self.energy_guidance.calculate_energy_guidance(t, x, condition, guidance_scale)

    def sample(
            self,
            condition: Union[torch.Tensor, TensorDict],
            guidance_scale: float = 1.0,
            with_grad: bool = False,
            solver_config: EasyDict = None,
        ):
        """
        Overview:
            Sample from the diffusion model.
        Arguments:
            - condition (:obj:`Union[torch.Tensor, TensorDict]`): The input condition.
            - guidance_scale (:obj:`float`): The scale of guidance.
            - with_grad (:obj:`bool`): Whether to return the gradient.
            - solver_config (:obj:`EasyDict`): The configuration of the solver.
        Returns:
            - x (:obj:`Union[torch.Tensor, TensorDict]`): The sampled result.
        """

        if solver_config is not None:
            solver = get_solver(solver_config.type)(**solver_config.args)
        else:
            assert hasattr(self, "solver"), "solver must be specified in config or solver_config"
            solver = self.solver

        def score_function_with_energy_guidance(t, x, condition):
            return self.score_function_with_energy_guidance(t, x, condition, guidance_scale)

        if isinstance(solver, DPMSolver):
            #TODO
            pass
        elif isinstance(solver, ODESolver):
            #TODO: make it compatible with TensorDict
            if not hasattr(self, "t_span") is None:
                self.t_span = torch.linspace(0, self.gaussian_conditional_probability_path.t_max, 2).to(self.device)
            x = self.gaussian_generator(batch_size=condition.shape[0])
            if with_grad:
                data = solver.integrate(
                    drift=self.diffusion_process.reverse_ode(score_function=score_function_with_energy_guidance, condition=condition).drift,
                    x0=x,
                    t_span=self.t_span,
                )[1]
            else:
                with torch.no_grad():
                    data = solver.integrate(
                        drift=self.diffusion_process.reverse_ode(score_function=score_function_with_energy_guidance, condition=condition).drift,
                        x0=x,
                        t_span=self.t_span,
                    )[1]
        elif isinstance(solver, SDESolver):
            #TODO: make it compatible with TensorDict
            #TODO: validate the implementation
            if not hasattr(self, "t_span") is None:
                self.t_span = torch.linspace(0, self.gaussian_conditional_probability_path.t_max, 2).to(self.device)
            x = self.gaussian_generator(batch_size=condition.shape[0])
            sde = self.diffusion_process.reverse_sde(score_function=score_function_with_energy_guidance, condition=condition)
            if with_grad:
                data = solver.integrate(
                    drift=sde.drift,
                    diffusion=sde.diffusion,
                    x0=x,
                    t_span=self.t_span,
                )[1]
            else:
                with torch.no_grad():
                    data = solver.integrate(
                        drift=sde.drift,
                        diffusion=sde.diffusion,
                        x0=x,
                        t_span=self.t_span,
                    )[1]
        else:
            raise NotImplementedError("Solver type {} is not implemented".format(self.config.solver.type))
        return data

    def sample_without_energy_guidance(
            self,
            condition: Union[torch.Tensor, TensorDict],
            solver_config: EasyDict = None,
        ):
        """
        Overview:
            Sample from the diffusion model without energy guidance.
        Arguments:
            - condition (:obj:`Union[torch.Tensor, TensorDict]`): The input condition.
            - solver_config (:obj:`EasyDict`): The configuration of the solver.
        Returns:
            - x (:obj:`Union[torch.Tensor, TensorDict]`): The sampled result.
        """

        return self.sample(condition, guidance_scale=0.0, solver_config=solver_config)

    def energy_guidance_loss(
            self,
            x: Union[torch.Tensor, TensorDict],
            condition: Union[torch.Tensor, TensorDict],
        ):
        """
        Overview:
            The loss function for training Energy Guidance.
        Arguments:
            - x (:obj:`Union[torch.Tensor, TensorDict]`): The input.
            - condition (:obj:`Union[torch.Tensor, TensorDict]`): The input condition.
        """
        # input many condition <bz, >  x <bz, M, >,
        energy = self.energy_model(x, torch.stack([condition] * x.shape[1], axis=1)).detach().squeeze()

        # CEP guidance method, as proposed in the paper

        #TODO: check math correctness
        #TODO: make it compatible with TensorDict
        #TODO: check eps = 1e-3
        eps = 1e-3
        t_random = torch.rand((x.shape[0], ), device=self.device) * (1. - eps) + eps
        t_random = torch.stack([t_random] * x.shape[1], dim=1)
        x_t = self.diffusion_process.direct_sample(t_random, x, condition)
        xt_energy_guidance = self.energy_guidance(t_random, x_t, torch.stack([condition] * x.shape[1], axis=1)).squeeze()
        log_xt_relative_energy = nn.LogSoftmax(dim=1)(xt_energy_guidance)
        x0_relative_energy = nn.Softmax(dim=1)(energy * self.alpha)
        loss = -torch.mean(torch.sum(x0_relative_energy * log_xt_relative_energy, axis=-1))
        return loss

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
