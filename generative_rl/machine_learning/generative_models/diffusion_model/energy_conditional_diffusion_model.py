from typing import Union, List, Tuple, Dict, Any, Callable
from easydict import EasyDict
import torch
import torch.nn as nn
from tensordict import TensorDict
from generative_rl.machine_learning.generative_models.random_generator import gaussian_random_variable
from generative_rl.numerical_methods.numerical_solvers import get_solver
from generative_rl.numerical_methods.numerical_solvers.dpm_solver import DPMSolver
from generative_rl.numerical_methods.numerical_solvers.ode_solver import ODESolver
from generative_rl.numerical_methods.numerical_solvers.sde_solver import SDESolver
from generative_rl.numerical_methods.probability_path import GaussianConditionalProbabilityPath
from generative_rl.machine_learning.generative_models.intrinsic_model import IntrinsicModel
from generative_rl.machine_learning.generative_models.diffusion_process import DiffusionProcess
from generative_rl.machine_learning.generative_models.model_functions.score_function import ScoreFunction
from generative_rl.machine_learning.generative_models.model_functions.velocity_function import VelocityFunction

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
        self.model = IntrinsicModel(self.config)

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

        return self.model(t, x, condition)

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
        ``__init__``, ``sample``, ``sample_without_energy_guidance``, ``sample_forward_process``, ``score_function``,
        ``score_function_with_energy_guidance``, ``score_matching_loss``, ``velocity_function``, ``flow_matching_loss``,
        ``energy_guidance_loss``
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

        self.path = GaussianConditionalProbabilityPath(config.path)
        self.model_type = config.model.type
        assert self.model_type in ["score_function", "data_prediction_function", "velocity_function", "noise_function"], \
            "Unknown type of model {}".format(self.model_type)
        self.model = IntrinsicModel(config.model.args)
        self.diffusion_process = DiffusionProcess(self.path)
        self.score_function_ = ScoreFunction(self.model_type, self.diffusion_process)
        self.velocity_function_ = VelocityFunction(self.model_type, self.diffusion_process)

        self.energy_model = energy_model
        self.energy_guidance = EnergyGuidance(self.config.energy_guidance)

        if hasattr(config, "solver"):
            self.solver=get_solver(config.solver.type)(**config.solver.args)

    def sample(
            self,
            t_span: torch.Tensor = None,
            condition: Union[torch.Tensor, TensorDict] = None,
            batch_size: Union[torch.Size, int, Tuple[int], List[int]]  = None,
            guidance_scale: float = 1.0,
            with_grad: bool = False,
            solver_config: EasyDict = None,
        ):
        """
        Overview:
            Sample from the diffusion model.
        Arguments:
            - t_span (:obj:`torch.Tensor`): The time span.
            - condition (:obj:`Union[torch.Tensor, TensorDict]`): The input condition.
            - batch_size (:obj:`Union[torch.Size, int, Tuple[int], List[int]]`): The batch size.
            - with_grad (:obj:`bool`): Whether to return the gradient.
            - solver_config (:obj:`EasyDict`): The configuration of the solver.
        Returns:
            - x (:obj:`Union[torch.Tensor, TensorDict]`): The sampled result.
        """

        if t_span is not None:
            self.t_span = t_span.to(self.device)

        if batch_size is not None:
            pass
        elif condition is not None:
            batch_size = condition.shape[0]
        else:
            batch_size = 1

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
            if not hasattr(self, "t_span"):
                self.t_span = torch.linspace(0, self.diffusion_process.t_max, 2).to(self.device)
            x = self.gaussian_generator(batch_size=batch_size)
            if with_grad:
                data = solver.integrate(
                    drift=self.diffusion_process.reverse_ode(function=score_function_with_energy_guidance, function_type="score_function", condition=condition).drift,
                    x0=x,
                    t_span=self.t_span,
                )[-1]
            else:
                with torch.no_grad():
                    data = solver.integrate(
                        drift=self.diffusion_process.reverse_ode(function=score_function_with_energy_guidance, function_type="score_function", condition=condition).drift,
                        x0=x,
                        t_span=self.t_span,
                    )[-1]
        elif isinstance(solver, SDESolver):
            #TODO: make it compatible with TensorDict
            #TODO: validate the implementation
            if not hasattr(self, "t_span"):
                self.t_span = torch.linspace(0, self.diffusion_process.t_max, 2).to(self.device)
            x = self.gaussian_generator(batch_size=batch_size)
            sde = self.diffusion_process.reverse_sde(function=score_function_with_energy_guidance, function_type="score_function", condition=condition)
            if with_grad:
                data = solver.integrate(
                    drift=sde.drift,
                    diffusion=sde.diffusion,
                    x0=x,
                    t_span=self.t_span,
                )[-1]
            else:
                with torch.no_grad():
                    data = solver.integrate(
                        drift=sde.drift,
                        diffusion=sde.diffusion,
                        x0=x,
                        t_span=self.t_span,
                    )[-1]
        else:
            raise NotImplementedError("Solver type {} is not implemented".format(self.config.solver.type))
        return data

    def sample_without_energy_guidance(
            self,
            t_span: torch.Tensor = None,
            condition: Union[torch.Tensor, TensorDict] = None,
            batch_size: Union[torch.Size, int, Tuple[int], List[int]]  = None,
            solver_config: EasyDict = None,
        ):
        """
        Overview:
            Sample from the diffusion model without energy guidance.
        Arguments:
            - t_span (:obj:`torch.Tensor`): The time span.
            - condition (:obj:`Union[torch.Tensor, TensorDict]`): The input condition.
            - batch_size (:obj:`Union[torch.Size, int, Tuple[int], List[int]]`): The batch size.
            - solver_config (:obj:`EasyDict`): The configuration of the solver.
        Returns:
            - x (:obj:`Union[torch.Tensor, TensorDict]`): The sampled result.
        """

        return self.sample(
            t_span=t_span,
            condition=condition,
            batch_size=batch_size,
            guidance_scale=0.0,
            solver_config=solver_config)

    def sample_forward_process(
            self,
            t_span: torch.Tensor,
            batch_size: Union[torch.Size, int, Tuple[int], List[int]] = None,
            condition: Union[torch.Tensor, TensorDict] = None,
            guidance_scale: float = 1.0,
            with_grad: bool = False,
            solver_config: EasyDict = None,
        ):
        """
        Overview:
            Sample from the diffusion model.
        Arguments:
            - t_span (:obj:`torch.Tensor`): The time span.
            - batch_size (:obj:`Union[torch.Size, int, Tuple[int], List[int]]`): The batch size.
            - condition (:obj:`Union[torch.Tensor, TensorDict]`): The input condition.
            - with_grad (:obj:`bool`): Whether to return the gradient.
            - solver_config (:obj:`EasyDict`): The configuration of the solver.
        Returns:
            - x (:obj:`Union[torch.Tensor, TensorDict]`): The sampled result.
        """

        t_span = t_span.to(self.device)
        
        if batch_size is not None:
            pass
        elif condition is not None:
            batch_size = condition.shape[0]
        else:
            batch_size = 1
        
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
            x = self.gaussian_generator(batch_size=batch_size)
            if with_grad:
                data = solver.integrate(
                    drift=self.diffusion_process.reverse_ode(function=score_function_with_energy_guidance, function_type="score_function", condition=condition).drift,
                    x0=x,
                    t_span=t_span,
                )
            else:
                with torch.no_grad():
                    data = solver.integrate(
                        drift=self.diffusion_process.reverse_ode(function=score_function_with_energy_guidance, function_type="score_function", condition=condition).drift,
                        x0=x,
                        t_span=t_span,
                    )
        elif isinstance(solver, SDESolver):
            #TODO: make it compatible with TensorDict
            #TODO: validate the implementation
            x = self.gaussian_generator(batch_size=batch_size)
            sde = self.diffusion_process.reverse_sde(function=score_function_with_energy_guidance, function_type="score_function", condition=condition)
            if with_grad:
                data = solver.integrate(
                    drift=sde.drift,
                    diffusion=sde.diffusion,
                    x0=x,
                    t_span=t_span,
                )
            else:
                with torch.no_grad():
                    data = solver.integrate(
                        drift=sde.drift,
                        diffusion=sde.diffusion,
                        x0=x,
                        t_span=t_span,
                    )
        else:
            raise NotImplementedError("Solver type {} is not implemented".format(self.config.solver.type))
        return data

    def score_function(
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

        return self.score_function_.forward(self.model, t, x, condition)

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

        return self.score_function_.forward(self.model, t, x, condition) + self.energy_guidance.calculate_energy_guidance(t, x, condition, guidance_scale)

    def score_matching_loss(
            self,
            x: Union[torch.Tensor, TensorDict],
            condition: Union[torch.Tensor, TensorDict] = None,
        ) -> torch.Tensor:
        """
        Overview:
            The loss function for training unconditional diffusion model.
        Arguments:
            - x (:obj:`Union[torch.Tensor, TensorDict]`): The input.
            - condition (:obj:`Union[torch.Tensor, TensorDict]`): The input condition.
        """

        return self.score_function_.score_matching_loss(self.model, x, condition)

    def velocity_function(
            self,
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
            - t (:obj:`torch.Tensor`): The input time.
            - x (:obj:`Union[torch.Tensor, TensorDict]`): The input state at time t.
            - condition (:obj:`Union[torch.Tensor, TensorDict]`): The input condition.
        """

        return self.velocity_function_.forward(self.model, t, x, condition)
    
    def flow_matching_loss(
            self,
            x: Union[torch.Tensor, TensorDict],
            condition: Union[torch.Tensor, TensorDict] = None,
        ) -> torch.Tensor:
        """
        Overview:
            Return the flow matching loss function of the model given the initial state and the condition.
        Arguments:
            - x (:obj:`Union[torch.Tensor, TensorDict]`): The input state.
            - condition (:obj:`Union[torch.Tensor, TensorDict]`): The input condition.
        """

        return self.velocity_function_.flow_matching_loss(self.model, x, condition)

    def energy_guidance_loss(
            self,
            x: Union[torch.Tensor, TensorDict],
            condition: Union[torch.Tensor, TensorDict],
        ):
        """
        Overview:
            The loss function for training Energy Guidance, CEP guidance method, as proposed in the paper \
            "Contrastive Energy Prediction for Exact Energy-Guided Diffusion Sampling in Offline Reinforcement Learning"
        Arguments:
            - x (:obj:`Union[torch.Tensor, TensorDict]`): The input.
            - condition (:obj:`Union[torch.Tensor, TensorDict]`): The input condition.
        """
        #TODO: check math correctness
        #TODO: make it compatible with TensorDict
        #TODO: check eps = 1e-3
        eps = 1e-3
        t_random = torch.rand((x.shape[0], ), device=self.device) * (1. - eps) + eps
        t_random = torch.stack([t_random] * x.shape[1], dim=1)
        energy = self.energy_model(x, torch.stack([condition] * x.shape[1], axis=1)).detach().squeeze()
        x_t = self.diffusion_process.direct_sample(t_random, x, condition)
        xt_energy_guidance = self.energy_guidance(t_random, x_t, torch.stack([condition] * x.shape[1], axis=1)).squeeze()
        log_xt_relative_energy = nn.LogSoftmax(dim=1)(xt_energy_guidance)
        x0_relative_energy = nn.Softmax(dim=1)(energy * self.alpha)
        loss = -torch.mean(torch.sum(x0_relative_energy * log_xt_relative_energy, axis=-1))
        return loss
