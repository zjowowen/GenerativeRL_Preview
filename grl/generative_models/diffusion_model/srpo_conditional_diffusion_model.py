from typing import Union, List, Tuple, Dict, Any, Callable
from easydict import EasyDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
from grl.generative_models.random_generator import gaussian_random_variable
from grl.numerical_methods.numerical_solvers import get_solver
from grl.numerical_methods.numerical_solvers.dpm_solver import DPMSolver
from grl.numerical_methods.numerical_solvers.ode_solver import ODESolver
from grl.numerical_methods.numerical_solvers.sde_solver import SDESolver
from grl.numerical_methods.probability_path import GaussianConditionalProbabilityPath
from grl.generative_models.intrinsic_model import IntrinsicModel
from grl.generative_models.diffusion_process import DiffusionProcess
from grl.generative_models.model_functions.score_function import ScoreFunction
from grl.generative_models.model_functions.velocity_function import VelocityFunction
from grl.generative_models.model_functions.data_prediction_function import DataPredictionFunction
from grl.generative_models.model_functions.noise_function import NoiseFunction

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
        distribution,
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
        if hasattr(config, "reverse_path"):
            self.reverse_path = GaussianConditionalProbabilityPath(config.reverse_path)
        else:
            self.reverse_path = None
        self.model_type = config.model.type
        assert self.model_type in [
            "score_function",
            "data_prediction_function",
            "velocity_function",
            "noise_function",
        ], "Unknown type of model {}".format(self.model_type)
        self.model = IntrinsicModel(config.model.args)
        self.diffusion_process = DiffusionProcess(self.path)
        if self.reverse_path is not None:
            self.reverse_diffusion_process = DiffusionProcess(self.reverse_path)
        else:
            self.reverse_diffusion_process = None
        self.score_function_ = ScoreFunction(self.model_type, self.diffusion_process)
        self.velocity_function_ = VelocityFunction(
            self.model_type, self.diffusion_process
        )
        self.noise_function_ = NoiseFunction(self.model_type, self.diffusion_process)
        self.data_prediction_function_ = DataPredictionFunction(
            self.model_type, self.diffusion_process
        )

        self.value = value_model
        self.env_beta = config.beta

        if hasattr(config, "solver"):
            self.solver = get_solver(config.solver.type)(**config.solver.args)

    def sample(
        self,
        t_span: torch.Tensor = None,
        condition: Union[torch.Tensor, TensorDict] = None,
        batch_size: Union[torch.Size, int, Tuple[int], List[int]] = None,
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
            assert hasattr(
                self, "solver"
            ), "solver must be specified in config or solver_config"
            solver = self.solver

        def score_function_with_energy_guidance(t, x, condition):
            return self.score_function_with_energy_guidance(
                t, x, condition, guidance_scale
            )

        def noise_function_with_energy_guidance(t, x, condition):
            return self.noise_function_with_energy_guidance(
                t, x, condition, guidance_scale
            )

        def data_prediction_function_with_energy_guidance(t, x, condition):
            return self.data_prediction_function_with_energy_guidance(
                t, x, condition, guidance_scale
            )

        if isinstance(solver, DPMSolver):
            # TODO: make it compatible with TensorDict
            x = self.gaussian_generator(batch_size=batch_size)
            if with_grad:
                data = solver.integrate(
                    diffusion_process=self.diffusion_process,
                    noise_function=noise_function_with_energy_guidance,
                    data_prediction_function=data_prediction_function_with_energy_guidance,
                    x=x,
                    condition=condition,
                    save_intermediate=False,
                )
            else:
                with torch.no_grad():
                    data = solver.integrate(
                        diffusion_process=self.diffusion_process,
                        noise_function=noise_function_with_energy_guidance,
                        data_prediction_function=data_prediction_function_with_energy_guidance,
                        x=x,
                        condition=condition,
                        save_intermediate=False,
                    )
        elif isinstance(solver, ODESolver):
            # TODO: make it compatible with TensorDict
            if not hasattr(self, "t_span"):
                self.t_span = torch.linspace(0, self.diffusion_process.t_max, 2).to(
                    self.device
                )
            x = self.gaussian_generator(batch_size=batch_size)
            if with_grad:
                data = solver.integrate(
                    drift=self.diffusion_process.reverse_ode(
                        function=score_function_with_energy_guidance,
                        function_type="score_function",
                        condition=condition,
                    ).drift,
                    x0=x,
                    t_span=self.t_span,
                )[-1]
            else:
                with torch.no_grad():
                    data = solver.integrate(
                        drift=self.diffusion_process.reverse_ode(
                            function=score_function_with_energy_guidance,
                            function_type="score_function",
                            condition=condition,
                        ).drift,
                        x0=x,
                        t_span=self.t_span,
                    )[-1]
        elif isinstance(solver, SDESolver):
            # TODO: make it compatible with TensorDict
            # TODO: validate the implementation
            assert (
                self.reverse_diffusion_process is not None
            ), "reverse_path must be specified in config"
            if not hasattr(self, "t_span"):
                self.t_span = torch.linspace(0, self.diffusion_process.t_max, 2).to(
                    self.device
                )
            x = self.gaussian_generator(batch_size=batch_size)
            sde = self.diffusion_process.reverse_sde(
                function=self.model,
                function_type=self.model_type,
                condition=condition,
                reverse_diffusion_function=self.reverse_diffusion_process.diffusion,
                reverse_diffusion_squared_function=self.reverse_diffusion_process.diffusion_squared,
            )
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
            raise NotImplementedError(
                "Solver type {} is not implemented".format(self.config.solver.type)
            )
        return data

    def sample_without_energy_guidance(
        self,
        t_span: torch.Tensor = None,
        condition: Union[torch.Tensor, TensorDict] = None,
        batch_size: Union[torch.Size, int, Tuple[int], List[int]] = None,
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

        def noise_function_with_energy_guidance(t, x, condition):
            return self.noise_function_with_energy_guidance(t, x, condition, guidance_scale)

        def data_prediction_function_with_energy_guidance(t, x, condition):
            return self.data_prediction_function_with_energy_guidance(t, x, condition, guidance_scale)

        if isinstance(solver, DPMSolver):
            # TODO: make it compatible with TensorDict
            x = self.gaussian_generator(batch_size=batch_size)
            if with_grad:
                data = solver.integrate(
                    diffusion_process=self.diffusion_process,
                    noise_function=noise_function_with_energy_guidance,
                    data_prediction_function=data_prediction_function_with_energy_guidance,
                    x=x,
                    condition=condition,
                    save_intermediate=True,
                )
            else:
                with torch.no_grad():
                    data = solver.integrate(
                        diffusion_process=self.diffusion_process,
                        noise_function=noise_function_with_energy_guidance,
                        data_prediction_function=data_prediction_function_with_energy_guidance,
                        x=x,
                        condition=condition,
                        save_intermediate=True,
                    )
        elif isinstance(solver, ODESolver):
            # TODO: make it compatible with TensorDict
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
            # TODO: make it compatible with TensorDict
            # TODO: validate the implementation
            assert self.reverse_diffusion_process is not None, "reverse_path must be specified in config"
            if not hasattr(self, "t_span"):
                self.t_span = torch.linspace(0, self.diffusion_process.t_max, 2).to(self.device)
            x = self.gaussian_generator(batch_size=batch_size)
            sde = self.diffusion_process.reverse_sde(function=self.model, function_type=self.model_type, condition=condition, reverse_diffusion_function=self.reverse_diffusion_process.diffusion, reverse_diffusion_squared_function=self.reverse_diffusion_process.diffusion_squared)
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
        ) -> Union[torch.Tensor, TensorDict]:
        """
        Overview:
            Return score function of the model at time t given the initial state, which is the gradient of the log-likelihood.
            .. math::
                $$\nabla_{x_t} \log p_{\theta}(x_t)$$
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
        ) -> Union[torch.Tensor, TensorDict]:
        """
        Overview:
            The score function for energy guidance.
        Arguments:
            - t (:obj:`torch.Tensor`): The input time.
            - x (:obj:`Union[torch.Tensor, TensorDict]`): The input state.
            - condition (:obj:`Union[torch.Tensor, TensorDict]`): The input condition.
            - guidance_scale (:obj:`float`): The scale of guidance.
        Returns:
            - score (:obj:`Union[torch.Tensor, TensorDict]`): The score function.
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
        ) -> Union[torch.Tensor, TensorDict]:
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

    def noise_function(
            self,
            t: torch.Tensor,
            x: Union[torch.Tensor, TensorDict],
            condition: Union[torch.Tensor, TensorDict] = None,
        ) -> Union[torch.Tensor, TensorDict]:
        """
        Overview:
            Return noise function of the model at time t given the initial state.
            .. math::
                - \sigma(t) \nabla_{x_t} \log p_{\theta}(x_t)
        Arguments:
            - t (:obj:`torch.Tensor`): The input time.
            - x (:obj:`Union[torch.Tensor, TensorDict]`): The input state.
            - condition (:obj:`Union[torch.Tensor, TensorDict]`): The input condition.
        """

        return self.noise_function_.forward(self.model, t, x, condition)

    def noise_function_with_energy_guidance(
            self,
            t: torch.Tensor,
            x: Union[torch.Tensor, TensorDict],
            condition: Union[torch.Tensor, TensorDict] = None,
            guidance_scale: float = 1.0
        ) -> Union[torch.Tensor, TensorDict]:
        """
        Overview:
            The noise function for energy guidance.
        Arguments:
            - t (:obj:`torch.Tensor`): The input time.
            - x (:obj:`Union[torch.Tensor, TensorDict]`): The input state.
            - condition (:obj:`Union[torch.Tensor, TensorDict]`): The input condition.
            - guidance_scale (:obj:`float`): The scale of guidance.
        Returns:
            - noise (:obj:`Union[torch.Tensor, TensorDict]`): The nose function.
        """

        return - self.score_function_with_energy_guidance(t, x, condition, guidance_scale) * self.diffusion_process.std(t, x)

    def data_prediction_function(
            self,
            t: torch.Tensor,
            x: Union[torch.Tensor, TensorDict],
            condition: Union[torch.Tensor, TensorDict] = None,
        ) -> Union[torch.Tensor, TensorDict]:
        """
        Overview:
            Return data prediction function of the model at time t given the initial state.
            .. math::
                \frac{- \sigma(t) x_t + \sigma^2(t) \nabla_{x_t} \log p_{\theta}(x_t)}{s(t)}
        Arguments:
            - t (:obj:`torch.Tensor`): The input time.
            - x (:obj:`Union[torch.Tensor, TensorDict]`): The input state.
            - condition (:obj:`Union[torch.Tensor, TensorDict]`): The input condition.
        """

        return self.data_prediction_function_.forward(self.model, t, x, condition)

    def data_prediction_function_with_energy_guidance(
            self,
            t: torch.Tensor,
            x: Union[torch.Tensor, TensorDict],
            condition: Union[torch.Tensor, TensorDict] = None,
            guidance_scale: float = 1.0
        ) -> Union[torch.Tensor, TensorDict]:
        """
        Overview:
            The data prediction function for energy guidance.
        Arguments:
            - t (:obj:`torch.Tensor`): The input time.
            - x (:obj:`Union[torch.Tensor, TensorDict]`): The input state.
            - condition (:obj:`Union[torch.Tensor, TensorDict]`): The input condition.
            - guidance_scale (:obj:`float`): The scale of guidance.
        Returns:
            - x (:obj:`torch.Tensor`): The score function.
        """

        return (- self.diffusion_process.std(t, x) * x + self.diffusion_process.covariance(t, x) * self.score_function_with_energy_guidance(t, x, condition, guidance_scale)) / self.diffusion_process.scale(t, x)

    def srpo_loss(
        self,
        x: Union[torch.Tensor, TensorDict],  #action
        condition: Union[torch.Tensor, TensorDict],  #state
    ):
        eps = 1e-4
        t_random = torch.rand(x.shape[0], device=self.device) *  (1. - eps) + eps  #[256]
        x_t = self.diffusion_process.direct_sample(t_random, x)#[256,6x]
        wt = self.diffusion_process.std(t_random,x) ** 2
        with torch.no_grad():
            episilon = self.noise_function( t_random,x_t, condition).detach()
        detach_x = x.detach().requires_grad_(True)
        qs = self.value.q0_target.compute_double_q(detach_x , condition)
        q = (qs[0].squeeze() + qs[1].squeeze()) / 2.0
        guidance =  torch.autograd.grad(torch.sum(q), detach_x)[0].detach()
        loss = (episilon * x) * wt - (guidance * x) * self.env_beta
        return loss
