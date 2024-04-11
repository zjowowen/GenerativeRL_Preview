from typing import Union, List, Tuple, Dict, Any, Callable
from easydict import EasyDict
import torch
import torch.nn as nn
from tensordict import TensorDict

from grl.numerical_methods.probability_path import GaussianConditionalProbabilityPath
from grl.numerical_methods.numerical_solvers import get_solver
from grl.numerical_methods.numerical_solvers.dpm_solver import DPMSolver
from grl.numerical_methods.numerical_solvers.ode_solver import ODESolver
from grl.numerical_methods.numerical_solvers.sde_solver import SDESolver
from grl.generative_models.random_generator import gaussian_random_variable
from grl.generative_models.intrinsic_model import IntrinsicModel
from grl.generative_models.diffusion_process import DiffusionProcess
from grl.generative_models.model_functions.score_function import ScoreFunction
from grl.generative_models.model_functions.velocity_function import VelocityFunction
from grl.generative_models.model_functions.data_prediction_function import DataPredictionFunction
from grl.generative_models.model_functions.noise_function import NoiseFunction

class DiffusionModel(nn.Module):
    """
    Overview:
        Diffusion Model.
    Interfaces:
        ``__init__``, ``sample``, ``score_function``, ``score_matching_loss``, ``velocity_function``, ``flow_matching_loss``.
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

        self.gaussian_generator = gaussian_random_variable(config.x_size, config.device)

        self.path = GaussianConditionalProbabilityPath(config.path)
        if hasattr(config, "reverse_path"):
            self.reverse_path = GaussianConditionalProbabilityPath(config.reverse_path)
        else:
            self.reverse_path = None
        self.model_type = config.model.type
        assert self.model_type in ["score_function", "data_prediction_function", "velocity_function", "noise_function"], \
            "Unknown type of model {}".format(self.model_type)
        self.model = IntrinsicModel(config.model.args)
        self.diffusion_process = DiffusionProcess(self.path)
        if self.reverse_path is not None:
            self.reverse_diffusion_process = DiffusionProcess(self.reverse_path)
        else:
            self.reverse_diffusion_process = None
        self.score_function_ = ScoreFunction(self.model_type, self.diffusion_process)
        self.velocity_function_ = VelocityFunction(self.model_type, self.diffusion_process)
        self.noise_function_ = NoiseFunction(self.model_type, self.diffusion_process)
        self.data_prediction_function_ = DataPredictionFunction(self.model_type, self.diffusion_process)

        if hasattr(config, "solver"):
            self.solver=get_solver(config.solver.type)(**config.solver.args)

    def sample(
            self,
            t_span: torch.Tensor = None,
            condition: Union[torch.Tensor, TensorDict] = None,
            batch_size: Union[torch.Size, int, Tuple[int], List[int]]  = None,
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

        if isinstance(solver, DPMSolver):
            #TODO: make it compatible with TensorDict
            x = self.gaussian_generator(batch_size=batch_size)
            if with_grad:
                data = solver.integrate(
                    diffusion_process=self.diffusion_process,
                    noise_function=self.noise_function,
                    data_prediction_function=self.data_prediction_function,
                    x=x,
                    condition=condition,
                    save_intermediate=False,
                )
            else:
                with torch.no_grad():
                    data = solver.integrate(
                        diffusion_process=self.diffusion_process,
                        noise_function=self.noise_function,
                        data_prediction_function=self.data_prediction_function,
                        x=x,
                        condition=condition,
                        save_intermediate=False,
                    )
        elif isinstance(solver, ODESolver):
            #TODO: make it compatible with TensorDict
            if not hasattr(self, "t_span"):
                self.t_span = torch.linspace(0, self.diffusion_process.t_max, 2).to(self.device)
            x = self.gaussian_generator(batch_size=batch_size)
            if with_grad:
                data = solver.integrate(
                    drift=self.diffusion_process.reverse_ode(function=self.model, function_type=self.model_type, condition=condition).drift,
                    x0=x,
                    t_span=self.t_span,
                )[-1]
            else:
                with torch.no_grad():
                    data = solver.integrate(
                        drift=self.diffusion_process.reverse_ode(function=self.model, function_type=self.model_type, condition=condition).drift,
                        x0=x,
                        t_span=self.t_span,
                    )[-1]
        elif isinstance(solver, SDESolver):
            #TODO: make it compatible with TensorDict
            #TODO: validate the implementation
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

    def sample_forward_process(
            self,
            t_span: torch.Tensor = None,
            batch_size: Union[torch.Size, int, Tuple[int], List[int]] = None,
            condition: Union[torch.Tensor, TensorDict] = None,
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

        if isinstance(solver, DPMSolver):
            #TODO: make it compatible with TensorDict
            x = self.gaussian_generator(batch_size=batch_size)
            if with_grad:
                data = solver.integrate(
                    diffusion_process=self.diffusion_process,
                    noise_function=self.noise_function,
                    data_prediction_function=self.data_prediction_function,
                    x=x,
                    condition=condition,
                    save_intermediate=True,
                )
            else:
                with torch.no_grad():
                    data = solver.integrate(
                        diffusion_process=self.diffusion_process,
                        noise_function=self.noise_function,
                        data_prediction_function=self.data_prediction_function,
                        x=x,
                        condition=condition,
                        save_intermediate=True,
                    )
        elif isinstance(solver, ODESolver):
            #TODO: make it compatible with TensorDict
            x = self.gaussian_generator(batch_size=batch_size)
            if with_grad:
                data = solver.integrate(
                    drift=self.diffusion_process.reverse_ode(function=self.model, function_type=self.model_type, condition=condition).drift,
                    x0=x,
                    t_span=t_span,
                )
            else:
                with torch.no_grad():
                    data = solver.integrate(
                        drift=self.diffusion_process.reverse_ode(function=self.model, function_type=self.model_type, condition=condition).drift,
                        x0=x,
                        t_span=t_span,
                    )
        elif isinstance(solver, SDESolver):
            #TODO: make it compatible with TensorDict
            #TODO: validate the implementation
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

    def sample_with_fixed_x(
            self,
            fixed_x: Union[torch.Tensor, TensorDict],
            fixed_mask: Union[torch.Tensor, TensorDict],
            t_span: torch.Tensor = None,
            condition: Union[torch.Tensor, TensorDict] = None,
            batch_size: Union[torch.Size, int, Tuple[int], List[int]]  = None,
            with_grad: bool = False,
            solver_config: EasyDict = None,
        ):
        """
        Overview:
            Sample from the diffusion model with fixed x.
        Arguments:
            - fixed_x (:obj:`Union[torch.Tensor, TensorDict]`): The fixed x.
            - fixed_mask (:obj:`Union[torch.Tensor, TensorDict]`): The fixed mask.
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

        if isinstance(solver, DPMSolver):
            #TODO: make it compatible with DPM solver
            assert False, "Not implemented"
            #TODO: make it compatible with TensorDict
            x = self.gaussian_generator(batch_size=batch_size)
            if with_grad:
                data = solver.integrate(
                    diffusion_process=self.diffusion_process,
                    noise_function=self.noise_function,
                    data_prediction_function=self.data_prediction_function,
                    x=x,
                    condition=condition,
                    save_intermediate=False,
                )
            else:
                with torch.no_grad():
                    data = solver.integrate(
                        diffusion_process=self.diffusion_process,
                        noise_function=self.noise_function,
                        data_prediction_function=self.data_prediction_function,
                        x=x,
                        condition=condition,
                        save_intermediate=False,
                    )
        elif isinstance(solver, ODESolver):
            #TODO: make it compatible with TensorDict
            if not hasattr(self, "t_span"):
                self.t_span = torch.linspace(0, self.diffusion_process.t_max, 2).to(self.device)
            # make fixed_x and fixed_mask compatible with batch_size
            fixed_x = fixed_x.unsqueeze(0).expand(batch_size, -1)
            fixed_mask = fixed_mask.unsqueeze(0).expand(batch_size, -1)

            x = fixed_x * (1 - fixed_mask) + self.gaussian_generator(batch_size=batch_size) * fixed_mask
            def drift_fixed_x(t, x):
                xt_partially_fixed = self.diffusion_process.direct_sample(self.diffusion_process.t_max-t, fixed_x) * (1 - fixed_mask) + x * fixed_mask
                return fixed_mask * self.diffusion_process.reverse_ode(function=self.model, function_type=self.model_type, condition=condition).drift(t, xt_partially_fixed)
            if with_grad:
                data = solver.integrate(
                    drift=drift_fixed_x,
                    x0=x,
                    t_span=self.t_span,
                )[-1]
            else:
                with torch.no_grad():
                    data = solver.integrate(
                        drift=drift_fixed_x,
                        x0=x,
                        t_span=self.t_span,
                    )[-1]
        elif isinstance(solver, SDESolver):
            #TODO: make it compatible with TensorDict
            #TODO: validate the implementation
            assert self.reverse_diffusion_process is not None, "reverse_path must be specified in config"
            if not hasattr(self, "t_span"):
                self.t_span = torch.linspace(0, self.diffusion_process.t_max, 2).to(self.device)
            # make fixed_x and fixed_mask compatible with batch_size
            fixed_x = fixed_x.unsqueeze(0).expand(batch_size, -1)
            fixed_mask = fixed_mask.unsqueeze(0).expand(batch_size, -1)

            x = fixed_x * (1 - fixed_mask) + self.gaussian_generator(batch_size=batch_size) * fixed_mask
            sde = self.diffusion_process.reverse_sde(
                function=self.model,
                function_type=self.model_type,
                condition=condition,
                reverse_diffusion_function=self.reverse_diffusion_process.diffusion,
                reverse_diffusion_squared_function=self.reverse_diffusion_process.diffusion_squared,
            )
            def drift_fixed_x(t, x):
                xt_partially_fixed = self.diffusion_process.direct_sample(self.diffusion_process.t_max-t, fixed_x) * (1 - fixed_mask) + x * fixed_mask
                return fixed_mask * sde.drift(t, xt_partially_fixed)
            def diffusion_fixed_x(t, x):
                xt_partially_fixed = self.diffusion_process.direct_sample(self.diffusion_process.t_max-t, fixed_x) * (1 - fixed_mask) + x * fixed_mask
                return fixed_mask * sde.diffusion(t, xt_partially_fixed)
            if with_grad:
                data = solver.integrate(
                    drift=drift_fixed_x,
                    diffusion=diffusion_fixed_x,
                    x0=x,
                    t_span=self.t_span,
                )[-1]
            else:
                with torch.no_grad():
                    data = solver.integrate(
                        drift=drift_fixed_x,
                        diffusion=diffusion_fixed_x,
                        x0=x,
                        t_span=self.t_span,
                    )[-1]
        else:
            raise NotImplementedError("Solver type {} is not implemented".format(self.config.solver.type))
        return data

    def sample_forward_process_with_fixed_x(
            self,
            fixed_x: Union[torch.Tensor, TensorDict],
            fixed_mask: Union[torch.Tensor, TensorDict],
            t_span: torch.Tensor = None,
            batch_size: Union[torch.Size, int, Tuple[int], List[int]] = None,
            condition: Union[torch.Tensor, TensorDict] = None,
            with_grad: bool = False,
            solver_config: EasyDict = None,
        ):
        """
        Overview:
            Sample from the diffusion model with fixed x.
        Arguments:
            - fixed_x (:obj:`Union[torch.Tensor, TensorDict]`): The fixed x.
            - fixed_mask (:obj:`Union[torch.Tensor, TensorDict]`): The fixed mask.
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

        if isinstance(solver, DPMSolver):
            #TODO: make it compatible with DPM solver
            assert False, "Not implemented"
            #TODO: make it compatible with TensorDict
            x = self.gaussian_generator(batch_size=batch_size)
            if with_grad:
                data = solver.integrate(
                    diffusion_process=self.diffusion_process,
                    noise_function=self.noise_function,
                    data_prediction_function=self.data_prediction_function,
                    x=x,
                    condition=condition,
                    save_intermediate=True,
                )
            else:
                with torch.no_grad():
                    data = solver.integrate(
                        diffusion_process=self.diffusion_process,
                        noise_function=self.noise_function,
                        data_prediction_function=self.data_prediction_function,
                        x=x,
                        condition=condition,
                        save_intermediate=True,
                    )
        elif isinstance(solver, ODESolver):
            #TODO: make it compatible with TensorDict
            # make fixed_x and fixed_mask compatible with batch_size
            fixed_x = fixed_x.unsqueeze(0).expand(batch_size, -1)
            fixed_mask = fixed_mask.unsqueeze(0).expand(batch_size, -1)

            x = fixed_x * (1 - fixed_mask) + self.gaussian_generator(batch_size=batch_size) * fixed_mask
            def drift_fixed_x(t, x):
                xt_partially_fixed = self.diffusion_process.direct_sample(self.diffusion_process.t_max-t, fixed_x) * (1 - fixed_mask) + x * fixed_mask
                return fixed_mask * self.diffusion_process.reverse_ode(function=self.model, function_type=self.model_type, condition=condition).drift(t, xt_partially_fixed)
            if with_grad:
                data = solver.integrate(
                    drift=drift_fixed_x,
                    x0=x,
                    t_span=t_span,
                )
            else:
                with torch.no_grad():
                    data = solver.integrate(
                        drift=drift_fixed_x,
                        x0=x,
                        t_span=t_span,
                    )
        elif isinstance(solver, SDESolver):
            #TODO: make it compatible with TensorDict
            #TODO: validate the implementation
            assert self.reverse_diffusion_process is not None, "reverse_path must be specified in config"
            if not hasattr(self, "t_span"):
                self.t_span = torch.linspace(0, self.diffusion_process.t_max, 2).to(self.device)
            # make fixed_x and fixed_mask compatible with batch_size
            fixed_x = fixed_x.unsqueeze(0).expand(batch_size, -1)
            fixed_mask = fixed_mask.unsqueeze(0).expand(batch_size, -1)

            x = fixed_x * (1 - fixed_mask) + self.gaussian_generator(batch_size=batch_size) * fixed_mask
            sde = self.diffusion_process.reverse_sde(
                function=self.model,
                function_type=self.model_type,
                condition=condition,
                reverse_diffusion_function=self.reverse_diffusion_process.diffusion,
                reverse_diffusion_squared_function=self.reverse_diffusion_process.diffusion_squared,
            )
            def drift_fixed_x(t, x):
                xt_partially_fixed = self.diffusion_process.direct_sample(self.diffusion_process.t_max-t, fixed_x) * (1 - fixed_mask) + x * fixed_mask
                return fixed_mask * sde.drift(t, xt_partially_fixed)
            def diffusion_fixed_x(t, x):
                xt_partially_fixed = self.diffusion_process.direct_sample(self.diffusion_process.t_max-t, fixed_x) * (1 - fixed_mask) + x * fixed_mask
                return fixed_mask * sde.diffusion(t, xt_partially_fixed)
            if with_grad:
                data = solver.integrate(
                    drift=drift_fixed_x,
                    diffusion=diffusion_fixed_x,
                    x0=x,
                    t_span=t_span,
                )
            else:
                with torch.no_grad():
                    data = solver.integrate(
                        drift=drift_fixed_x,
                        diffusion=diffusion_fixed_x,
                        x0=x,
                        t_span=t_span,
                    )

        else:
            raise NotImplementedError("Solver type {} is not implemented".format(self.config.solver.type))
        return data

    def forward_sample(
            self,
            x: Union[torch.Tensor, TensorDict],
            t_span: torch.Tensor,
            condition: Union[torch.Tensor, TensorDict] = None,
            with_grad: bool = False,
            solver_config: EasyDict = None,
        ):
        """
        Overview:
            Sample from the diffusion model given the sampled x.
        Arguments:
            - x (:obj:`Union[torch.Tensor, TensorDict]`): The input state.
            - t_span (:obj:`torch.Tensor`): The time span.
            - condition (:obj:`Union[torch.Tensor, TensorDict]`): The input condition.
            - with_grad (:obj:`bool`): Whether to return the gradient.
            - solver_config (:obj:`EasyDict`): The configuration of the solver.
        """

        #TODO: very important function
        #TODO: validate these functions

        t_span = t_span.to(self.device)

        batch_size = x.shape[0]

        if solver_config is not None:
            solver = get_solver(solver_config.type)(**solver_config.args)
        else:
            assert hasattr(self, "solver"), "solver must be specified in config or solver_config"
            solver = self.solver

        if isinstance(solver, ODESolver):
            #TODO: make it compatible with TensorDict

            if with_grad:
                data = solver.integrate(
                    drift=self.diffusion_process.forward_reversed_ode(function=self.model, function_type=self.model_type, condition=condition).drift,
                    x0=x,
                    t_span=t_span,
                )[-1]
            else:
                with torch.no_grad():
                    data = solver.integrate(
                        drift=self.diffusion_process.forward_reversed_ode(function=self.model, function_type=self.model_type, condition=condition).drift,
                        x0=x,
                        t_span=t_span,
                    )[-1]
        elif isinstance(solver, SDESolver):
            #TODO: make it compatible with TensorDict
            #TODO: validate the implementation
            assert self.diffusion_process is not None, "path must be specified in config"

            sde = self.diffusion_process.forward_reversed_sde(function=self.model, function_type=self.model_type, condition=condition, forward_diffusion_function=self.diffusion_process.diffusion, forward_diffusion_squared_function=self.diffusion_process.diffusion_squared)
            if with_grad:
                data = solver.integrate(
                    drift=sde.drift,
                    diffusion=sde.diffusion,
                    x0=x,
                    t_span=t_span,
                )[-1]
            else:
                with torch.no_grad():
                    data = solver.integrate(
                        drift=sde.drift,
                        diffusion=sde.diffusion,
                        x0=x,
                        t_span=t_span,
                    )[-1]
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
                \nabla_{x_t} \log p_{\theta}(x_t)
        Arguments:
            - t (:obj:`torch.Tensor`): The input time.
            - x (:obj:`Union[torch.Tensor, TensorDict]`): The input state.
            - condition (:obj:`Union[torch.Tensor, TensorDict]`): The input condition.
        """

        return self.score_function_.forward(self.model, t, x, condition)

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
    
