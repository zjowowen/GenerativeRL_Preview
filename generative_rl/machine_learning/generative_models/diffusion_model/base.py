from typing import Union, List, Tuple, Dict, Any, Callable
from easydict import EasyDict
import torch
import torch.nn as nn
from tensordict import TensorDict

from generative_rl.numerical_methods.probability_path import GaussianConditionalProbabilityPath
from generative_rl.numerical_methods.numerical_solvers import get_solver
from generative_rl.numerical_methods.numerical_solvers.dpm_solver import DPMSolver
from generative_rl.numerical_methods.numerical_solvers.ode_solver import ODESolver
from generative_rl.numerical_methods.numerical_solvers.sde_solver import SDESolver
from generative_rl.numerical_methods.diffusion_process import get_diffusion_process
from generative_rl.machine_learning.generative_models.diffusion_model.random_init import gaussian_random_variable
from generative_rl.machine_learning.generative_models.diffusion_model.score_model import ScoreFunction

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

        self.gaussian_conditional_probability_path = GaussianConditionalProbabilityPath(config.gaussian_conditional_probability_path)
        self.diffusion_process = get_diffusion_process(config.diffusion_process)(self.gaussian_conditional_probability_path)
        self.score_function = ScoreFunction(config.score_function, self.gaussian_conditional_probability_path)

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
            #TODO
            pass
        elif isinstance(solver, ODESolver):
            #TODO: make it compatible with TensorDict
            if not hasattr(self, "t_span") is None:
                self.t_span = torch.linspace(0, self.gaussian_conditional_probability_path.t_max, 2).to(self.device)
            x = self.gaussian_generator(batch_size=batch_size)
            if with_grad:
                data = solver.integrate(
                    drift=self.diffusion_process.reverse_ode(score_function=self.score_function, condition=condition).drift,
                    x0=x,
                    t_span=self.t_span,
                )[1]
            else:
                with torch.no_grad():
                    data = solver.integrate(
                        drift=self.diffusion_process.reverse_ode(score_function=self.score_function, condition=condition).drift,
                        x0=x,
                        t_span=self.t_span,
                    )[1]
        elif isinstance(solver, SDESolver):
            #TODO: make it compatible with TensorDict
            #TODO: validate the implementation
            if not hasattr(self, "t_span") is None:
                self.t_span = torch.linspace(0, self.gaussian_conditional_probability_path.t_max, 2).to(self.device)
            x = self.gaussian_generator(batch_size=batch_size)
            sde = self.diffusion_process.reverse_sde(score_function=self.score_function, condition=condition)
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

    def sample_forward_process(
            self,
            t_span: torch.Tensor,
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
            #TODO
            pass
        elif isinstance(solver, ODESolver):
            #TODO: make it compatible with TensorDict
            x = self.gaussian_generator(batch_size=batch_size)
            if with_grad:
                data = solver.integrate(
                    drift=self.diffusion_process.reverse_ode(score_function=self.score_function, condition=condition).drift,
                    x0=x,
                    t_span=t_span,
                )
            else:
                with torch.no_grad():
                    data = solver.integrate(
                        drift=self.diffusion_process.reverse_ode(score_function=self.score_function, condition=condition).drift,
                        x0=x,
                        t_span=t_span,
                    )
        elif isinstance(solver, SDESolver):
            #TODO: make it compatible with TensorDict
            #TODO: validate the implementation
            x = self.gaussian_generator(batch_size=batch_size)
            sde = self.diffusion_process.reverse_sde(score_function=self.score_function, condition=condition)
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

        return self.score_function.score_matching_loss(x, condition)
