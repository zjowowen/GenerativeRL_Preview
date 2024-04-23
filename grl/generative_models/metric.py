from typing import Callable, Union

import torch
import torch.nn as nn
import treetensor
from easydict import EasyDict
from tensordict import TensorDict
from torch.distributions import Distribution, MultivariateNormal

from grl.generative_models.diffusion_model import (
    DiffusionModel, EnergyConditionalDiffusionModel)
from grl.numerical_methods.numerical_solvers.ode_solver import (
    DictTensorODESolver, ODESolver)
from grl.numerical_methods.ode import ODE


def compute_likelihood(
        model: Union[DiffusionModel, EnergyConditionalDiffusionModel],
        x: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
        t: torch.Tensor = None,
        condition: Union[torch.Tensor, TensorDict] = None,
        using_Hutchinson_trace_estimator: bool = False,
    ) -> torch.Tensor:
    """
    Overview:
        Compute Likelihood of samples in generative model for gaussian prior.

    Arguments:
        - model (:obj:`Union[Callable, nn.Module]`): The model.
        - x (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`): The input state.
        - t (:obj:`torch.Tensor`): The input time.
        - condition (:obj:`Union[torch.Tensor, TensorDict]`): The input condition.

    Returns:
        - log_likelihood (:obj:`torch.Tensor`): The likelihood of the samples.
    """
    #TODO: Add support for EnergyConditionalDiffusionModel; Add support for t; Add support for treetensor.torch.Tensor

    model_drift = model.diffusion_process.forward_ode(function=model.model, function_type=model.model_type, condition=condition).drift

    def divergence_bf(dx, x):
        sum_diag = 0.
        for i in range(x.shape[1]):
            sum_diag += torch.autograd.grad(dx[:, i].sum(), x, create_graph=True)[0].contiguous()[:, i].contiguous()
        return sum_diag.contiguous()

    def divergence_approx(dx, x, e):
        e_dzdx = torch.autograd.grad(dx, x, e, create_graph=True)[0]
        e_dzdx_e = e_dzdx * e
        approx_tr_dzdx = e_dzdx_e.view(x.shape[0], -1).sum(dim=1)
        return approx_tr_dzdx

    def composite_drift(t, x):
        # where x is actually x0_and_diff_logp, which is a dict containing x and logp_xt_minus_logp_x0
        with torch.set_grad_enabled(True):
            t = t.detach()
            x_t = x['x'].detach()
            logp_xt_minus_logp_x0 = x['logp_xt_minus_logp_x0']
            
            x_t_shape = x_t.shape
            x_t_flatten = x_t.reshape(x_t_shape[0], -1).detach()
            x_t_flatten.requires_grad = True

            x_t_reshape = x_t_flatten.reshape(x_t_shape)

            t.requires_grad = True

            dx = model_drift(t, x_t_reshape)

            dx_flatten = dx.reshape(x_t_shape[0], -1)

            if using_Hutchinson_trace_estimator:
                noise = torch.randn_like(x_t_flatten, device=x_t_flatten.device)
                logp_drift = - divergence_approx(dx_flatten, x_t_flatten, noise)
            else:
                logp_drift = - divergence_bf(dx_flatten, x_t_flatten)

            delta_x = treetensor.torch.tensor({'x': dx, 'logp_xt_minus_logp_x0': logp_drift}, device=x_t.device)
            return delta_x

    # x.shape = [batch_size, state_dim]
    x0_and_diff_logp = treetensor.torch.tensor({'x': x, 'logp_xt_minus_logp_x0': torch.zeros(x.shape[0])}, device=x.device)
    
    eps = 1e-3
    t_span = torch.linspace(eps, 1.0, 1000).to(x.device)

    
    solver = DictTensorODESolver(library="torchdyn", dict_type="treetensor")

    x1_and_logp1 = solver.integrate(
        drift=composite_drift,
        x0=x0_and_diff_logp,
        t_span=t_span,
        batch_size=x.shape[0],
        x_size=x0_and_diff_logp.shape
    )[-1]

    logp_x1_minus_logp_x0 = x1_and_logp1['logp_xt_minus_logp_x0']
    x1 = x1_and_logp1['x']
    x1_1d = x1.reshape(x1.shape[0], -1)
    logp_x1 = MultivariateNormal(loc=torch.zeros_like(x1_1d, device=x1_1d.device), covariance_matrix=torch.eye(x1_1d.shape[-1], device=x1_1d.device)).log_prob(x1_1d)

    log_likelihood = logp_x1 - logp_x1_minus_logp_x0

    return log_likelihood
