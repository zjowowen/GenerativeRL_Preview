import copy
import math
import os
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import d4rl
import gym
import numpy as np
import torch
import torch.nn as nn
from easydict import EasyDict
from rich.progress import Progress, track
from tensordict import TensorDict
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

import wandb
from grl.agents.srpo import SRPOAgent
from grl.datasets import create_dataset
from grl.datasets.d4rl import D4RLDataset
from grl.generative_models.diffusion_model.diffusion_model import DiffusionModel
from grl.generative_models.sro import SRPOConditionalDiffusionModel
from grl.neural_network import MultiLayerPerceptron, register_module
from grl.rl_modules.simulators import create_simulator
from grl.rl_modules.value_network.q_network import DoubleQNetwork
from grl.utils import set_seed
from grl.utils.config import merge_two_dicts_into_newone
from grl.utils.log import log


class CONCATMLP(nn.Module):
    def __init__(self, state_dim, action_dim, t_dim):
        super().__init__()
        self.t_cond = MultiLayerPerceptron(
            hidden_sizes=[t_dim, t_dim * 2],
            output_size=t_dim,
            activation="mish",
        )
        self.main = MultiLayerPerceptron(
            hidden_sizes=[state_dim + action_dim + t_dim, 256, 256, 256],
            output_size=action_dim,
            activation="mish",
        )

    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        condition: torch.Tensor = None,
    ) -> torch.Tensor:

        embed = self.t_cond(t)
        result = self.main(torch.cat([x, embed, condition], dim=-1))
        return result


register_module(CONCATMLP, "CONCATMLP")


# class SinusoidalPosEmb(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.dim = dim

#     def forward(self, x):
#         device = x.device
#         half_dim = self.dim // 2
#         emb = math.log(10000) / (half_dim - 1)
#         emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
#         emb = x[:, None] * emb[None, :]
#         emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
#         return emb


# register_module(SinusoidalPosEmb, "SinusoidalPosEmb")


class CPSCritic(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.q = DoubleQNetwork(config.DoubleQNetwork)
        self.q_target = copy.deepcopy(self.q).requires_grad_(False)

    def q_loss(
        self,
        action: Union[torch.Tensor, TensorDict],
        state: Union[torch.Tensor, TensorDict],
        reward: Union[torch.Tensor, TensorDict],
        next_state: Union[torch.Tensor, TensorDict],
        next_action: Union[torch.Tensor, TensorDict],
        done: Union[torch.Tensor, TensorDict],
        discount_factor: float,
    ) -> torch.Tensor:
        q1, q2 = self.q.compute_double_q(action, state)
        with torch.no_grad():
            target_q = self.q_target.compute_mininum_q(next_action, next_state)
        target_q = (reward + (1.0 - done) * discount_factor * target_q).detach()
        q_loss = torch.nn.functional.mse_loss(
            q1, target_q
        ) + torch.nn.functional.mse_loss(q2, target_q)
        return q_loss


class CPSPolicy(nn.Module):
    def __init__(self, config: EasyDict):
        super().__init__()
        self.config = config
        self.device = config.device
        self.critic = CPSCritic(config.critic)
        self.diffusionmodel = DiffusionModel(config.diffusion_model)
        self.target_model = copy.deepcopy(self.diffusionmodel)
        self.LA = torch.tensor(config.LA, dtype=torch.float).to(self.device)  # LA
        self.LA.requires_grad = True
        self.target_kl = config.target_kl
        self.LA_min = config.LA_min
        self.LA_max = config.LA_max
        self.max_action = 1.0

    def q_loss(
        self,
        action: Union[torch.Tensor, TensorDict],
        state: Union[torch.Tensor, TensorDict],
        reward: Union[torch.Tensor, TensorDict],
        next_state: Union[torch.Tensor, TensorDict],
        next_action: Union[torch.Tensor, TensorDict],
        done: Union[torch.Tensor, TensorDict],
        discount_factor: float = 1.0,
    ) -> torch.Tensor:
        q_loss = self.critic.q_loss(
            action, state, reward, next_state, next_action, done, discount_factor
        )
        return q_loss

    def policy_loss(
        self,
        action: Union[torch.Tensor, TensorDict],
        state: Union[torch.Tensor, TensorDict],
        reward: Union[torch.Tensor, TensorDict],
        next_state: Union[torch.Tensor, TensorDict],
        next_action: Union[torch.Tensor, TensorDict],
        done: Union[torch.Tensor, TensorDict],
    ) -> torch.Tensor:
        kl_loss = self.diffusionmodel.score_matching_loss(action, state)
        q1, q2 = self.critic.q.compute_double_q(next_action, state)
        if np.random.uniform() > 0.5:
            q_loss = -q1.mean() / q2.abs().mean().detach()
        else:
            q_loss = -q2.mean() / q1.abs().mean().detach()
        policy_loss = (
            self.LA.clamp(self.LA_min, self.LA_max).detach() * kl_loss + q_loss
        )
        LA_loss = (self.target_kl - kl_loss).detach() * self.LA
        return policy_loss, LA_loss

    def sample(self, state):
        state_rpt = torch.repeat_interleave(state, repeats=50, dim=0)
        action = self.sample_action(state_rpt, with_grad=False)
        q_value = self.critic.q_target(state_rpt, action).flatten()
        idx = torch.multinomial(torch.nn.functional.softmax(q_value), 1)
        return action[idx]

    def sample_action(self, state, with_grad=False):
        t_span = torch.linspace(0.0, 1.0, 5).to(self.device)
        action = self.diffusionmodel.sample(
            t_span=t_span,
            condition=state,
            with_grad=with_grad,
            solver_config=self.config.diffusion_model.solver,
        ).clamp(
            -self.max_action,
            self.max_action,
        )
        return action

    def sample_target_action(self, state, with_grad=False):
        t_span = torch.linspace(0.0, 1.0, 5).to(self.device)
        action = self.target_model.sample(
            t_span=t_span,
            condition=state,
            with_grad=with_grad,
            solver_config=self.config.diffusion_model.solver,
        ).clamp(
            -self.max_action,
            self.max_action,
        )
        return action


class CPSAlgorithm:

    def __init__(
        self,
        config: EasyDict = None,
        simulator=None,
        dataset: D4RLDataset = None,
        model: Union[torch.nn.Module, torch.nn.ModuleDict] = None,
    ):
        """
        Overview:
            Initialize the QGPO algorithm.
        Arguments:
            config (:obj:`EasyDict`): The configuration , which must contain the following keys:
                train (:obj:`EasyDict`): The training configuration.
                deploy (:obj:`EasyDict`): The deployment configuration.
            simulator (:obj:`object`): The environment simulator.
            dataset (:obj:`QGPODataset`): The dataset.
            model (:obj:`Union[torch.nn.Module, torch.nn.ModuleDict]`): The model.
        Interface:
            ``__init__``, ``train``, ``deploy``
        """
        self.config = config
        self.simulator = simulator
        self.dataset = dataset

        # ---------------------------------------
        # Customized model initialization code ↓
        # ---------------------------------------

        self.model = model if model is not None else torch.nn.ModuleDict()

        # ---------------------------------------
        # Customized model initialization code ↑
        # ---------------------------------------

    def train(self, config: EasyDict = None):
        """
            Overview:
                Train the model using the given configuration. \
                A weight-and-bias run will be created automatically when this function is called.
            Arguments:
                config (:obj:`EasyDict`): The training configuration.
            """
        set_seed(self.config.deploy.env["seed"])

        config = (
            merge_two_dicts_into_newone(
                self.config.train if hasattr(self.config, "train") else EasyDict(),
                config,
            )
            if config is not None
            else self.config.train
        )

        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
        directory_path = os.path.join(
            f"./{config.project}",
            formatted_time,
        )
        os.makedirs(directory_path, exist_ok=True)

        with wandb.init(
            project=(
                config.project if hasattr(config, "project") else __class__.__name__
            ),
            **config.wandb if hasattr(config, "wandb") else {},
        ) as wandb_run:
            config = merge_two_dicts_into_newone(EasyDict(wandb_run.config), config)

            wandb_run.config.update(config)
            self.config.train = config

            self.simulator = (
                create_simulator(config.simulator)
                if hasattr(config, "simulator")
                else self.simulator
            )
            self.dataset = (
                create_dataset(config.dataset)
                if hasattr(config, "dataset")
                else self.dataset
            )

            # ---------------------------------------
            # Customized model initialization code ↓
            # ---------------------------------------
            if hasattr(config.model, "CPSPolicy"):
                self.model["CPSPolicy"] = CPSPolicy(config.model.CPSPolicy)
                self.model["CPSPolicy"].to(config.model.CPSPolicy.device)
                if torch.__version__ >= "2.0.0":
                    self.model["CPSPolicy"] = torch.compile(self.model["CPSPolicy"])
            # ---------------------------------------
            # test model ↓
            # ---------------------------------------
            assert isinstance(
                self.model, (torch.nn.Module, torch.nn.ModuleDict)
            ), "self.model must be torch.nn.Module or torch.nn.ModuleDict."
            if isinstance(self.model, torch.nn.ModuleDict):
                assert (
                    "CPSPolicy" in self.model and self.model["CPSPolicy"]
                ), "self.model['CPSPolicy'] cannot be empty."
            else:  # self.model is torch.nn.Module
                assert self.model, "self.model cannot be empty."

            def get_train_data(dataloader):
                while True:
                    yield from dataloader

            def evaluate(policy_fn, train_iter):
                evaluation_results = dict()

                def policy(obs: np.ndarray) -> np.ndarray:
                    obs = torch.tensor(
                        obs, dtype=torch.float32, device=config.model.CPSPolicy.device
                    ).unsqueeze(0)
                    with torch.no_grad():
                        action = policy_fn(obs).squeeze(0).detach().cpu().numpy()
                    return action

                result = self.simulator.evaluate(
                    policy=policy,
                )[0]
                evaluation_results["evaluation/total_return"] = result["total_return"]
                evaluation_results["evaluation/total_steps"] = result["total_steps"]
                return evaluation_results

            data_generator = get_train_data(
                DataLoader(
                    self.dataset,
                    batch_size=config.parameter.behaviour_policy.batch_size,
                    shuffle=True,
                    collate_fn=None,
                )
            )
            critic_optimizer = torch.optim.Adam(
                self.model["CPSPolicy"].critic.q.parameters(),
                lr=config.parameter.critic.learning_rate,
            )
            policy_optimizer = torch.optim.Adam(
                self.model["CPSPolicy"].diffusionmodel.parameters(),
                lr=config.parameter.behaviour_policy.learning_rate,
            )
            LA_optimizer = torch.optim.Adam(
                [self.model["CPSPolicy"].LA],
                lr=config.parameter.behaviour_policy.lr_learning_rate,
            )
            critic_lr_scheduler = CosineAnnealingLR(
                critic_optimizer, T_max=config.parameter.critic.t_max, eta_min=0.0
            )
            policy_lr_scheduler = CosineAnnealingLR(
                policy_optimizer,
                T_max=config.parameter.behaviour_policy.t_max,
                eta_min=0.0,
            )
            lambda_lr_scheduler = CosineAnnealingLR(
                LA_optimizer, T_max=config.parameter.behaviour_policy.t_max, eta_min=0.0
            )
            ctiric_iter = 0
            policy_iter = 0
            for train_iter in track(
                range(config.parameter.behaviour_policy.iterations),
                description="training",
            ):
                data = next(data_generator)
                next_action = self.model["CPSPolicy"].sample_target_action(
                    data["s"], with_grad=False
                )
                q_loss = self.model["CPSPolicy"].q_loss(
                    data["a"],
                    data["s"],
                    data["r"],
                    data["s_"],
                    next_action,
                    data["d"],
                    config.parameter.critic.discount_factor,
                )
                critic_optimizer.zero_grad()
                q_loss.backward()
                # if self.grad_norm > 0:
                critic_grad_norms = nn.utils.clip_grad_norm_(
                    self.model["CPSPolicy"].critic.q.parameters(),
                    max_norm=config.parameter.critic.grad_norm,
                    norm_type=2,
                )
                critic_optimizer.step()
                ctiric_iter += 1
                if (
                    train_iter % config.parameter.behaviour_policy.update_policy_every
                    == 0
                ):
                    next_action = self.model["CPSPolicy"].sample_action(
                        data["s"], with_grad=True
                    )
                    policy_loss, LA_loss = self.model["CPSPolicy"].policy_loss(
                        data["a"],
                        data["s"],
                        data["r"],
                        data["s_"],
                        next_action,
                        data["d"],
                    )
                    policy_optimizer.zero_grad()
                    policy_loss.backward()
                    policy_grad_norms = nn.utils.clip_grad_norm_(
                        self.model["CPSPolicy"].diffusionmodel.parameters(),
                        max_norm=config.parameter.behaviour_policy.grad_norm,
                        norm_type=2,
                    )
                    policy_optimizer.step()
                    LA_optimizer.zero_grad()
                    LA_loss.backward()
                    LA_grad_norms = nn.utils.clip_grad_norm_(
                        self.model["CPSPolicy"].LA,
                        max_norm=config.parameter.behaviour_policy.grad_norm,
                        norm_type=2,
                    )
                    LA_optimizer.step()
                    wandb.log(
                        {
                            "policy_loss": policy_loss.item(),
                            "LA_loss": LA_loss.item(),
                            "policy_grad_norms": policy_grad_norms,
                            "LA_grad_norms": LA_grad_norms,
                            "policy_iter": policy_iter,
                        }
                    )
                    policy_iter += 1

                # update target network
                if (
                    train_iter % config.parameter.behaviour_policy.update_target_every
                    == 0
                ) and (
                    train_iter > config.parameter.behaviour_policy.step_start_target
                ):
                    for param, target_param in zip(
                        self.model["CPSPolicy"].diffusionmodel.parameters(),
                        self.model["CPSPolicy"].target_model.parameters(),
                    ):
                        target_param.data.copy_(
                            config.parameter.behaviour_policy.update_momentum
                            * param.data
                            + (1 - config.parameter.behaviour_policy.update_momentum)
                            * target_param.data
                        )

                for param, target_param in zip(
                    self.model["CPSPolicy"].critic.q.parameters(),
                    self.model["CPSPolicy"].critic.q_target.parameters(),
                ):
                    target_param.data.copy_(
                        config.parameter.critic.update_momentum * param.data
                        + (1 - config.parameter.critic.update_momentum)
                        * target_param.data
                    )

                if (
                    train_iter + 1
                ) % config.parameter.behaviour_policy.update_lr_every == 0:
                    policy_lr_scheduler.step()
                    critic_lr_scheduler.step()
                    lambda_lr_scheduler.step()

                # update policy
                wandb.log(
                    {
                        "q_loss": q_loss.item(),
                        "critic_grad_norms": critic_grad_norms,
                        "critic_iter": ctiric_iter,
                    }
                )

                if train_iter % config.parameter.evaluation.evaluation_interval == 0:
                    evaluation_results = evaluate(
                        self.model["CPSPolicy"].sample, train_iter
                    )
                    wandb.log(evaluation_results, commit=False)
