import copy
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
from torch.utils.data import DataLoader

import wandb

from grl.datasets import create_dataset
from grl.datasets.qgpo import QGPODataset
from grl.neural_network import MultiLayerPerceptron
from grl.rl_modules.simulators import create_simulator
from grl.rl_modules.value_network.q_network import DoubleQNetwork
from grl.utils import set_seed
from grl.utils.config import merge_two_dicts_into_newone
from grl.utils.log import log
from grl.generative_models.diffusion_model.diffusion_model import DiffusionModel


# class Dirac_Policy(nn.Module):
#     def __init__(self, action_dim, state_dim, layer=2):
#         super().__init__()
#         self.net = MultiLayerPerceptron(
#             hidden_sizes=[state_dim] + [256 for _ in range(layer)],
#             output_size=action_dim,
#             activation="relu",
#             final_activation="tanh",
#         )

#     def forward(self, state):
#         return self.net(state)

#     def select_actions(self, state):
#         return self(state)


def asymmetric_l2_loss(u, tau):
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)


class ValueFunction(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.v = MultiLayerPerceptron(
            hidden_sizes=[state_dim, 256, 256],
            output_size=1,
            activation="relu",
        )

    def forward(self, state):
        return self.v(state)


class CPSCritic(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.q = DoubleQNetwork(config.DoubleQNetwork)
        self.q_target = copy.deepcopy(self.q).requires_grad_(False)

    def q_loss(
        self,
        data,
        discount_factor: float,
    ) -> torch.Tensor:
        q1, q2 = self.q.compute_double_q(data["a"], data["s"])
        with torch.no_grad():
            target_q = self.q_target.compute_mininum_q(data["a_"], data["s_"])
        target_q = (
            data["r"] + (1.0 - data["done"]) * discount_factor * target_q
        ).detach()
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
        self.cps = DiffusionModel(config.diffusion_model)

    def forward(
        self, state: Union[torch.Tensor, TensorDict]
    ) -> Union[torch.Tensor, TensorDict]:
        """
        Overview:
            Return the output of QGPO policy, which is the action conditioned on the state.
        Arguments:
            state (:obj:`Union[torch.Tensor, TensorDict]`): The input state.
        Returns:
            action (:obj:`Union[torch.Tensor, TensorDict]`): The output action.
        """
        return self.cps.sample(conditio=state)

    def behaviour_policy_loss(
        self,
        action: Union[torch.Tensor, TensorDict],
        state: Union[torch.Tensor, TensorDict],
    ):
        """
        Overview:
            Calculate the behaviour policy loss.
        Arguments:
            action (:obj:`torch.Tensor`): The input action.
            state (:obj:`torch.Tensor`): The input state.
        """
        return self.cps.score_matching_loss(action, state)

    def q_loss(
        self,
        data,
        discount_factor: float = 1.0,
    ) -> torch.Tensor:
        """
        Overview:
            Calculate the Q loss.
        Arguments:
            action (:obj:`torch.Tensor`): The input action.
            state (:obj:`torch.Tensor`): The input state.
            reward (:obj:`torch.Tensor`): The input reward.
            next_state (:obj:`torch.Tensor`): The input next state.
            done (:obj:`torch.Tensor`): The input done.
            fake_next_action (:obj:`torch.Tensor`): The input fake next action.
            discount_factor (:obj:`float`): The discount factor.
        """
        loss = self.critic.q_loss(data, discount_factor)
        return loss


class CPSAlgorithm:

    def __init__(
        self,
        config: EasyDict = None,
        simulator=None,
        dataset: QGPODataset = None,
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
            # ---------------------------------------
            # Customized model initialization code ↑
            # ---------------------------------------

            # ---------------------------------------
            # Customized training code ↓
            # ---------------------------------------

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

            behaviour_model_optimizer = torch.optim.Adam(
                self.model["CPSPolicy"].sro.diffusion_model.model.parameters(),
                lr=config.parameter.behaviour_policy.learning_rate,
            )

            # checkpoint = torch.load(
            #     "/root/github/GenerativeRL_Preview/grl_pipelines/d4rl-halfcheetah-CPS/2024-04-17 06:22:21/checkpoint_diffusion_600000.pt"
            # )
            # self.model["CPSPolicy"].sro.diffusion_model.model.load_state_dict(
            #     checkpoint["diffusion_model"]
            # )
            # behaviour_model_optimizer.load_state_dict(
            #     checkpoint["behaviour_model_optimizer"]
            # )

            for train_diffusion_iter in track(
                range(config.parameter.behaviour_policy.iterations),
                description="Behaviour policy training",
            ):
                data = next(data_generator)
                # data["s"].shape  torch.Size([2048, 17])   data["a"].shape torch.Size([2048, 6])  data["r"].shape torch.Size([2048, 1])
                behaviour_model_training_loss = self.model[
                    "CPSPolicy"
                ].behaviour_policy_loss(data["a"], data["s"])
                behaviour_model_optimizer.zero_grad()
                behaviour_model_training_loss.backward()
                behaviour_model_optimizer.step()

                if (
                    train_diffusion_iter == 0
                    or (train_diffusion_iter + 1)
                    % config.parameter.evaluation.evaluation_interval
                    == 0
                ):
                    evaluation_results = evaluate(
                        self.model["CPSPolicy"], train_iter=train_diffusion_iter
                    )
                    wandb_run.log(data=evaluation_results, commit=False)

                wandb_run.log(
                    data=dict(
                        train_diffusion_iter=train_diffusion_iter,
                        behaviour_model_training_loss=behaviour_model_training_loss.item(),
                    ),
                    commit=True,
                )

            if train_diffusion_iter == config.parameter.behaviour_policy.iterations - 1:
                file_path = os.path.join(
                    directory_path, f"checkpoint_diffusion_{train_diffusion_iter+1}.pt"
                )
                torch.save(
                    dict(
                        diffusion_model=self.model[
                            "CPSPolicy"
                        ].sro.diffusion_model.model.state_dict(),
                        behaviour_model_optimizer=behaviour_model_optimizer.state_dict(),
                        diffusion_iteration=train_diffusion_iter + 1,
                    ),
                    f=file_path,
                )

            # q_optimizer = torch.optim.Adam(
            #     self.model["CPSPolicy"].critic.q0.parameters(),
            #     lr=config.parameter.critic.learning_rate,
            # )
            # v_optimizer = torch.optim.Adam(
            #     self.model["CPSPolicy"].critic.vf.parameters(),
            #     lr=config.parameter.critic.learning_rate,
            # )

            # # checkpoint = torch.load(
            # #     "/root/github/GenerativeRL_Preview/grl_pipelines/d4rl-halfcheetah-CPS/2024-04-17 06:22:21/checkpoint_critic_600000.pt"
            # # )
            # # self.model["CPSPolicy"].critic.q0.load_state_dict(checkpoint["q_model"])
            # # self.model["CPSPolicy"].critic.vf.load_state_dict(checkpoint["v_model"])
            # data_generator = get_train_data(
            #     DataLoader(
            #         self.dataset,
            #         batch_size=config.parameter.critic.batch_size,
            #         shuffle=True,
            #         collate_fn=None,
            #     )
            # )

            # for train_critic_iter in track(
            #     range(config.parameter.critic.iterations), description="Critic training"
            # ):
            #     data = next(data_generator)

            #     v_loss, next_v = self.model["CPSPolicy"].v_loss(
            #         data,
            #         config.parameter.critic.tau,
            #     )
            #     v_optimizer.zero_grad(set_to_none=True)
            #     v_loss.backward()
            #     v_optimizer.step()

            #     q_loss = self.model["CPSPolicy"].q_loss(
            #         data,
            #         next_v,
            #         config.parameter.critic.discount_factor,
            #     )
            #     q_optimizer.zero_grad(set_to_none=True)
            #     q_loss.backward()
            #     q_optimizer.step()

            #     # Update target
            #     for param, target_param in zip(
            #         self.model["CPSPolicy"].critic.q0.parameters(),
            #         self.model["CPSPolicy"].critic.q0_target.parameters(),
            #     ):
            #         target_param.data.copy_(
            #             config.parameter.critic.moment * param.data
            #             + (1 - config.parameter.critic.moment) * target_param.data
            #         )

            #     wandb_run.log(
            #         data=dict(
            #             train_critic_iter=train_critic_iter,
            #             q_loss=q_loss.item(),
            #             v_loss=v_loss.item(),
            #         ),
            #         commit=True,
            #     )

            # if train_critic_iter == config.parameter.critic.iterations - 1:
            #     file_path = os.path.join(
            #         directory_path, f"checkpoint_critic_{train_critic_iter+1}.pt"
            #     )
            #     torch.save(
            #         dict(
            #             q_model=self.model["CPSPolicy"].critic.q0.state_dict(),
            #             v_model=self.model["CPSPolicy"].critic.vf.state_dict(),
            #             q_optimizer=q_optimizer.state_dict(),
            #             v_optimizer=v_optimizer.state_dict(),
            #             critic_iteration=train_critic_iter + 1,
            #         ),
            #         f=file_path,
            #     )

            # data_generator = get_train_data(
            #     DataLoader(
            #         self.dataset,
            #         batch_size=config.parameter.actor.batch_size,
            #         shuffle=True,
            #         collate_fn=None,
            #     )
            # )
            # CPS_policy_optimizer = torch.optim.Adam(
            #     self.model["CPSPolicy"].deter_policy.parameters(), lr=3e-4
            # )
            # CPS_policy_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            #     CPS_policy_optimizer,
            #     T_max=config.parameter.actor.iterations,
            #     eta_min=0.0,
            # )
            # for train_policy_iter in track(
            #     range(config.parameter.actor.iterations), description="actor training"
            # ):
            #     data = next(data_generator)
            #     self.model["CPSPolicy"].sro.diffusion_model.model.eval()
            #     actor_loss, q = self.model["CPSPolicy"].CPS_actor_loss(data)
            #     actor_loss = actor_loss.sum(-1).mean()
            #     CPS_policy_optimizer.zero_grad(set_to_none=True)
            #     actor_loss.backward()
            #     CPS_policy_optimizer.step()
            #     CPS_policy_lr_scheduler.step()
            #     self.model["CPSPolicy"].sro.diffusion_model.model.train()
            #     wandb_run.log(
            #         data=dict(
            #             train_policy_iter=train_policy_iter,
            #             actor_loss=actor_loss,
            #             q=q,
            #         ),
            #         commit=True,
            #     )

            #     if (
            #         train_policy_iter == 0
            #         or (train_policy_iter + 1)
            #         % config.parameter.evaluation.evaluation_interval
            #         == 0
            #     ):
            #         evaluation_results = evaluate(
            #             self.model["CPSPolicy"], train_iter=train_policy_iter
            #         )

            #         wandb_run.log(
            #             data=evaluation_results,
            #             commit=False,
            #         )

            #     if train_policy_iter == config.parameter.actor.iterations - 1:
            #         file_path = os.path.join(
            #             directory_path, f"checkpoint_policy_{train_policy_iter+1}.pt"
            #         )
            #         torch.save(
            #             dict(
            #                 actor_model=self.model[
            #                     "CPSPolicy"
            #                 ].deter_policy.state_dict(),
            #                 actor_optimizer=CPS_policy_optimizer.state_dict(),
            #                 policy_iteration=train_policy_iter + 1,
            #             ),
            #             f=file_path,
            #         )

            # ---------------------------------------
            # Customized training code ↑
            # ---------------------------------------

        wandb.finish()

    # def deploy(self, config: EasyDict = None) -> CPSAgent:

    #     if config is not None:
    #         config = merge_two_dicts_into_newone(self.config.deploy, config)
    #     else:
    #         config = self.config.deploy

    #     return CPSAgent(
    #         config=config,
    #         model=torch.nn.ModuleDict(
    #             {
    #                 "CPSPolicy": self.deter_policy.select_actions,
    #             }
    #         ),
    #     )
