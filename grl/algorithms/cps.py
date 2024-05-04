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
from torch.utils.data import DataLoader

import wandb
from grl.agents.srpo import SRPOAgent
from grl.datasets import create_dataset
from grl.datasets.d4rl import D4RLDataset
from grl.generative_models.diffusion_model.diffusion_model import \
    DiffusionModel
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
            hidden_sizes=[t_dim, t_dim * 2, t_dim * 2, t_dim],
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


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


register_module(SinusoidalPosEmb, "SinusoidalPosEmb")


class CPSCritic(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.q = DoubleQNetwork(config.DoubleQNetwork)
        self.q_target = copy.deepcopy(self.q0).requires_grad_(False)

    def q_loss(self, data):
        
        q1, q2 = self.q.compute_double_q(data["action"], data["state"])
        target_q1, target_q2 = self.q_target(next_state, next_action)
                target_q = torch.min(target_q1, target_q2)
        return self.q(data)


class CPSPolicy(nn.Module):
    def __init__(self, config: EasyDict):
        super().__init__()
        self.config = config
        self.device = config.device
        self.critic = CPSCritic(config.critic)
        self.diffusionmodel = DiffusionModel(config.diffusion_model)
        self.LA = torch.tensor(LA, dtype=torch.float).to(self.device)  # LA
        self.LA.requires_grad = True

    def q_loss(self, data):
        next_state= data["next_state"]
        return self.critic(data)


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
                self.model["CPSPolicy"] = CPSPolicy(config.model.SRPOPolicy)
                self.model["CPSPolicy"].to(config.model.CPSPolicy.device)
                if torch.__version__ >= "2.0.0":
                    self.model["CPSPolicy"] = torch.compile(self.model["CPSÍPolicy"])
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

            data_generator = get_train_data(
                DataLoader(
                    self.dataset,
                    batch_size=config.parameter.behaviour_policy.batch_size,
                    shuffle=True,
                    collate_fn=None,
                )
            )

            for train_iter in track(
                range(config.parameter.behaviour_policy.iterations),
                description="training",
            ):
                data = next(data_generator)
                q_loss = self.model["CPSPolicy"].qloss(data)
                q_loss.backward()
                # if self.grad_norm > 0:
                critic_grad_norms = nn.utils.clip_grad_norm_(
                    self.critic.parameters(), max_norm=self.grad_norm, norm_type=2
                )
                self.critic_optimizer.step()
