#############################################################
# This QGPO model is a modification implementation from https://github.com/ChenDRAG/CEP-energy-guided-diffusion
#############################################################

import os
import copy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from easydict import EasyDict
from rich.progress import Progress, track
from tensordict import TensorDict
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

import wandb
from grl.agents.qgpo import QGPOAgent
from grl.datasets import create_dataset
from grl.datasets.qgpo import QGPODataset
from grl.generative_models.diffusion_model.energy_conditional_diffusion_model import (
    EnergyConditionalDiffusionModel,
)
from grl.rl_modules.simulators import create_simulator
from grl.rl_modules.value_network.q_network import DoubleQNetwork
from grl.utils.config import merge_two_dicts_into_newone
from grl.utils.log import log
from grl.utils.statistics import sort_files_by_criteria


class QGPOCritic(nn.Module):
    """
    Overview:
        Critic network for QGPO algorithm.
    Interfaces:
        ``__init__``, ``forward``
    """

    def __init__(self, config: EasyDict):
        """
        Overview:
            Initialization of QGPO critic network.
        Arguments:
            config (:obj:`EasyDict`): The configuration dict.
        """

        super().__init__()
        self.config = config
        self.q_alpha = config.q_alpha
        self.q = DoubleQNetwork(config.DoubleQNetwork)
        self.q_target = copy.deepcopy(self.q).requires_grad_(False)

    def forward(
        self,
        action: Union[torch.Tensor, TensorDict],
        state: Union[torch.Tensor, TensorDict] = None,
    ) -> torch.Tensor:
        """
        Overview:
            Return the output of QGPO critic.
        Arguments:
            action (:obj:`torch.Tensor`): The input action.
            state (:obj:`torch.Tensor`): The input state.
        """

        return self.q(action, state)

    def compute_double_q(
        self,
        action: Union[torch.Tensor, TensorDict],
        state: Union[torch.Tensor, TensorDict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Overview:
            Return the output of two Q networks.
        Arguments:
            action (:obj:`Union[torch.Tensor, TensorDict]`): The input action.
            state (:obj:`Union[torch.Tensor, TensorDict]`): The input state.
        Returns:
            q1 (:obj:`Union[torch.Tensor, TensorDict]`): The output of the first Q network.
            q2 (:obj:`Union[torch.Tensor, TensorDict]`): The output of the second Q network.
        """
        return self.q.compute_double_q(action, state)

    def q_loss(
        self,
        action: Union[torch.Tensor, TensorDict],
        state: Union[torch.Tensor, TensorDict],
        reward: Union[torch.Tensor, TensorDict],
        next_state: Union[torch.Tensor, TensorDict],
        done: Union[torch.Tensor, TensorDict],
        fake_next_action: Union[torch.Tensor, TensorDict],
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
        with torch.no_grad():
            softmax = nn.Softmax(dim=1)
            next_energy = (
                self.q_target(
                    fake_next_action,
                    torch.stack([next_state] * fake_next_action.shape[1], axis=1),
                )
                .detach()
                .squeeze(dim=-1)
            )
            next_v = torch.sum(
                softmax(self.q_alpha * next_energy) * next_energy, dim=-1, keepdim=True
            )
        # Update Q function
        targets = reward + (1.0 - done.float()) * discount_factor * next_v.detach()
        q0, q1 = self.q.compute_double_q(action, state)
        q_loss = (
            torch.nn.functional.mse_loss(q0, targets)
            + torch.nn.functional.mse_loss(q1, targets)
        ) / 2
        return q_loss


class QGPOPolicy(nn.Module):

    def __init__(self, config: EasyDict):
        super().__init__()
        self.config = config
        self.device = config.device

        self.critic = QGPOCritic(config.critic)
        self.diffusion_model = EnergyConditionalDiffusionModel(
            config.diffusion_model, energy_model=self.critic
        )

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
        return self.sample(state)

    def sample(
        self,
        state: Union[torch.Tensor, TensorDict],
        batch_size: Union[torch.Size, int, Tuple[int], List[int]] = None,
        guidance_scale: Union[torch.Tensor, float] = torch.tensor(1.0),
        solver_config: EasyDict = None,
        t_span: torch.Tensor = None,
    ) -> Union[torch.Tensor, TensorDict]:
        """
        Overview:
            Return the output of QGPO policy, which is the action conditioned on the state.
        Arguments:
            state (:obj:`Union[torch.Tensor, TensorDict]`): The input state.
            guidance_scale (:obj:`Union[torch.Tensor, float]`): The guidance scale.
            solver_config (:obj:`EasyDict`): The configuration for the ODE solver.
            t_span (:obj:`torch.Tensor`): The time span for the ODE solver or SDE solver.
        Returns:
            action (:obj:`Union[torch.Tensor, TensorDict]`): The output action.
        """
        return self.diffusion_model.sample(
            t_span=t_span,
            condition=state,
            batch_size=batch_size,
            guidance_scale=guidance_scale,
            with_grad=False,
            solver_config=solver_config,
        )

    def behaviour_policy_sample(
        self,
        state: Union[torch.Tensor, TensorDict],
        batch_size: Union[torch.Size, int, Tuple[int], List[int]] = None,
        solver_config: EasyDict = None,
        t_span: torch.Tensor = None,
    ) -> Union[torch.Tensor, TensorDict]:
        """
        Overview:
            Return the output of behaviour policy, which is the action conditioned on the state.
        Arguments:
            state (:obj:`Union[torch.Tensor, TensorDict]`): The input state.
            solver_config (:obj:`EasyDict`): The configuration for the ODE solver.
            t_span (:obj:`torch.Tensor`): The time span for the ODE solver or SDE solver.
        Returns:
            action (:obj:`Union[torch.Tensor, TensorDict]`): The output action.
        """
        return self.diffusion_model.sample_without_energy_guidance(
            t_span=t_span,
            condition=state,
            batch_size=batch_size,
            solver_config=solver_config,
        )

    def compute_q(
        self,
        state: Union[torch.Tensor, TensorDict],
        action: Union[torch.Tensor, TensorDict],
    ) -> torch.Tensor:
        """
        Overview:
            Calculate the Q value.
        Arguments:
            state (:obj:`Union[torch.Tensor, TensorDict]`): The input state.
            action (:obj:`Union[torch.Tensor, TensorDict]`): The input action.
        Returns:
            q (:obj:`torch.Tensor`): The Q value.
        """

        return self.critic(action, state)

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

        return self.diffusion_model.score_matching_loss(
            action, state, weighting_scheme="vanilla"
        )

    def energy_guidance_loss(
        self,
        state: Union[torch.Tensor, TensorDict],
        fake_next_action: Union[torch.Tensor, TensorDict],
    ) -> torch.Tensor:
        """
        Overview:
            Calculate the energy guidance loss of QGPO.
        Arguments:
            state (:obj:`Union[torch.Tensor, TensorDict]`): The input state.
            fake_next_action (:obj:`Union[torch.Tensor, TensorDict]`): The input fake next action.
        """

        return self.diffusion_model.energy_guidance_loss(fake_next_action, state)

    def q_loss(
        self,
        action: Union[torch.Tensor, TensorDict],
        state: Union[torch.Tensor, TensorDict],
        reward: Union[torch.Tensor, TensorDict],
        next_state: Union[torch.Tensor, TensorDict],
        done: Union[torch.Tensor, TensorDict],
        fake_next_action: Union[torch.Tensor, TensorDict],
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
        return self.critic.q_loss(
            action, state, reward, next_state, done, fake_next_action, discount_factor
        )


class QGPOAlgorithm:

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

        if model is not None:
            self.model = model
            self.behaviour_policy_train_epoch = 0
            self.critic_train_epoch = 0
            self.guided_policy_train_epoch = 0
        else:
            self.model = torch.nn.ModuleDict()
            config = self.config.train
            assert hasattr(config.model, "QGPOPolicy")

            self.model["QGPOPolicy"] = QGPOPolicy(config.model.QGPOPolicy)
            self.model["QGPOPolicy"].to(config.model.QGPOPolicy.device)
            if torch.__version__ >= "2.0.0":
                self.model["QGPOPolicy"] = torch.compile(self.model["QGPOPolicy"])

            if (
                hasattr(config.parameter, "checkpoint_path")
                and config.parameter.checkpoint_path is not None
            ):
                if not os.path.exists(config.parameter.checkpoint_path):
                    log.warning(
                        f"Checkpoint path {config.parameter.checkpoint_path} does not exist"
                    )
                    self.behaviour_policy_train_epoch = 0
                    self.critic_train_epoch = 0
                    self.guided_policy_train_epoch = 0
                else:
                    base_model_files = sort_files_by_criteria(
                        folder_path=config.parameter.checkpoint_path,
                        start_string="basemodel_",
                        end_string=".pt",
                    )
                    if len(base_model_files) == 0:
                        self.behaviour_policy_train_epoch = 0
                        log.warning(
                            f"No basemodel file found in {config.parameter.checkpoint_path}"
                        )
                    else:
                        checkpoint = torch.load(
                            os.path.join(
                                config.parameter.checkpoint_path,
                                base_model_files[0],
                            ),
                            map_location="cpu",
                        )
                        self.model["QGPOPolicy"].diffusion_model.model.load_state_dict(
                            checkpoint["base_model"]
                        )
                        self.behaviour_policy_train_epoch = checkpoint.get(
                            "behaviour_policy_train_epoch", 0
                        )
                        log.info(f"Loaded base model from {base_model_files[0]}")

                    guided_model_files = sort_files_by_criteria(
                        folder_path=config.parameter.checkpoint_path,
                        start_string="guidedmodel_",
                        end_string=".pt",
                    )
                    if len(guided_model_files) == 0:
                        self.guided_policy_train_epoch = 0
                        log.warning(
                            f"No guidedmodel file found in {config.parameter.checkpoint_path}"
                        )
                    else:
                        checkpoint = torch.load(
                            os.path.join(
                                config.parameter.checkpoint_path,
                                guided_model_files[0],
                            ),
                            map_location="cpu",
                        )
                        self.model[
                            "QGPOPolicy"
                        ].diffusion_model.energy_guidance.model.load_state_dict(
                            checkpoint["guided_model"]
                        )
                        self.guided_policy_train_epoch = checkpoint.get(
                            "guided_policy_train_epoch", 0
                        )
                        log.info(f"Loaded guided model from {guided_model_files[0]}")

                    critic_model_files = sort_files_by_criteria(
                        folder_path=config.parameter.checkpoint_path,
                        start_string="critic_",
                        end_string=".pt",
                    )
                    if len(critic_model_files) == 0:
                        self.critic_train_epoch = 0
                        log.warning(
                            f"No criticmodel file found in {config.parameter.checkpoint_path}"
                        )
                    else:
                        checkpoint = torch.load(
                            os.path.join(
                                config.parameter.checkpoint_path,
                                critic_model_files[0],
                            ),
                            map_location="cpu",
                        )
                        self.model["QGPOPolicy"].critic.load_state_dict(
                            checkpoint["critic_model"]
                        )
                        self.critic_train_epoch = checkpoint.get(
                            "critic_train_epoch", 0
                        )
                        log.info(f"Loaded critic model from {critic_model_files[0]}")

    def train(self, config: EasyDict = None):
        """
        Overview:
            Train the model using the given configuration. \
            A weight-and-bias run will be created automatically when this function is called.
        Arguments:
            config (:obj:`EasyDict`): The training configuration.
        """

        config = (
            merge_two_dicts_into_newone(
                self.config.train if hasattr(self.config, "train") else EasyDict(),
                config,
            )
            if config is not None
            else self.config.train
        )

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

            # if hasattr(config.model, "QGPOPolicy"):
            #     self.model["QGPOPolicy"] = QGPOPolicy(config.model.QGPOPolicy)
            #     self.model["QGPOPolicy"].to(config.model.QGPOPolicy.device)
            #     if torch.__version__ >= "2.0.0":
            #         self.model["QGPOPolicy"] = torch.compile(self.model["QGPOPolicy"])

            # ---------------------------------------
            # Customized model initialization code ↑
            # ---------------------------------------

            def save_checkpoint(model, iteration=None, model_type=False):
                if iteration == None:
                    iteration = 0
                if model_type == "base_model":
                    if (
                        hasattr(config.parameter, "checkpoint_path")
                        and config.parameter.checkpoint_path is not None
                    ):
                        if not os.path.exists(config.parameter.checkpoint_path):
                            os.makedirs(config.parameter.checkpoint_path)
                        torch.save(
                            dict(
                                base_model=model[
                                    "QGPOPolicy"
                                ].diffusion_model.model.state_dict(),
                                behaviour_policy_train_epoch=self.behaviour_policy_train_epoch,
                                behaviour_policy_train_iter=iteration,
                            ),
                            f=os.path.join(
                                config.parameter.checkpoint_path,
                                f"basemodel_{self.behaviour_policy_train_epoch}_{iteration}.pt",
                            ),
                        )
                elif model_type == "guided_model":
                    if (
                        hasattr(config.parameter, "checkpoint_path")
                        and config.parameter.checkpoint_path is not None
                    ):
                        if not os.path.exists(config.parameter.checkpoint_path):
                            os.makedirs(config.parameter.checkpoint_path)
                        torch.save(
                            dict(
                                guided_model=model[
                                    "QGPOPolicy"
                                ].diffusion_model.energy_guidance.model.state_dict(),
                                guided_policy_train_epoch=self.guided_policy_train_epoch,
                                guided_policy_train_iteration=iteration,
                            ),
                            f=os.path.join(
                                config.parameter.checkpoint_path,
                                f"guidedmodel_{self.guided_policy_train_epoch}_{iteration}.pt",
                            ),
                        )
                elif model_type == "critic_model":
                    if (
                        hasattr(config.parameter, "checkpoint_path")
                        and config.parameter.checkpoint_path is not None
                    ):
                        if not os.path.exists(config.parameter.checkpoint_path):
                            os.makedirs(config.parameter.checkpoint_path)
                        if config.parameter.critic.method == "iql":
                            torch.save(
                                dict(
                                    critic_model=model[
                                        "QGPOPolicy"
                                    ].critic.state_dict(),
                                    critic_train_epoch=self.critic_train_epoch,
                                    critic_train_iter=iteration,
                                    value_function=self.vf.state_dict(),
                                ),
                                f=os.path.join(
                                    config.parameter.checkpoint_path,
                                    f"critic_{self.critic_train_epoch}_{iteration}.pt",
                                ),
                            )
                        elif config.parameter.critic.method == "in_support_ql":
                            torch.save(
                                dict(
                                    critic_model=model[
                                        "QGPOPolicy"
                                    ].critic.state_dict(),
                                    critic_train_epoch=self.critic_train_epoch,
                                    critic_train_iter=iteration,
                                ),
                                f=os.path.join(
                                    config.parameter.checkpoint_path,
                                    f"critic_{self.critic_train_epoch}_{iteration}.pt",
                                ),
                            )

            # ---------------------------------------
            # Customized training code ↓
            # ---------------------------------------

            def get_train_data(dataloader):
                while True:
                    yield from dataloader

            def generate_fake_action(model, states, sample_per_state):
                # model.eval()
                fake_actions_sampled = []
                for states in track(
                    np.array_split(states, states.shape[0] // 4096 + 1),
                    description="Generate fake actions",
                ):
                    # TODO: mkae it batchsize
                    fake_actions_per_state = []
                    for _ in range(sample_per_state):
                        fake_actions_per_state.append(
                            model.sample(
                                state=states,
                                guidance_scale=0.0,
                                t_span=(
                                    torch.linspace(
                                        0.0, 1.0, config.parameter.t_span
                                    ).to(states.device)
                                    if config.parameter.t_span is not None
                                    else None
                                ),
                            )
                        )
                    fake_actions_sampled.append(
                        torch.stack(fake_actions_per_state, dim=1)
                    )
                fake_actions = torch.cat(fake_actions_sampled, dim=0)
                return fake_actions

            def evaluate(model, train_epoch, guidance_scales, repeat=1):
                evaluation_results = dict()
                for guidance_scale in guidance_scales:

                    def policy(obs: np.ndarray) -> np.ndarray:
                        obs = torch.tensor(
                            obs,
                            dtype=torch.float32,
                            device=config.model.QGPOPolicy.device,
                        ).unsqueeze(0)
                        action = (
                            model.sample(
                                state=obs,
                                guidance_scale=guidance_scale,
                                t_span=(
                                    torch.linspace(
                                        0.0, 1.0, config.parameter.t_span
                                    ).to(config.model.QGPOPolicy.device)
                                    if config.parameter.t_span is not None
                                    else None
                                ),
                            )
                            .squeeze(0)
                            .cpu()
                            .detach()
                            .numpy()
                        )
                        return action

                    eval_results = self.simulator.evaluate(
                        policy=policy, num_episodes=repeat
                    )
                    return_results = [
                        eval_results[i]["total_return"] for i in range(repeat)
                    ]
                    log.info(f"Return: {return_results}")
                    return_mean = np.mean(return_results)
                    return_std = np.std(return_results)
                    return_max = np.max(return_results)
                    return_min = np.min(return_results)
                    evaluation_results[
                        f"evaluation/guidance_scale:[{guidance_scale}]/return_mean"
                    ] = return_mean
                    evaluation_results[
                        f"evaluation/guidance_scale:[{guidance_scale}]/return_std"
                    ] = return_std
                    evaluation_results[
                        f"evaluation/guidance_scale:[{guidance_scale}]/return_max"
                    ] = return_max
                    evaluation_results[
                        f"evaluation/guidance_scale:[{guidance_scale}]/return_min"
                    ] = return_min
                    if repeat > 1:
                        log.info(
                            f"Train epoch: {train_epoch}, guidance_scale: {guidance_scale}, return_mean: {return_mean}, return_std: {return_std}, return_max: {return_max}, return_min: {return_min}"
                        )
                    else:
                        log.info(
                            f"Train epoch: {train_epoch}, guidance_scale: {guidance_scale}, return: {return_mean}"
                        )

                return evaluation_results

            data_generator = get_train_data(
                DataLoader(
                    self.dataset,
                    batch_size=config.parameter.behaviour_policy.batch_size,
                    shuffle=True,
                    collate_fn=None,
                )
            )

            # ---------------------------------------
            # behavior training code ↓
            # ---------------------------------------

            behaviour_policy_optimizer = torch.optim.Adam(
                self.model["QGPOPolicy"].diffusion_model.model.parameters(),
                lr=config.parameter.behaviour_policy.learning_rate,
            )

            if (
                hasattr(config.parameter.behaviour_policy, "lr_decy")
                and config.parameter.behaviour_policy.lr_decy is True
            ):

                behaviour_lr_scheduler = CosineAnnealingLR(
                    behaviour_policy_optimizer,
                    T_max=config.parameter.behaviour_policy.epochs,
                    eta_min=0.0,
                )

            behaviour_policy_train_iter = 0
            self.behaviour_policy_train_epoch = 0
            for epoch in track(
                range(config.parameter.behaviour_policy.epochs),
                description="Behaviour policy training",
            ):

                if self.behaviour_policy_train_epoch >= epoch:
                    if (
                        hasattr(config.parameter.behaviour_policy, "lr_decy")
                        and config.parameter.behaviour_policy.lr_decy is True
                    ):
                        behaviour_lr_scheduler.step()
                    continue

                sampler = torch.utils.data.RandomSampler(
                    self.dataset, replacement=False
                )
                data_loader = torch.utils.data.DataLoader(
                    self.dataset,
                    batch_size=config.parameter.behaviour_policy.batch_size,
                    shuffle=False,
                    sampler=sampler,
                    pin_memory=False,
                    drop_last=True,
                )

                if (
                    hasattr(config.parameter.evaluation, "eval")
                    and config.parameter.evaluation.eval
                ):
                    if (
                        epoch
                        % config.parameter.evaluation.evaluation_behavior_policy_interval
                        == 0
                        or (epoch + 1) == config.parameter.behaviour_policy.epochs
                    ):
                        evaluation_results = evaluate(
                            self.model["QGPOPolicy"],
                            train_epoch=epoch,
                            guidance_scales=[0.0],
                            repeat=(
                                1
                                if not hasattr(config.parameter.evaluation, "repeat")
                                else config.parameter.evaluation.repeat
                            ),
                        )
                        wandb.log(data=evaluation_results, commit=False)

                counter = 1
                behaviour_policy_loss_sum = 0
                for data in data_loader:

                    behaviour_policy_loss = self.model[
                        "QGPOPolicy"
                    ].behaviour_policy_loss(
                        data["a"],
                        data["s"],
                        # maximum_likelihood=(
                        #     config.parameter.behaviour_policy.maximum_likelihood
                        #     if hasattr(
                        #         config.parameter.behaviour_policy, "maximum_likelihood"
                        #     )
                        #     else False
                        # ),
                    )
                    behaviour_policy_optimizer.zero_grad()
                    behaviour_policy_loss.backward()
                    if hasattr(config.parameter.behaviour_policy, "grad_norm_clip"):
                        behaviour_model_grad_norms = nn.utils.clip_grad_norm_(
                            self.model["QGPOPolicy"].base_model.parameters(),
                            max_norm=config.parameter.behaviour_policy.grad_norm_clip,
                            norm_type=2,
                        )
                    behaviour_policy_optimizer.step()

                    counter += 1
                    behaviour_policy_loss_sum += behaviour_policy_loss.item()

                    behaviour_policy_train_iter += 1
                    self.behaviour_policy_train_epoch = epoch

                self.behaviour_policy_train_epoch += 1

                wandb.log(
                    data=dict(
                        behaviour_policy_train_iter=behaviour_policy_train_iter,
                        behaviour_policy_train_epoch=epoch,
                        behaviour_policy_loss=behaviour_policy_loss_sum / counter,
                        behaviour_model_grad_norms=(
                            behaviour_model_grad_norms.item()
                            if hasattr(
                                config.parameter.behaviour_policy, "grad_norm_clip"
                            )
                            else 0.0
                        ),
                    ),
                    commit=True,
                )

                if (
                    hasattr(config.parameter.behaviour_policy, "lr_decy")
                    and config.parameter.behaviour_policy.lr_decy is True
                ):
                    behaviour_lr_scheduler.step()
                if (
                    hasattr(config.parameter, "checkpoint_freq")
                    and (epoch + 1) % config.parameter.checkpoint_freq == 0
                ):
                    save_checkpoint(
                        self.model,
                        iteration=behaviour_policy_train_iter,
                        model_type="base_model",
                    )

            # ---------------------------------------
            # behavior training code ↑
            # ---------------------------------------

            # ---------------------------------------
            # make fake action ↓
            # ---------------------------------------

            self.dataset.fake_actions = generate_fake_action(
                self.model["QGPOPolicy"],
                self.dataset.states[:],
                config.parameter.sample_per_state,
            )
            self.dataset.fake_next_actions = generate_fake_action(
                self.model["QGPOPolicy"],
                self.dataset.next_states[:],
                config.parameter.sample_per_state,
            )

            # ---------------------------------------
            # make fake action ↑
            # ---------------------------------------

            # ---------------------------------------
            # critic and energy guidance training code ↓
            # ---------------------------------------

            q_optimizer = torch.optim.Adam(
                self.model["QGPOPolicy"].critic.q.parameters(),
                lr=config.parameter.critic.learning_rate,
            )

            energy_guidance_optimizer = torch.optim.Adam(
                self.model["QGPOPolicy"].diffusion_model.energy_guidance.parameters(),
                lr=config.parameter.energy_guidance.learning_rate,
            )

            self.critic_train_epoch = 0
            self.energy_guidance_train_epoch = 0
            critic_train_iter = 0
            energy_guidance_train_iter = 0
            for epoch in track(
                range(
                    max(
                        config.parameter.critic.epochs,
                        config.parameter.energy_guidance.epochs,
                    )
                ),
                description="Critic and energy guidance training",
            ):
                if self.critic_train_epoch >= epoch:
                    continue

                sampler = torch.utils.data.RandomSampler(
                    self.dataset, replacement=False
                )
                data_loader = torch.utils.data.DataLoader(
                    self.dataset,
                    batch_size=config.parameter.critic.batch_size,
                    shuffle=False,
                    sampler=sampler,
                    pin_memory=False,
                    drop_last=True,
                )

                counter = 1
                q_loss_sum = 0.0
                q_sum = 0.0
                q_target_sum = 0.0
                q_grad_norms_sum = 0.0
                energy_guidance_loss_sum = 0.0

                for data in data_loader:
                    if (
                        critic_train_iter
                        < config.parameter.critic.stop_training_iterations
                    ):
                        q_loss = self.model["QGPOPolicy"].q_loss(
                            data["a"],
                            data["s"],
                            data["r"],
                            data["s_"],
                            data["d"],
                            data["fake_a_"],
                            discount_factor=config.parameter.critic.discount_factor,
                        )

                        q_optimizer.zero_grad()
                        q_loss.backward()
                        q_optimizer.step()
                        critic_train_iter += 1
                        q_loss_sum += q_loss.item()

                        # Update target
                        for param, target_param in zip(
                            self.model["QGPOPolicy"].critic.parameters(),
                            self.model["QGPOPolicy"].critic.q_target.parameters(),
                        ):
                            target_param.data.copy_(
                                config.parameter.critic.update_momentum * param.data
                                + (1 - config.parameter.critic.update_momentum)
                                * target_param.data
                            )

                    if self.energy_guidance_train_epoch >= epoch:
                        continue

                    energy_guidance_loss = self.model[
                        "QGPOPolicy"
                    ].energy_guidance_loss(data["s"], data["fake_a"])
                    energy_guidance_optimizer.zero_grad()
                    energy_guidance_loss.backward()
                    energy_guidance_optimizer.step()
                    energy_guidance_train_iter += 1
                    energy_guidance_loss_sum += energy_guidance_loss.item()
                    counter += 1

                self.critic_train_epoch += 1
                self.energy_guidance_train_epoch += 1

                if epoch % config.parameter.evaluation.evaluation_interval == 0 or (
                    epoch + 1
                ) == max(
                    config.parameter.critic.epochs,
                    config.parameter.energy_guidance.epochs,
                ):
                    evaluation_results = evaluate(
                        self.model["QGPOPolicy"],
                        train_epoch=epoch,
                        guidance_scales=config.parameter.evaluation.guidance_scale,
                    )
                    wandb_run.log(data=evaluation_results, commit=False)

                wandb_run.log(
                    data=dict(
                        critic_train_iter=critic_train_iter,
                        critic_train_epoch=epoch,
                        energy_guidance_train_epoch=epoch,
                        q_loss=q_loss_sum / counter,
                        energy_guidance_train_iter=energy_guidance_train_iter,
                        energy_guidance_loss=energy_guidance_loss_sum / counter,
                    ),
                    commit=True,
                )

                if (
                    hasattr(config.parameter, "checkpoint_freq")
                    and (epoch + 1) % config.parameter.checkpoint_freq == 0
                ):
                    save_checkpoint(
                        self.model,
                        iteration=energy_guidance_train_iter,
                        model_type="guided_model",
                    )
                    save_checkpoint(
                        self.model,
                        iteration=critic_train_iter,
                        model_type="critic_model",
                    )

            # ---------------------------------------
            # critic and energy guidance training code ↑
            # ---------------------------------------

            # ---------------------------------------
            # Customized training code ↑
            # ---------------------------------------

            wandb.finish()

    def deploy(self, config: EasyDict = None) -> QGPOAgent:

        if config is not None:
            config = merge_two_dicts_into_newone(self.config.deploy, config)
        else:
            config = self.config.deploy

        assert "QGPOPolicy" in self.model, "The model must be trained first."
        return QGPOAgent(
            config=config,
            model=copy.deepcopy(self.model),
        )
