import copy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from easydict import EasyDict
from rich.progress import Progress, track
from tensordict import TensorDict
from torch.utils.data import DataLoader

import wandb
from grl.agents.qgpo import QGPOISAgent
from grl.datasets import create_dataset
from grl.datasets.qgpo import QGPODataset
from grl.generative_models.diffusion_model import DiffusionModel
from grl.generative_models.diffusion_model.guided_diffusion_model import (
    GuidedDiffusionModel,
)
from grl.rl_modules.simulators import create_simulator
from grl.rl_modules.value_network.q_network import DoubleQNetwork
from grl.utils.config import merge_two_dicts_into_newone
from grl.utils.log import log


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
        return q_loss, torch.mean(q0), torch.mean(targets)


class GuidedPolicy(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.model = GuidedDiffusionModel(config)

    def sample(
        self,
        base_model,
        guided_model,
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

        return self.model.sample(
            base_model=base_model,
            guided_model=guided_model,
            t_span=t_span,
            condition=state,
            batch_size=batch_size,
            guidance_scale=guidance_scale,
            with_grad=False,
            solver_config=solver_config,
        )


class QGPOPolicy(nn.Module):

    def __init__(self, config: EasyDict):
        super().__init__()
        self.config = config
        self.device = config.device

        self.critic = QGPOCritic(config.critic)
        self.diffusion_model = DiffusionModel(config.diffusion_model)
        self.diffusion_model_important_sampling = DiffusionModel(config.diffusion_model)

        self.softmax = nn.Softmax(dim=1)

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
        with_grad: bool = False,
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

        return self.diffusion_model_important_sampling.sample(
            t_span=t_span,
            condition=state,
            batch_size=batch_size,
            with_grad=with_grad,
            solver_config=solver_config,
        )

    def behaviour_policy_sample(
        self,
        state: Union[torch.Tensor, TensorDict],
        batch_size: Union[torch.Size, int, Tuple[int], List[int]] = None,
        with_grad: bool = False,
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
        return self.diffusion_model.sample(
            t_span=t_span,
            condition=state,
            batch_size=batch_size,
            with_grad=with_grad,
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
        maximum_likelihood: bool = False,
    ):
        """
        Overview:
            Calculate the behaviour policy loss.
        Arguments:
            action (:obj:`torch.Tensor`): The input action.
            state (:obj:`torch.Tensor`): The input state.
        """

        if maximum_likelihood:
            return self.diffusion_model.score_matching_loss(action, state)
        else:
            return self.diffusion_model.score_matching_loss(
                action, state, weighting_scheme="vanilla"
            )

    def policy_loss(
        self,
        action: Union[torch.Tensor, TensorDict],
        state: Union[torch.Tensor, TensorDict],
        fake_action: Union[torch.Tensor, TensorDict],
    ):
        """
        Overview:
            Calculate the behaviour policy loss.
        Arguments:
            action (:obj:`torch.Tensor`): The input action.
            state (:obj:`torch.Tensor`): The input state.
        """

        score_loss = self.diffusion_model_important_sampling.score_matching_loss(
            action, state, weighting_scheme="vanilla", average=False
        )

        with torch.no_grad():
            q_value = self.critic(action, state).squeeze(dim=-1)
            fake_q_value = (
                self.critic(
                    fake_action, torch.stack([state] * fake_action.shape[1], axis=1)
                )
                .squeeze(dim=-1)
                .detach()
                .squeeze(dim=-1)
            )

            v_value = torch.sum(
                self.softmax(self.critic.q_alpha * fake_q_value) * fake_q_value,
                dim=-1,
                keepdim=True,
            ).squeeze(dim=-1)

            # TODO: which is better? I think maybe the following is incorrect.
            # concatenate q_value to fake_q_value
            # fake_q_value_plus_q_value = torch.cat(
            #     [fake_q_value, q_value.unsqueeze(1)], dim=1
            # )

            # v_value = torch.sum(
            #     self.softmax(self.critic.q_alpha * fake_q_value_plus_q_value)
            #     * fake_q_value_plus_q_value,
            #     dim=-1,
            #     keepdim=True,
            # ).squeeze(dim=-1)

        return torch.mean(score_loss * torch.exp(q_value - v_value))

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


class QGPOISAlgorithm:

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

            if hasattr(config.model, "QGPOPolicy"):
                self.model["QGPOPolicy"] = QGPOPolicy(config.model.QGPOPolicy)
                self.model["QGPOPolicy"].to(config.model.QGPOPolicy.device)
                if torch.__version__ >= "2.0.0":
                    self.model["QGPOPolicy"] = torch.compile(self.model["QGPOPolicy"])
                self.model["GuidedPolicy"] = GuidedPolicy(
                    config=config.model.QGPOPolicy.diffusion_model
                )

            # ---------------------------------------
            # Customized model initialization code ↑
            # ---------------------------------------

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
                            model.behaviour_policy_sample(
                                state=states,
                                t_span=(
                                    torch.linspace(
                                        0.0, 1.0, config.parameter.fake_data_t_span
                                    ).to(states.device)
                                    if config.parameter.fake_data_t_span is not None
                                    else None
                                ),
                            )
                        )
                    fake_actions_sampled.append(
                        torch.stack(fake_actions_per_state, dim=1)
                    )
                fake_actions = torch.cat(fake_actions_sampled, dim=0)
                return fake_actions

            def evaluate(model, train_iter):
                evaluation_results = dict()
                for guidance_scale in config.parameter.evaluation.guidance_scale:

                    def policy(obs: np.ndarray) -> np.ndarray:
                        obs = torch.tensor(
                            obs,
                            dtype=torch.float32,
                            device=config.model.QGPOPolicy.device,
                        ).unsqueeze(0)
                        action = (
                            model["GuidedPolicy"]
                            .sample(
                                base_model=self.model[
                                    "QGPOPolicy"
                                ].diffusion_model.model,
                                guided_model=self.model[
                                    "QGPOPolicy"
                                ].diffusion_model_important_sampling.model,
                                state=obs,
                                guidance_scale=guidance_scale,
                                t_span=(
                                    torch.linspace(
                                        0.0, 1.0, config.parameter.fake_data_t_span
                                    ).to(config.model.QGPOPolicy.device)
                                    if config.parameter.fake_data_t_span is not None
                                    else None
                                ),
                            )
                            .squeeze(0)
                            .cpu()
                            .detach()
                            .numpy()
                        )
                        return action

                    evaluation_results[
                        f"evaluation/guidance_scale:[{guidance_scale}]/total_return"
                    ] = self.simulator.evaluate(policy=policy,)[0]["total_return"]
                    log.info(
                        f"Train iter: {train_iter}, guidance_scale: {guidance_scale}, total_return: {evaluation_results[f'evaluation/guidance_scale:[{guidance_scale}]/total_return']}"
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

            behaviour_model_optimizer = torch.optim.Adam(
                self.model["QGPOPolicy"].diffusion_model.model.parameters(),
                lr=config.parameter.behaviour_policy.learning_rate,
            )

            for train_iter in track(
                range(config.parameter.behaviour_policy.iterations),
                description="Behaviour policy training",
            ):
                data = next(data_generator)
                behaviour_model_training_loss = self.model[
                    "QGPOPolicy"
                ].behaviour_policy_loss(
                    data["a"],
                    data["s"],
                    maximum_likelihood=(
                        config.parameter.behaviour_policy.maximum_likelihood
                        if hasattr(
                            config.parameter.behaviour_policy, "maximum_likelihood"
                        )
                        else False
                    ),
                )
                behaviour_model_optimizer.zero_grad()
                behaviour_model_training_loss.backward()
                behaviour_model_optimizer.step()

                if (
                    train_iter < 0
                    or (train_iter + 1)
                    % config.parameter.evaluation.evaluation_interval
                    == 0
                ):
                    evaluation_results = evaluate(self.model, train_iter=train_iter)
                    wandb_run.log(data=evaluation_results, commit=False)

                wandb_run.log(
                    data=dict(
                        train_iter=train_iter,
                        behaviour_model_training_loss=behaviour_model_training_loss.item(),
                    ),
                    commit=True,
                )

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

            # TODO add notation
            data_generator = get_train_data(
                DataLoader(
                    self.dataset,
                    batch_size=config.parameter.critic.batch_size,
                    shuffle=True,
                    collate_fn=None,
                )
            )

            q_optimizer = torch.optim.Adam(
                self.model["QGPOPolicy"].critic.q.parameters(),
                lr=config.parameter.critic.learning_rate,
            )

            for train_iter in track(
                range(config.parameter.critic.iterations),
                description="Critic training",
            ):

                data = next(data_generator)

                q_loss, q, q_target = self.model["QGPOPolicy"].q_loss(
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

                wandb_run.log(
                    data=dict(
                        train_iter=train_iter,
                        q_loss=q_loss.item(),
                        q=q.item(),
                        q_target=q_target.item(),
                    ),
                    commit=True,
                )

            diffusion_model_important_sampling_optimizer = torch.optim.Adam(
                self.model[
                    "QGPOPolicy"
                ].diffusion_model_important_sampling.parameters(),
                lr=config.parameter.diffusion_model_important_sampling.learning_rate,
            )

            data_generator = get_train_data(
                DataLoader(
                    self.dataset,
                    batch_size=config.parameter.diffusion_model_important_sampling.batch_size,
                    shuffle=True,
                    collate_fn=None,
                )
            )

            for train_iter in track(
                range(config.parameter.diffusion_model_important_sampling.iterations),
                description="Energy conditioned diffusion model training",
            ):

                data = next(data_generator)

                diffusion_model_important_sampling_loss = self.model[
                    "QGPOPolicy"
                ].policy_loss(data["a"], data["s"], data["fake_a"])
                diffusion_model_important_sampling_optimizer.zero_grad()
                diffusion_model_important_sampling_loss.backward()
                diffusion_model_important_sampling_optimizer.step()

                if (
                    train_iter < 0
                    or (train_iter + 1)
                    % config.parameter.evaluation.evaluation_interval
                    == 0
                ):
                    evaluation_results = evaluate(self.model, train_iter=train_iter)
                    wandb_run.log(data=evaluation_results, commit=False)

                wandb_run.log(
                    data=dict(
                        train_iter=train_iter,
                        diffusion_model_important_sampling_loss=diffusion_model_important_sampling_loss.item(),
                    ),
                    commit=True,
                )

            # ---------------------------------------
            # Customized training code ↑
            # ---------------------------------------

            wandb.finish()

    def deploy(self, config: EasyDict = None) -> QGPOISAgent:

        if config is not None:
            config = merge_two_dicts_into_newone(self.config.deploy, config)
        else:
            config = self.config.deploy

        assert "QGPOPolicy" in self.model, "The model must be trained first."
        assert "GuidedPolicy" in self.model, "The model must be trained first."
        return QGPOISAgent(
            config=config,
            model=copy.deepcopy(self.model),
        )
