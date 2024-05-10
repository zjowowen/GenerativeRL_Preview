import copy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from easydict import EasyDict
from rich.progress import Progress, track
from tensordict import TensorDict
from torch.utils.data import DataLoader
import os
import wandb
from grl.agents.gpo import GPOAgent
from grl.datasets import create_dataset
from grl.datasets.gpo import GPODataset
from grl.generative_models.diffusion_model import DiffusionModel
from grl.generative_models.conditional_flow_model.optimal_transport_conditional_flow_model import (
    OptimalTransportConditionalFlowModel,
)
from grl.generative_models.diffusion_model.guided_diffusion_model import (
    GuidedDiffusionModel,
)
from grl.generative_models.conditional_flow_model.guided_conditional_flow_model import (
    GuidedConditionalFlowModel,
)
from grl.rl_modules.simulators import create_simulator
from grl.rl_modules.value_network.q_network import DoubleQNetwork
from grl.utils.config import merge_two_dicts_into_newone
from grl.utils.log import log


class GPGCritic(nn.Module):
    """
    Overview:
        Critic network for GPG algorithm.
    Interfaces:
        ``__init__``, ``forward``
    """

    def __init__(self, config: EasyDict):
        """
        Overview:
            Initialization of GPG critic network.
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
            Return the output of GPG critic.
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
        self.type = config.model_type
        if self.type == "DiffusionModel":
            self.model = GuidedDiffusionModel(config.model)
        elif self.type in ["OptimalTransportConditionalFlowModel"]:
            self.model = GuidedConditionalFlowModel(config.model)
        else:
            raise NotImplementedError

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
            Return the output of GPG policy, which is the action conditioned on the state.
        Arguments:
            state (:obj:`Union[torch.Tensor, TensorDict]`): The input state.
            guidance_scale (:obj:`Union[torch.Tensor, float]`): The guidance scale.
            solver_config (:obj:`EasyDict`): The configuration for the ODE solver.
            t_span (:obj:`torch.Tensor`): The time span for the ODE solver or SDE solver.
        Returns:
            action (:obj:`Union[torch.Tensor, TensorDict]`): The output action.
        """

        if self.type == "DiffusionModel":
            return self.model.sample(
                base_model=base_model.model,
                guided_model=guided_model.model,
                t_span=t_span,
                condition=state,
                batch_size=batch_size,
                guidance_scale=guidance_scale,
                with_grad=False,
                solver_config=solver_config,
            )

        elif self.type in ["OptimalTransportConditionalFlowModel"]:

            x_0 = base_model.gaussian_generator(batch_size=state.shape[0])

            return self.model.sample(
                base_model=base_model.model,
                guided_model=guided_model.model,
                x_0=x_0,
                t_span=t_span,
                condition=state,
                batch_size=batch_size,
                guidance_scale=guidance_scale,
                with_grad=False,
                solver_config=solver_config,
            )


class GPGPolicy(nn.Module):

    def __init__(self, config: EasyDict):
        super().__init__()
        self.config = config
        self.device = config.device

        self.critic = GPGCritic(config.critic)
        self.model_type = config.model_type
        if self.model_type == "DiffusionModel":
            self.model = DiffusionModel(config.model)
            self.model_fine_tune = DiffusionModel(config.model)
        elif self.model_type == "OptimalTransportConditionalFlowModel":
            self.model = OptimalTransportConditionalFlowModel(config.model)
            self.model_fine_tune = OptimalTransportConditionalFlowModel(config.model)
        else:
            raise NotImplementedError

        self.softmax = nn.Softmax(dim=1)

    def forward(
        self, state: Union[torch.Tensor, TensorDict]
    ) -> Union[torch.Tensor, TensorDict]:
        """
        Overview:
            Return the output of GPG policy, which is the action conditioned on the state.
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
        solver_config: EasyDict = None,
        t_span: torch.Tensor = None,
        with_grad: bool = False,
    ) -> Union[torch.Tensor, TensorDict]:
        """
        Overview:
            Return the output of GPG policy, which is the action conditioned on the state.
        Arguments:
            state (:obj:`Union[torch.Tensor, TensorDict]`): The input state.
            batch_size (:obj:`Union[torch.Size, int, Tuple[int], List[int]]`): The batch size.
            solver_config (:obj:`EasyDict`): The configuration for the ODE solver.
            t_span (:obj:`torch.Tensor`): The time span for the ODE solver or SDE solver.
        Returns:
            action (:obj:`Union[torch.Tensor, TensorDict]`): The output action.
        """

        return self.model_fine_tune.sample(
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
        solver_config: EasyDict = None,
        t_span: torch.Tensor = None,
        with_grad: bool = False,
    ) -> Union[torch.Tensor, TensorDict]:
        """
        Overview:
            Return the output of behaviour policy, which is the action conditioned on the state.
        Arguments:
            state (:obj:`Union[torch.Tensor, TensorDict]`): The input state.
            batch_size (:obj:`Union[torch.Size, int, Tuple[int], List[int]]`): The batch size.
            solver_config (:obj:`EasyDict`): The configuration for the ODE solver.
            t_span (:obj:`torch.Tensor`): The time span for the ODE solver or SDE solver.
            with_grad (:obj:`bool`): Whether to calculate the gradient.
        Returns:
            action (:obj:`Union[torch.Tensor, TensorDict]`): The output action.
        """
        return self.model.sample(
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

        if self.model_type == "DiffusionModel":
            return self.model.flow_matching_loss(action, state)
            # if maximum_likelihood:
            #     return self.model.score_matching_loss(action, state)
            # else:
            #     return self.model.score_matching_loss(
            #         action, state, weighting_scheme="vanilla"
            #     )
        elif self.model_type == "OptimalTransportConditionalFlowModel":
            x0 = self.model.gaussian_generator(batch_size=state.shape[0])
            return self.model.flow_matching_loss(x0=x0, x1=action, condition=state)

    def policy_loss_withgrade(
        self,
        action: Union[torch.Tensor, TensorDict],
        state: Union[torch.Tensor, TensorDict],
        fake_action: Union[torch.Tensor, TensorDict],
        maximum_likelihood: bool = False,
    ):
        """
        Overview:
            Calculate the behaviour policy loss.
        Arguments:
            action (:obj:`torch.Tensor`): The input action.
            state (:obj:`torch.Tensor`): The input state.
        """

        if self.model_type == "DiffusionModel":
            if maximum_likelihood:
                model_loss = self.model_fine_tune.score_matching_loss(
                    action, state, average=False
                )
            else:
                model_loss = self.model_fine_tune.score_matching_loss(
                    action, state, weighting_scheme="vanilla", average=False
                )
        elif self.model_type in ["OptimalTransportConditionalFlowModel"]:
            x0 = self.model_fine_tune.gaussian_generator(batch_size=state.shape[0])
            model_loss = self.model_fine_tune.flow_matching_loss(
                x0=x0, x1=action, condition=state, average=False
            )
        else:
            raise NotImplementedError

        new_action = self.model_fine_tune.sample(condition=state, with_grad=True)
        q_value = self.critic(new_action, state).squeeze(dim=-1)
        loss = (-q_value + model_loss).mean()
        return loss

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


class GPGAlgorithm:

    def __init__(
        self,
        config: EasyDict = None,
        simulator=None,
        dataset: GPODataset = None,
        model: Union[torch.nn.Module, torch.nn.ModuleDict] = None,
    ):
        """
        Overview:
            Initialize the GPG algorithm.
        Arguments:
            config (:obj:`EasyDict`): The configuration , which must contain the following keys:
                train (:obj:`EasyDict`): The training configuration.
                deploy (:obj:`EasyDict`): The deployment configuration.
            simulator (:obj:`object`): The environment simulator.
            dataset (:obj:`GPODataset`): The dataset.
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

            if hasattr(config.model, "GPGPolicy"):
                self.model["GPGPolicy"] = GPGPolicy(config.model.GPGPolicy)
                self.model["GPGPolicy"].to(config.model.GPGPolicy.device)
                if torch.__version__ >= "2.0.0":
                    self.model["GPGPolicy"] = torch.compile(self.model["GPGPolicy"])
                self.model["GuidedPolicy"] = GuidedPolicy(
                    config=config.model.GuidedPolicy
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

                    fake_actions_ = model.behaviour_policy_sample(
                        state=states,
                        batch_size=sample_per_state,
                        t_span=(
                            torch.linspace(
                                0.0, 1.0, config.parameter.fake_data_t_span
                            ).to(states.device)
                            if config.parameter.fake_data_t_span is not None
                            else None
                        ),
                    )
                    fake_actions_sampled.append(torch.einsum("nbd->bnd", fake_actions_))

                fake_actions = torch.cat(fake_actions_sampled, dim=0)
                return fake_actions

            def evaluate(model, train_iter):
                evaluation_results = dict()
                for guidance_scale in config.parameter.evaluation.guidance_scale:

                    def policy(obs: np.ndarray) -> np.ndarray:
                        obs = torch.tensor(
                            obs,
                            dtype=torch.float32,
                            device=config.model.GPGPolicy.device,
                        ).unsqueeze(0)
                        action = (
                            model["GuidedPolicy"]
                            .sample(
                                base_model=self.model["GPGPolicy"].model,
                                guided_model=self.model["GPGPolicy"].model_fine_tune,
                                state=obs,
                                guidance_scale=guidance_scale,
                                t_span=(
                                    torch.linspace(
                                        0.0, 1.0, config.parameter.fake_data_t_span
                                    ).to(config.model.GPGPolicy.device)
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
                self.model["GPGPolicy"].model.model.parameters(),
                lr=config.parameter.behaviour_policy.learning_rate,
            )

            for train_iter in track(
                range(config.parameter.behaviour_policy.iterations),
                description="Behaviour policy training",
            ):
                data = next(data_generator)
                behaviour_model_training_loss = self.model[
                    "GPGPolicy"
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
                wandb_run.log(
                    data=dict(
                        train_iter=train_iter,
                        behaviour_model_training_loss=behaviour_model_training_loss.item(),
                    ),
                    commit=True,
                )

                if (
                    train_iter == 0
                    or (train_iter + 1)
                    % config.parameter.evaluation.evaluation_interval
                    == 0
                ):
                    evaluation_results = evaluate(self.model, train_iter=train_iter)
                    wandb_run.log(data=evaluation_results, commit=False)

                if train_iter == config.parameter.behaviour_policy.iterations - 1:
                    file_path = os.path.join(
                        f"./model/checkpoint_diffusion_{train_iter+1}.pt"
                    )
                    torch.save(
                        dict(
                            self.model["GPGPolicy"].model.model.state_dict(),
                            diffusion_iteration=train_iter + 1,
                        ),
                        f=file_path,
                    )

            self.dataset.fake_actions = generate_fake_action(
                self.model["GPGPolicy"],
                self.dataset.states[:],
                config.parameter.sample_per_state,
            )

            self.dataset.fake_next_actions = generate_fake_action(
                self.model["GPGPolicy"],
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
                self.model["GPGPolicy"].critic.q.parameters(),
                lr=config.parameter.critic.learning_rate,
            )

            for train_iter in track(
                range(config.parameter.critic.iterations),
                description="Critic training",
            ):

                data = next(data_generator)

                q_loss, q, q_target = self.model["GPGPolicy"].q_loss(
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
                    self.model["GPGPolicy"].critic.parameters(),
                    self.model["GPGPolicy"].critic.q_target.parameters(),
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
                if train_iter == config.parameter.critic.iterations - 1:
                    file_path = os.path.join(
                        f"./model/checkpoint_critic_{train_iter+1}.pt"
                    )
                    torch.save(
                        dict(
                            self.model["GPGPolicy"].critic.q.state_dict(),
                            diffusion_iteration=train_iter + 1,
                        ),
                        f=file_path,
                    )

            model_fine_tune_optimizer = torch.optim.Adam(
                self.model["GPGPolicy"].model_finetune.parameters(),
                lr=config.parameter.model_finetune.learning_rate,
            )

            data_generator = get_train_data(
                DataLoader(
                    self.dataset,
                    batch_size=config.parameter.model_fine_tune.batch_size,
                    shuffle=True,
                    collate_fn=None,
                )
            )

            for train_iter in track(
                range(config.parameter.model_finetune.iterations),
                description="Energy conditioned diffusion model training",
            ):

                data = next(data_generator)
                model_fine_tune_loss = self.model["GPGPolicy"].policy_loss(
                    data["a"], data["s"], data["fake_a"]
                )
                model_fine_tune_optimizer.zero_grad()
                model_fine_tune_loss.backward()
                model_fine_tune_optimizer.step()

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
                        model_fine_tune_loss=model_fine_tune_loss.item(),
                    ),
                    commit=True,
                )

            # ---------------------------------------
            # Customized training code ↑
            # ---------------------------------------

            wandb.finish()

    def deploy(self, config: EasyDict = None) -> GPOAgent:

        if config is not None:
            config = merge_two_dicts_into_newone(self.config.deploy, config)
        else:
            config = self.config.deploy

        assert "GPOPolicy" in self.model, "The model must be trained first."
        assert "GuidedPolicy" in self.model, "The model must be trained first."
        return GPOAgent(
            config=config,
            model=copy.deepcopy(self.model),
        )
