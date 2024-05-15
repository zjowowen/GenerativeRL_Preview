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

import wandb
from grl.agents.gpo import GPOAgent
from grl.datasets import create_dataset
from grl.datasets.gpo import GPODataset
from grl.generative_models.diffusion_model import DiffusionModel
from grl.generative_models.conditional_flow_model.optimal_transport_conditional_flow_model import (
    OptimalTransportConditionalFlowModel,
)
from grl.generative_models.conditional_flow_model.independent_conditional_flow_model import (
    IndependentConditionalFlowModel,
)
from grl.generative_models.bridge_flow_model.schrodinger_bridge_conditional_flow_model import (
    SchrodingerBridgeConditionalFlowModel,
)
from grl.generative_models.bridge_flow_model.guided_bridge_conditional_flow_model import (
    GuidedBridgeConditionalFlowModel,
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
from grl.utils import set_seed


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
        elif self.type in [
            "OptimalTransportConditionalFlowModel",
            "IndependentConditionalFlowModel",
        ]:
            self.model = GuidedConditionalFlowModel(config.model)
        elif self.type == "SchrodingerBridgeConditionalFlowModel":
            self.model = GuidedBridgeConditionalFlowModel(config.model)
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
                base_model=base_model,
                guided_model=guided_model,
                t_span=t_span,
                condition=state,
                batch_size=batch_size,
                guidance_scale=guidance_scale,
                with_grad=False,
                solver_config=solver_config,
            )

        elif self.type in [
            "OptimalTransportConditionalFlowModel",
            "IndependentConditionalFlowModel",
            "SchrodingerBridgeConditionalFlowModel",
        ]:
            x_0 = base_model.gaussian_generator(batch_size=state.shape[0])

            return self.model.sample(
                base_model=base_model,
                guided_model=guided_model,
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
            self.base_model = DiffusionModel(config.model)
            self.guided_model = DiffusionModel(config.model)
            self.model_loss_type = config.model_loss_type
            assert self.model_loss_type in ["score_matching", "flow_matching"]
        elif self.model_type == "OptimalTransportConditionalFlowModel":
            self.base_model = OptimalTransportConditionalFlowModel(config.model)
            self.guided_model = OptimalTransportConditionalFlowModel(config.model)
        elif self.model_type == "IndependentConditionalFlowModel":
            self.base_model = IndependentConditionalFlowModel(config.model)
            self.guided_model = IndependentConditionalFlowModel(config.model)
        elif self.model_type == "SchrodingerBridgeConditionalFlowModel":
            self.base_model = SchrodingerBridgeConditionalFlowModel(config.model)
            self.guided_model = SchrodingerBridgeConditionalFlowModel(config.model)
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

        return self.guided_model.sample(
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
        return self.base_model.sample(
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
            if self.model_loss_type == "score_matching":
                if maximum_likelihood:
                    return self.base_model.score_matching_loss(action, state)
                else:
                    return self.base_model.score_matching_loss(
                        action, state, weighting_scheme="vanilla"
                    )
            elif self.model_loss_type == "flow_matching":
                return self.base_model.flow_matching_loss(action, state)
        elif self.model_type in [
            "OptimalTransportConditionalFlowModel",
            "IndependentConditionalFlowModel",
            "SchrodingerBridgeConditionalFlowModel",
        ]:
            x0 = self.base_model.gaussian_generator(batch_size=state.shape[0])
            return self.base_model.flow_matching_loss(x0=x0, x1=action, condition=state)

    def policy_loss_withgrade(
        self,
        action: Union[torch.Tensor, TensorDict],
        state: Union[torch.Tensor, TensorDict],
        maximum_likelihood: bool = False,
        loss_type: str = "origin_loss",
    ):
        """
        Overview:
            Calculate the behaviour policy loss.
        Arguments:
            action (:obj:`torch.Tensor`): The input action.
            state (:obj:`torch.Tensor`): The input state.
        """

        if self.model_type == "DiffusionModel":
            if self.model_loss_type == "score_matching":
                if maximum_likelihood:
                    model_loss = self.guided_model.score_matching_loss(
                        action, state, average=True
                    )
                else:
                    model_loss = self.guided_model.score_matching_loss(
                        action, state, weighting_scheme="vanilla", average=True
                    )
            elif self.model_loss_type == "flow_matching":
                model_loss = self.guided_model.flow_matching_loss(
                    action, state, average=True
                )
        elif self.model_type in [
            "OptimalTransportConditionalFlowModel",
            "IndependentConditionalFlowModel",
            "SchrodingerBridgeConditionalFlowModel",
        ]:
            x0 = self.guided_model.gaussian_generator(batch_size=state.shape[0])
            model_loss = self.guided_model.flow_matching_loss(
                x0=x0, x1=action, condition=state, average=True
            )
        else:
            raise NotImplementedError
        t_span = torch.linspace(0.0, 1.0, 1000).to(state.device)
        new_action = self.guided_model.sample(
            t_span=t_span, condition=state, with_grad=True
        )

        q_value = self.critic(new_action, state)
        if loss_type == "origin_loss":
            return -q_value.mean() + model_loss
        elif loss_type == "detach_loss":
            return -(q_value / q_value.abs().detach()).mean() + model_loss
        elif loss_type == "minibatch_loss":
            q_loss = -q_value.mean() / q_value.abs().mean().detach()
            return q_loss + model_loss
        elif loss_type == "double_minibatch_loss":
            q1, q2 = self.critic.q.compute_double_q(new_action, state)
            if np.random.uniform() > 0.5:
                q_loss = -q1.mean() / q2.abs().mean().detach()
            else:
                q_loss = -q2.mean() / q1.abs().mean().detach()
            return q_loss + model_loss
        else:
            raise ValueError(("Unknown activation function {}".format(loss_type)))

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

    def train(self, config: EasyDict = None, seed=None):
        """
        Overview:
            Train the model using the given configuration. \
            A weight-and-bias run will be created automatically when this function is called.
        Arguments:
            config (:obj:`EasyDict`): The training configuration.
        """
        seed_value = set_seed(seed_value=seed)
        config = (
            merge_two_dicts_into_newone(
                self.config.train if hasattr(self.config, "train") else EasyDict(),
                config,
            )
            if config is not None
            else self.config.train
        )
        config["seed"] = seed_value

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
            if (
                hasattr(config.parameter, "checkpoint_path")
                and config.parameter.checkpoint_path is not None
            ):

                if not os.path.exists(config.parameter.checkpoint_path):
                    log.warning(
                        f"Checkpoint path {config.parameter.checkpoint_path} does not exist"
                    )
                    behaviour_policy_train_epoch = 0
                    critic_train_epoch = 0
                    guided_policy_train_epoch = 0
                else:
                    checkpoint_files = [
                        f
                        for f in os.listdir(config.parameter.checkpoint_path)
                        if f.endswith(".pt")
                    ]
                    checkpoint_files = sorted(
                        checkpoint_files,
                        key=lambda x: int(x.split("_")[-1].split(".")[0]),
                    )
                    checkpoint = torch.load(
                        os.path.join(
                            config.parameter.checkpoint_path, checkpoint_files[-1]
                        ),
                        map_location="cpu",
                    )
                    self.model.load_state_dict(checkpoint["model"])
                    behaviour_policy_train_epoch = checkpoint.get(
                        "behaviour_policy_train_epoch", 0
                    )
                    critic_train_epoch = checkpoint.get("critic_train_epoch", 0)
                    guided_policy_train_epoch = checkpoint.get(
                        "guided_policy_train_epoch", 0
                    )
            else:
                behaviour_policy_train_epoch = 0
                critic_train_epoch = 0
                guided_policy_train_epoch = 0

            def save_checkpoint(
                model,
                behaviour_policy_train_epoch,
                critic_train_epoch,
                guided_policy_train_epoch,
            ):
                if (
                    hasattr(config.parameter, "checkpoint_path")
                    and config.parameter.checkpoint_path is not None
                ):
                    if not os.path.exists(config.parameter.checkpoint_path):
                        os.makedirs(config.parameter.checkpoint_path)
                    torch.save(
                        dict(
                            model=model.state_dict(),
                            behaviour_policy_train_epoch=behaviour_policy_train_epoch,
                            critic_train_epoch=critic_train_epoch,
                            guided_policy_train_epoch=guided_policy_train_epoch,
                        ),
                        f=os.path.join(
                            config.parameter.checkpoint_path,
                            f"checkpoint_{behaviour_policy_train_epoch}_{critic_train_epoch}_{guided_policy_train_epoch}.pt",
                        ),
                    )

            # ---------------------------------------
            # Customized training code ↓
            # ---------------------------------------
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

            def evaluate(model, train_epoch, guidance_scales, repeat=1):
                evaluation_results = dict()
                for guidance_scale in guidance_scales:

                    def policy(obs: np.ndarray) -> np.ndarray:
                        obs = torch.tensor(
                            obs,
                            dtype=torch.float32,
                            device=config.model.GPGPolicy.device,
                        ).unsqueeze(0)
                        action = (
                            model["GuidedPolicy"]
                            .sample(
                                base_model=self.model["GPGPolicy"].base_model,
                                guided_model=self.model["GPGPolicy"].guided_model,
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

                    return_results = [
                        self.simulator.evaluate(
                            policy=policy,
                        )[
                            0
                        ]["total_return"]
                        for _ in range(repeat)
                    ]
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

            behaviour_model_optimizer = torch.optim.Adam(
                self.model["GPGPolicy"].base_model.parameters(),
                lr=config.parameter.behaviour_policy.learning_rate,
            )

            behaviour_policy_train_iter = 0
            for epoch in track(
                range(config.parameter.behaviour_policy.epochs),
                description="Behaviour policy training",
            ):
                if behaviour_policy_train_epoch > epoch:
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
                    epoch + 1
                ) % config.parameter.evaluation.evaluation_interval == 0 or (
                    epoch + 1
                ) == config.parameter.behaviour_policy.epochs:
                    evaluation_results = evaluate(
                        self.model,
                        train_epoch=epoch,
                        guidance_scales=[0.0],
                        repeat=(
                            1
                            if not hasattr(config.parameter.evaluation, "repeat")
                            else config.parameter.evaluation.repeat
                        ),
                    )
                    wandb_run.log(data=evaluation_results, commit=False)

                for data in data_loader:
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
                    behaviour_policy_train_iter += 1
                    behaviour_model_optimizer.zero_grad()
                    behaviour_model_training_loss.backward()
                    behaviour_model_optimizer.step()
                    wandb_run.log(
                        data=dict(
                            behaviour_policy_train_iter=behaviour_policy_train_iter,
                            behaviour_policy_train_epoch=epoch,
                            behaviour_model_training_loss=behaviour_model_training_loss.item(),
                        ),
                        commit=True,
                    )
                behaviour_policy_train_epoch = epoch

                if (
                    hasattr(config.parameter, "checkpoint_freq")
                    and (epoch + 1) % config.parameter.checkpoint_freq == 0
                ):
                    save_checkpoint(
                        self.model,
                        behaviour_policy_train_epoch + 1,
                        critic_train_epoch + 1,
                        guided_policy_train_epoch + 1,
                    )

                if (
                    hasattr(config.parameter.behaviour_policy, "iterations")
                    and behaviour_policy_train_iter
                    >= config.parameter.behaviour_policy.iterations
                ):
                    log.info("Behaviour policy training finished.")
                    break

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

            q_optimizer = torch.optim.Adam(
                self.model["GPGPolicy"].critic.q.parameters(),
                lr=config.parameter.critic.learning_rate,
            )

            critic_train_iter = 0
            for epoch in track(
                range(config.parameter.critic.epochs), description="Critic training"
            ):
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
                if critic_train_epoch > epoch:
                    continue
                for data in data_loader:
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
                            critic_train_iter=critic_train_iter,
                            critic_train_epoch=epoch,
                            q_loss=q_loss.item(),
                            q=q.item(),
                            q_target=q_target.item(),
                        ),
                        commit=True,
                    )
                    critic_train_iter += 1
                    critic_train_epoch = epoch

                if (
                    hasattr(config.parameter, "checkpoint_freq")
                    and (epoch + 1) % config.parameter.checkpoint_freq == 0
                ):
                    save_checkpoint(
                        self.model,
                        behaviour_policy_train_epoch + 1,
                        critic_train_epoch + 1,
                        guided_policy_train_epoch + 1,
                    )

                if (
                    hasattr(config.parameter.critic, "iterations")
                    and critic_train_iter >= config.parameter.critic.iterations
                ):
                    log.info("Critic training finished.")
                    break

            if config.parameter.guided_policy.copy_from_basemodel:
                self.model["GPGPolicy"].guided_model.load_state_dict(
                    self.model["GPGPolicy"].base_model.state_dict()
                )
            guided_policy_optimizer = torch.optim.Adam(
                self.model["GPGPolicy"].guided_model.parameters(),
                lr=config.parameter.guided_policy.learning_rate,
            )

            if config.parameter.guided_policy.lr_decy:
                from torch.optim.lr_scheduler import CosineAnnealingLR

                guided_lr_scheduler = CosineAnnealingLR(
                    guided_policy_optimizer,
                    T_max=config.parameter.guided_policy.epochs,
                    eta_min=0.0,
                )
            guided_policy_train_iter = 0
            for epoch in track(
                range(config.parameter.guided_policy.epochs),
                description="Guided policy training",
            ):
                if guided_policy_train_epoch > epoch:
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

                if (
                    epoch + 1
                ) % config.parameter.guided_policy.evaluation_interval == 0 or (
                    epoch + 1
                ) == config.parameter.guided_policy.epochs:
                    evaluation_results = evaluate(
                        self.model,
                        train_epoch=epoch,
                        guidance_scales=config.parameter.evaluation.guidance_scale,
                        repeat=(
                            1
                            if not hasattr(config.parameter.evaluation, "repeat")
                            else config.parameter.evaluation.repeat
                        ),
                    )
                    wandb_run.log(data=evaluation_results, commit=False)

                for data in data_loader:
                    guided_model_loss = self.model["GPGPolicy"].policy_loss_withgrade(
                        data["a"],
                        data["s"],
                        maximum_likelihood=True,
                        loss_type=config.parameter.guided_policy.loss_type,
                    )

                    guided_model_grad_norms = nn.utils.clip_grad_norm_(
                        self.model["GPGPolicy"].guided_model.parameters(),
                        max_norm=config.parameter.guided_policy.grad_norm_clip,
                        norm_type=2,
                    )
                    guided_policy_optimizer.zero_grad()
                    guided_model_loss.backward()
                    guided_policy_optimizer.step()
                    wandb_run.log(
                        data=dict(
                            guided_policy_train_iter=guided_policy_train_iter,
                            guided_policy_train_epoch=epoch,
                            guided_model_training_loss=guided_model_loss.item(),
                            guided_model_grad_norms=guided_model_grad_norms,
                        ),
                        commit=True,
                    )
                    guided_policy_train_iter += 1

                if config.parameter.guided_policy.lr_decy:
                    guided_lr_scheduler.step()

                # if (guided_policy_train_iter + 1) % 5 == 0:
                #     evaluation_results = evaluate(
                #         self.model,
                #         train_epoch=epoch,
                #         guidance_scales=config.parameter.evaluation.guidance_scale,
                #         repeat=(
                #             1
                #             if not hasattr(config.parameter.evaluation, "repeat")
                #             else config.parameter.evaluation.repeat
                #         ),
                #     )
                #     wandb_run.log(data=evaluation_results, commit=False)

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
