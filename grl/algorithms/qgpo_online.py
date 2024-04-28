#############################################################
# This QGPO model is a modification implementation from https://github.com/ChenDRAG/CEP-energy-guided-diffusion
#############################################################

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
from grl.agents.qgpo import QGPOAgent
from grl.datasets import create_dataset
from grl.datasets.qgpo import QGPODataset
from grl.generative_models.diffusion_model.energy_conditional_diffusion_model import \
    EnergyConditionalDiffusionModel
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
            next_energy = self.q_target(fake_next_action, torch.stack([next_state] * fake_next_action.shape[1], axis=1)).detach().squeeze(dim=-1)
            next_v = torch.sum(softmax(self.q_alpha * next_energy) * next_energy, dim=-1, keepdim=True)
        # Update Q function
        targets = reward + (1. - done.float()) * discount_factor * next_v.detach()
        q0, q1 = self.q.compute_double_q(action, state)
        q_loss = (torch.nn.functional.mse_loss(q0, targets) + torch.nn.functional.mse_loss(q1, targets)) / 2
        return q_loss, torch.mean((q0+q1)/2.0)

class QGPOPolicy(nn.Module):

    def __init__(self, config: EasyDict):
        super().__init__()
        self.config = config
        self.device = config.device

        self.critic = QGPOCritic(config.critic)
        self.diffusion_model = EnergyConditionalDiffusionModel(config.diffusion_model, energy_model=self.critic)

    def forward(self, state: Union[torch.Tensor, TensorDict]) -> Union[torch.Tensor, TensorDict]:
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
            batch_size (:obj:`Union[torch.Size, int, Tuple[int], List[int]]`): The batch size.
            guidance_scale (:obj:`Union[torch.Tensor, float]`): The guidance scale.
            solver_config (:obj:`EasyDict`): The configuration for the ODE solver.
            t_span (:obj:`torch.Tensor`): The time span for the ODE solver or SDE solver.
        Returns:
            action (:obj:`Union[torch.Tensor, TensorDict]`): The output action.
        """
        return self.diffusion_model.sample(
            t_span = t_span,
            condition=state,
            batch_size=batch_size,
            guidance_scale=guidance_scale,
            with_grad=False,
            solver_config=solver_config
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
            t_span = t_span,
            condition=state,
            batch_size=batch_size,
            solver_config=solver_config
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
        
        return self.diffusion_model.score_matching_loss(action, state)

    def energy_guidance_loss(
            self,
            state: Union[torch.Tensor, TensorDict],
            fake_next_action: Union[torch.Tensor, TensorDict]
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
        return self.critic.q_loss(action, state, reward, next_state, done, fake_next_action, discount_factor)

class QGPOOnlineAlgorithm:

    def __init__(
        self,
        config:EasyDict = None,
        simulator = None,
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
        
        #---------------------------------------
        # Customized model initialization code ↓
        #---------------------------------------

        self.model = model if model is not None else torch.nn.ModuleDict()

        #---------------------------------------
        # Customized model initialization code ↑
        #---------------------------------------

    def train(
        self,
        config: EasyDict = None
    ):
        """
        Overview:
            Train the model using the given configuration. \
            A weight-and-bias run will be created automatically when this function is called.
        Arguments:
            config (:obj:`EasyDict`): The training configuration.
        """
        
        config = merge_two_dicts_into_newone(
            self.config.train if hasattr(self.config, "train") else EasyDict(),
            config
        ) if config is not None else self.config.train

        with wandb.init(
            project=config.project if hasattr(config, "project") else __class__.__name__,
            **config.wandb if hasattr(config, "wandb") else {}
        ) as wandb_run:
            config=merge_two_dicts_into_newone(EasyDict(wandb_run.config), config)
            wandb_run.config.update(config)
            self.config.train = config

            self.simulator = create_simulator(config.simulator) if hasattr(config, "simulator") else self.simulator
            self.dataset = create_dataset(config.dataset) if hasattr(config, "dataset") else self.dataset

            #---------------------------------------
            # Customized model initialization code ↓
            #---------------------------------------

            if hasattr(config.model, "QGPOPolicy"):
                self.model["QGPOPolicy"] = QGPOPolicy(config.model.QGPOPolicy)
                self.model["QGPOPolicy"].to(config.model.QGPOPolicy.device)
                if torch.__version__ >= "2.0.0":
                    self.model["QGPOPolicy"] = torch.compile(self.model["QGPOPolicy"])

            #---------------------------------------
            # Customized model initialization code ↑
            #---------------------------------------


            #---------------------------------------
            # Customized training code ↓
            #---------------------------------------


            behaviour_model_optimizer = torch.optim.Adam(
                self.model["QGPOPolicy"].diffusion_model.model.parameters(),
                lr=config.parameter.behaviour_policy.learning_rate,
            )

            q_optimizer = torch.optim.Adam(
                self.model["QGPOPolicy"].critic.q.parameters(),
                lr=config.parameter.critic.learning_rate,
            )

            energy_guidance_optimizer = torch.optim.Adam(
                self.model["QGPOPolicy"].diffusion_model.energy_guidance.parameters(),
                lr=config.parameter.energy_guidance.learning_rate,
            )

            better_than_baseline = True
            baseline_return = -200
            baseline_guidance_scale = 1.0

            for online_rl_iteration in track(range(config.parameter.online_rl.iterations), description="Online RL iteration"):

                def get_train_data(dataloader):
                    while True:
                        yield from dataloader

                def generate_fake_action(model, states, sample_per_state):
                    # model.eval()
                    fake_actions_sampled = []
                    for states in np.array_split(states, states.shape[0] // 4096 + 1):
                        fake_actions = model.sample(
                            state = states,
                            batch_size = sample_per_state,
                            guidance_scale = 0.0,
                            t_span = torch.linspace(0.0, 1.0, config.parameter.fake_data_t_span).to(config.model.QGPOPolicy.device) if config.parameter.fake_data_t_span is not None else None
                        )
                        fake_actions = torch.einsum("hbd->bhd", fake_actions)
                        fake_actions_sampled.append(fake_actions)
                    fake_actions = torch.cat(fake_actions_sampled, dim=0)
                    return fake_actions

                def evaluate(model):
                    nonlocal better_than_baseline
                    nonlocal baseline_return
                    nonlocal baseline_guidance_scale
                    evaluation_results = dict()
                    for guidance_scale in config.parameter.evaluation.guidance_scale:
                        def policy(obs: np.ndarray) -> np.ndarray:
                            obs = torch.tensor(obs, dtype=torch.float32, device=config.model.QGPOPolicy.device).unsqueeze(0)
                            action = model.sample(
                                state = obs,
                                guidance_scale=guidance_scale,
                                t_span = torch.linspace(0.0, 1.0, config.parameter.fake_data_t_span).to(config.model.QGPOPolicy.device) if config.parameter.fake_data_t_span is not None else None
                            ).squeeze(0).cpu().detach().numpy()
                            return action
                        evaluation_results[f"evaluation/guidance_scale:[{guidance_scale}]/total_return"] = self.simulator.evaluate(policy=policy, )[0]["total_return"]
                        log.info(f"Online RL iteration: {online_rl_iteration}, guidance_scale: {guidance_scale}, total_return: {evaluation_results[f'evaluation/guidance_scale:[{guidance_scale}]/total_return']}")

                        if evaluation_results[f"evaluation/guidance_scale:[{guidance_scale}]/total_return"] > baseline_return:
                            better_than_baseline = True
                            baseline_return = evaluation_results[f"evaluation/guidance_scale:[{guidance_scale}]/total_return"]
                            baseline_guidance_scale = guidance_scale

                    return evaluation_results
                
                def collect(model, num_steps, guidance_scale=1.0):
                    def policy(obs: np.ndarray) -> np.ndarray:
                        obs = torch.tensor(obs, dtype=torch.float32, device=config.model.QGPOPolicy.device).unsqueeze(0)
                        action = model.sample(
                            state = obs,
                            guidance_scale=guidance_scale,
                            t_span = torch.linspace(0.0, 1.0, config.parameter.fake_data_t_span).to(config.model.QGPOPolicy.device) if config.parameter.fake_data_t_span is not None else None
                        ).squeeze(0).cpu().detach().numpy()
                        return action
                    return self.simulator.collect_steps(policy=policy, num_steps=num_steps)

                if better_than_baseline:
                    if online_rl_iteration > 0:
                        self.dataset.drop_data(config.parameter.online_rl.drop_ratio)
                        self.dataset.load_data(collect(self.model["QGPOPolicy"], num_steps=config.parameter.online_rl.collect_steps, guidance_scale=baseline_guidance_scale))
                    else:
                        # self.dataset.load_data(collect(self.model["QGPOPolicy"], num_steps=config.parameter.online_rl.collect_steps_at_the_beginning, guidance_scale=1.0))
                        self.dataset.load_data(self.simulator.collect_steps(policy=None, num_steps=config.parameter.online_rl.collect_steps_at_the_beginning, random_policy=True))
                else:
                    self.dataset.load_data(collect(self.model["QGPOPolicy"], num_steps=config.parameter.online_rl.collect_steps, guidance_scale=baseline_guidance_scale))

                dataloader = DataLoader(
                    self.dataset,
                    batch_size=config.parameter.online_rl.batch_size,
                    shuffle=True,
                    collate_fn=None,
                )
    
                if online_rl_iteration == 0 or (online_rl_iteration + 1) % config.parameter.evaluation.evaluation_interval == 0:
                    better_than_baseline = False
                
                counter = 0
                behaviour_model_training_loss_sum = 0.0
                energy_guidance_loss_sum = 0.0
                q_loss_sum = 0.0
                q_mean_sum = 0.0



                for data in dataloader:

                    behaviour_model_training_loss = self.model["QGPOPolicy"].behaviour_policy_loss(data['a'], data['s'])
                    behaviour_model_optimizer.zero_grad()
                    behaviour_model_training_loss.backward()
                    behaviour_model_optimizer.step()

                    behaviour_model_training_loss_sum += behaviour_model_training_loss.item()

                    data['fake_a_'] = generate_fake_action(
                        self.model["QGPOPolicy"],
                        data['s_'],
                        config.parameter.sample_per_state
                    )

                    q_loss, q_mean = self.model["QGPOPolicy"].q_loss(
                        data['a'], data['s'], data['r'], data['s_'], data['d'], data['fake_a_'],
                        discount_factor=config.parameter.critic.discount_factor
                    )

                    q_optimizer.zero_grad()
                    q_loss.backward()
                    q_optimizer.step()
                    
                    # Update target
                    for param, target_param in zip(self.model["QGPOPolicy"].critic.parameters(), self.model["QGPOPolicy"].critic.q_target.parameters()):
                        target_param.data.copy_(
                            config.parameter.critic.update_momentum * param.data + (1 - config.parameter.critic.update_momentum) * target_param.data
                        )

                    q_loss_sum += q_loss.item()
                    q_mean_sum += q_mean.item()

                    data['fake_a'] = generate_fake_action(
                        self.model["QGPOPolicy"],
                        data['s'],
                        config.parameter.sample_per_state
                    )

                    energy_guidance_loss = self.model["QGPOPolicy"].energy_guidance_loss(data['s'], data['fake_a'])
                    energy_guidance_optimizer.zero_grad()
                    energy_guidance_loss.backward()
                    energy_guidance_optimizer.step()

                    energy_guidance_loss_sum += energy_guidance_loss.item()
                    counter += 1

                wandb_run.log(
                    data=dict(
                        behaviour_model_training_loss=behaviour_model_training_loss_sum / counter,
                        q_loss=q_loss_sum / counter,
                        q_mean=q_mean_sum / counter,
                        energy_guidance_loss=energy_guidance_loss_sum / counter,
                    ),
                    commit=False)

                if online_rl_iteration == 0 or (online_rl_iteration + 1) % config.parameter.evaluation.evaluation_interval == 0:
                    evaluation_results = evaluate(self.model["QGPOPolicy"])
                    wandb_run.log(data=evaluation_results, commit=False)

                wandb_run.log(
                    data=dict(
                        online_rl_iteration=online_rl_iteration,
                    ),
                    commit=True)

            #---------------------------------------
            # Customized training code ↑
            #---------------------------------------

            wandb.finish()


    def deploy(self, config:EasyDict = None) -> QGPOAgent:
        
        if config is not None:
            config = merge_two_dicts_into_newone(self.config.deploy, config)
        else:
            config = self.config.deploy

        assert "QGPOPolicy" in self.model, "The model must be trained first."
        return QGPOAgent(
            config=config,
            model=copy.deepcopy(self.model),
        )
