#############################################################
# This QGPO model is a modification implementation from https://github.com/ChenDRAG/CEP-energy-guided-diffusion
#############################################################

from typing import Any, Dict, List, Tuple, Union
from easydict import EasyDict
import torch
import torch.nn as nn
from tensordict import TensorDict
from generative_rl.machine_learning.generative_models.diffusion_model.score_model import ScoreModel
from generative_rl.machine_learning.generative_models.diffusion_model.energy_conditional_diffusion_model import EnergyConditionalDiffusionModel
from generative_rl.numerical_methods.numerical_solvers.dpm_solver import NoiseScheduleVP
from generative_rl.rl_modules.value_network.qgpo import QGPOCritic



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
            - state (:obj:`Union[torch.Tensor, TensorDict]`): The input state.
        Returns:
            - action (:obj:`Union[torch.Tensor, TensorDict]`): The output action.
        """
        return self.sample(state)
    
    def sample(self, state: Union[torch.Tensor, TensorDict]) -> Union[torch.Tensor, TensorDict]:
        """
        Overview:
            Return the output of QGPO policy, which is the action conditioned on the state.
        Arguments:
            - state (:obj:`Union[torch.Tensor, TensorDict]`): The input state.
        Returns:
            - action (:obj:`Union[torch.Tensor, TensorDict]`): The output action.
        """
        return self.diffusion_model.sample(condition=state)

    def behavior_policy_sample(self, state: Union[torch.Tensor, TensorDict]) -> Union[torch.Tensor, TensorDict]:
        """
        Overview:
            Return the output of behavior policy, which is the action conditioned on the state.
        Arguments:
            - state (:obj:`Union[torch.Tensor, TensorDict]`): The input state.
        Returns:
            - action (:obj:`Union[torch.Tensor, TensorDict]`): The output action.
        """
        return self.diffusion_model.sample_without_energy_guidance(condition=state)
    

    def compute_q(
            self,
            state: Union[torch.Tensor, TensorDict],
            action: Union[torch.Tensor, TensorDict],
        ) -> torch.Tensor:
        """
        Overview:
            Calculate the Q value.
        Arguments:
            - state (:obj:`Union[torch.Tensor, TensorDict]`): The input state.
            - action (:obj:`Union[torch.Tensor, TensorDict]`): The input action.
        Returns:
            - q (:obj:`torch.Tensor`): The Q value.
        """

        return self.critic(action, state)

    def behaviour_policy_loss(
            self,
            action: Union[torch.Tensor, TensorDict],
            state: Union[torch.Tensor, TensorDict],
        ):
        """
        Overview:
            Calculate the behavior policy loss.
        Arguments:
            - action (:obj:`torch.Tensor`): The input action.
            - state (:obj:`torch.Tensor`): The input state.
        """
        
        return self.diffusion_model.unconditional_loss(action, state)

    def qt_loss(
            self,
            state: Union[torch.Tensor, TensorDict],
            fake_next_action: Union[torch.Tensor, TensorDict]
        ) -> torch.Tensor:
        """
        Overview:
            Calculate the energy guidance loss of QGPO.
        Arguments:
            - state (:obj:`Union[torch.Tensor, TensorDict]`): The input state.
            - fake_next_action (:obj:`Union[torch.Tensor, TensorDict]`): The input fake next action.
        """

        return self.diffusion_model.energy_guidance_loss(state, fake_next_action)

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
            - action (:obj:`torch.Tensor`): The input action.
            - state (:obj:`torch.Tensor`): The input state.
            - reward (:obj:`torch.Tensor`): The input reward.
            - next_state (:obj:`torch.Tensor`): The input next state.
            - done (:obj:`torch.Tensor`): The input done.
            - fake_next_action (:obj:`torch.Tensor`): The input fake next action.
            - discount_factor (:obj:`float`): The discount factor.
        """
        return self.critic.q_loss(action, state, reward, next_state, done, fake_next_action, discount_factor)
