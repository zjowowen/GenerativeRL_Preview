#############################################################
# This SRPO model is a modification implementation from https://github.com/ChenDRAG/CEP-energy-guided-diffusion
#############################################################

from typing import Any, Dict, List, Tuple, Union
from easydict import EasyDict
import torch
import torch.nn as nn
from tensordict import TensorDict
from generative_rl.machine_learning.generative_models.diffusion_model.diffusion_model import DiffusionModel
from generative_rl.rl_modules.value_network.srpo import SRPOCritic 
from generative_rl.machine_learning.modules import MLP,my_mlp

class Dirac_Policy(nn.Module):
    def __init__(self, action_dim, state_dim, layer=2):
        super().__init__()
        self.net = my_mlp([state_dim] + [256]*layer + [action_dim], output_activation=nn.Tanh)

    def forward(self, state):
        return self.net(state)
    def select_actions(self, state):
        return self(state)

class SRPOPolicy(nn.Module):
    def __init__(self, config: EasyDict):
        super().__init__()
        self.config = config
        self.device = config.device

        self.deter_policy = Dirac_Policy(**config.policy_model)
        self.critic = SRPOCritic(config.critic)
        self.diffusion_model = DiffusionModel(config.diffusion_model)
        self.q = []
        self.q.append(SRPOCritic(config.critic))

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
    
    def sample(
            self,
            state: Union[torch.Tensor, TensorDict],
            batch_size: Union[torch.Size, int, Tuple[int], List[int]] = None,
            guidance_scale: Union[torch.Tensor, float] = torch.tensor(1.0),
            solver_config: EasyDict = None,
        ) -> Union[torch.Tensor, TensorDict]:
        """
        Overview:
            Return the output of QGPO policy, which is the action conditioned on the state.
        Arguments:
            - state (:obj:`Union[torch.Tensor, TensorDict]`): The input state.
            - guidance_scale (:obj:`Union[torch.Tensor, float]`): The guidance scale.
            - solver_config (:obj:`EasyDict`): The configuration for the ODE solver.
        Returns:
            - action (:obj:`Union[torch.Tensor, TensorDict]`): The output action.
        """
        return self.diffusion_model.sample(
            t_span = torch.linspace(0.0, 1.0, 32).to(self.device),
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
        ) -> Union[torch.Tensor, TensorDict]:
        """
        Overview:
            Return the output of behaviour policy, which is the action conditioned on the state.
        Arguments:
            - state (:obj:`Union[torch.Tensor, TensorDict]`): The input state.
            - solver_config (:obj:`EasyDict`): The configuration for the ODE solver.
        Returns:
            - action (:obj:`Union[torch.Tensor, TensorDict]`): The output action.
        """
        return self.diffusion_model.sample_without_energy_guidance(
            t_span = torch.linspace(0.0, 1.0, 32).to(self.device),
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
            Calculate the behaviour policy loss.
        Arguments:
            - action (:obj:`torch.Tensor`): The input action.
            - state (:obj:`torch.Tensor`): The input state.
        """
        
        return self.diffusion_model.score_matching_loss(action, state)


    def v_loss(
            self,
            data,
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

        loss=self.q[0].v_loss(data)
        return loss

    def q_loss(
            self,
            data,
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

        loss=self.q[0].q_loss(data)
        return loss
