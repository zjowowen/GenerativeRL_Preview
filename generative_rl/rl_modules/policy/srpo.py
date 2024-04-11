#############################################################
# This SRPO model is a modification implementation from https://github.com/ChenDRAG/CEP-energy-guided-diffusion
#############################################################

from typing import Any, Dict, List, Tuple, Union
from easydict import EasyDict
import torch
import torch.nn as nn
from tensordict import TensorDict
from generative_rl.machine_learning.generative_models.diffusion_model.srpo_conditional_diffusion_model import SRPOConditionalDiffusionModel
from generative_rl.rl_modules.value_network.srpo import SRPOCritic 
from generative_rl.machine_learning.modules import MLP,my_mlp

class Dirac_Policy(nn.Module):
    def __init__(self, action_dim, state_dim, layer=2):
        super().__init__()
        self.net = my_mlp([state_dim] + [256]*layer + [action_dim], output_activation=nn.Tanh)
        # self.net = MLP(
                #     in_channels=state_dim,
                #     hidden_channels=256,
                #     out_channels=action_dim,
                #     layer_num=layer+1,
                #     activation=nn.ReLU,
                #     output_activation=nn.Tanh,
                #     output_norm=False,
                # )
    def forward(self, state):
        return self.net(state)
    
    def select_actions(self, state):
        return self(state)
    
    def sample(self, size:None, state:None):
        return self.forward(state)

class SRPOPolicy(nn.Module):
    def __init__(self, config: EasyDict):
        super().__init__()
        self.config = config
        self.device = config.device

        self.deter_policy = Dirac_Policy(**config.policy_model)
        self.critic = SRPOCritic(config.critic)
        self.diffusion_model = SRPOConditionalDiffusionModel(config=config.diffusion_model, value_model=self.critic,distribution= self.deter_policy  )
        self.q = nn.ModuleList([SRPOCritic(config.critic)])

    def forward(self, state: Union[torch.Tensor, TensorDict]) -> Union[torch.Tensor, TensorDict]:
        """
        Overview:
            Return the output of QGPO policy, which is the action conditioned on the state.
        Arguments:
            - state (:obj:`Union[torch.Tensor, TensorDict]`): The input state.
        Returns:
            - action (:obj:`Union[torch.Tensor, TensorDict]`): The output action.
        """
        return self.deter_policy.select_actions(state)
    
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
        """
        v_loss, next_v =self.q[0].v_loss(data)
        return v_loss,next_v

    def q_loss(
            self,
            data,
            next_v,
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

        loss=self.q[0].q_loss(data,next_v,discount_factor)
        return loss

    def srpo_actor_loss(
            self,
            data,
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
        state = data['s']
        action = self.deter_policy(state)
        loss = self.diffusion_model.srpo_loss(action,state)
        return loss