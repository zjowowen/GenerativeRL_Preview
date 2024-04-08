from typing import Any, Dict, List, Tuple, Union
from easydict import EasyDict
import copy
import torch
import torch.nn as nn
from tensordict import TensorDict
from generative_rl.rl_modules.value_network.q_network import DoubleQNetwork

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
            - config (:obj:`EasyDict`): The configuration dict.
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
            - action (:obj:`torch.Tensor`): The input action.
            - state (:obj:`torch.Tensor`): The input state.
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
            - action (:obj:`Union[torch.Tensor, TensorDict]`): The input action.
            - state (:obj:`Union[torch.Tensor, TensorDict]`): The input state.
        Returns:
            - q1 (:obj:`Union[torch.Tensor, TensorDict]`): The output of the first Q network.
            - q2 (:obj:`Union[torch.Tensor, TensorDict]`): The output of the second Q network.
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
            - action (:obj:`torch.Tensor`): The input action.
            - state (:obj:`torch.Tensor`): The input state.
            - reward (:obj:`torch.Tensor`): The input reward.
            - next_state (:obj:`torch.Tensor`): The input next state.
            - done (:obj:`torch.Tensor`): The input done.
            - fake_next_action (:obj:`torch.Tensor`): The input fake next action.
            - discount_factor (:obj:`float`): The discount factor.
        """
        with torch.no_grad():
            softmax = nn.Softmax(dim=1)
            next_energy = self.q_target(fake_next_action, torch.stack([next_state] * fake_next_action.shape[1], axis=1)).detach().squeeze()
            next_v = torch.sum(softmax(self.q_alpha * next_energy) * next_energy, dim=-1, keepdim=True)
        # Update Q function
        targets = reward + (1. - done.float()) * discount_factor * next_v.detach()
        q0, q1 = self.q.compute_double_q(action, state)
        q_loss = (torch.nn.functional.mse_loss(q0, targets) + torch.nn.functional.mse_loss(q1, targets)) / 2
        return q_loss
