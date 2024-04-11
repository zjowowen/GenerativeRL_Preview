from typing import Tuple, Union
from easydict import EasyDict
import copy
import torch
import torch.nn as nn
from tensordict import TensorDict
from generative_rl.machine_learning.modules import MLP,my_mlp
from generative_rl.machine_learning.encoders import get_encoder
from generative_rl.machine_learning.modules import get_module


class QNetwork(nn.Module):

    def __init__(self, config: EasyDict):
        super().__init__()
        self.config = config
        self.model = torch.nn.ModuleDict()
        if hasattr(config, "action_encoder"):
            self.model["action_encoder"] = get_encoder(config.action_encoder.type)(**config.action_encoder.args)
        else:
            self.model["action_encoder"] = torch.nn.Identity()
        if hasattr(config, "state_encoder"):
            self.model["state_encoder"] = get_encoder(config.state_encoder.type)(**config.state_encoder.args)
        else:
            self.model["state_encoder"] = torch.nn.Identity()
        #TODO
        # specific backbone network
        self.model["backbone"] = get_module(config.backbone.type)(**config.backbone.args)


    def forward(
        self,
        action: Union[torch.Tensor, TensorDict],
        state: Union[torch.Tensor, TensorDict],
    ) -> torch.Tensor:
        """
        Overview:
            Return output of Q networks.
        Arguments:
            - action (:obj:`Union[torch.Tensor, TensorDict]`): The input action.
            - state (:obj:`Union[torch.Tensor, TensorDict]`): The input state.
        Returns:
            - q (:obj:`Union[torch.Tensor, TensorDict]`): The output of Q network.
        """
        action_embedding = self.model["action_encoder"](action)
        state_embedding = self.model["state_encoder"](state)
        return self.model["backbone"](action_embedding, state_embedding)


class DoubleQNetwork(nn.Module):
    """
    Overview:
        Double Q network, which has two Q networks.
    Interfaces:
        ``__init__``, ``forward``, ``compute_double_q``, ``compute_mininum_q``
    """

    def __init__(self, config: EasyDict):
        super().__init__()

        self.model = torch.nn.ModuleDict()
        self.model["q1"] = QNetwork(config)
        self.model["q2"] = QNetwork(config)

    def compute_double_q(
            self,
            action: Union[torch.Tensor, TensorDict],
            state: Union[torch.Tensor, TensorDict],
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

        return self.model["q1"](action, state), self.model["q2"](action, state)

    def compute_mininum_q(
            self,
            action: Union[torch.Tensor, TensorDict],
            state: Union[torch.Tensor, TensorDict],
        ) -> torch.Tensor:
        """
        Overview:
            Return the minimum output of two Q networks.
        Arguments:
            - action (:obj:`Union[torch.Tensor, TensorDict]`): The input action.
            - state (:obj:`Union[torch.Tensor, TensorDict]`): The input state.
        Returns:
            - minimum_q (:obj:`Union[torch.Tensor, TensorDict]`): The minimum output of Q network.
        """

        return torch.min(*self.compute_double_q(action, state))


    def forward(
            self,
            action: Union[torch.Tensor, TensorDict],
            state: Union[torch.Tensor, TensorDict],
        ) -> torch.Tensor:
        """
        Overview:
            Return the minimum output of two Q networks.
        Arguments:
            - action (:obj:`Union[torch.Tensor, TensorDict]`): The input action.
            - state (:obj:`Union[torch.Tensor, TensorDict]`): The input state.
        Returns:
            - minimum_q (:obj:`Union[torch.Tensor, TensorDict]`): The minimum output of Q network.
        """

        return self.compute_mininum_q(action, state)


class TwinQ(nn.Module):
    def __init__(self, action_dim, state_dim, layers=2):
        super().__init__()
        dims = [state_dim + action_dim] +[256]*layers +[1]
        # dims = [state_dim + action_dim, 256, 256, 1] #
        self.q1 = my_mlp(dims)
        self.q2 = my_mlp(dims)
        # self.q1 = MLP(
        #     in_channels=state_dim,
        #     hidden_channels=256,
        #     out_channels=action_dim,
        #     layer_num=layer + 1,
        #     activation=nn.ReLU,
        #     output_activation=nn.Tanh,
        #     output_norm=False,
        # )
        # self.q2 = MLP(
        #     in_channels=state_dim,
        #     hidden_channels=256,
        #     out_channels=action_dim,
        #     layer_num=layer + 1,
        #     activation=nn.ReLU,
        #     output_activation=nn.Tanh,
        #     output_norm=False,
        # )
    def both(self, action, condition=None):
        as_ = torch.cat([action, condition], -1) if condition is not None else action
        return self.q1(as_), self.q2(as_)

    def forward(self, action, condition=None):
        return torch.min(*self.both(action, condition))
