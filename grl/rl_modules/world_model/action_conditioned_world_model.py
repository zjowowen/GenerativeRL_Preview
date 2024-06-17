import torch

from easydict import EasyDict
from torch import nn

from grl.generative_models import get_generative_model

class ActionConditionedWorldModel(nn.Module):
    """
    Overview:
        World model.
    Interfaces:
        ``__init__``, ``forward``
    """

    def __init__(
        self,
        config: EasyDict,
    ):
        """
        Overview:
            Initialize the world model.
        Arguments:
            - config (:obj:`EasyDict`): The configuration.
        """
        super(ActionConditionedWorldModel, self).__init__()

        self.config = config
        self.model = get_generative_model(config.model_type)(config.model_config)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Overview:
            Return the next state given the current state, action, and condition.
        Arguments:
            - state (:obj:`torch.Tensor`): The current state.
            - action (:obj:`torch.Tensor`): The action.
            - condition (:obj:`torch.Tensor`): The condition.
        """

        return self.model.sample(
            x0=state, 
            condition=action)
