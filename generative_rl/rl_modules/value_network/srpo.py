from typing import Any, Dict, List, Tuple, Union
from easydict import EasyDict
import copy
import torch
import torch.nn as nn
from tensordict import TensorDict
from generative_rl.rl_modules.value_network.q_network import DoubleQNetwork,TwinQ
from generative_rl.machine_learning.modules import MLP,my_mlp

def asymmetric_l2_loss(u, tau):
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)

class ValueFunction(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        dims = [state_dim, 256, 256, 1]
        self.v = my_mlp(dims)
        # self.v = MLP(
        #     in_channels=state_dim,
        #     hidden_channels=256,
        #     out_channels=1,
        #     layer_num=3,
        #     activation=nn.ReLU,
        #     output_activation=None,
        #     output_norm=False,
        # )
    def forward(self, state):
        return self.v(state)

class SRPOCritic(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        adim = config.adim
        sdim = config.sdim
        layers = config.layers
        self.q0 = TwinQ(adim, sdim, layers=layers)
        self.q0_target = copy.deepcopy(self.q0)
        self.vf = ValueFunction(sdim)
        self.discount = 0.99
        self.tau = 0.7
        # self.tau = 0.9 if "maze" in config.env else 0.7


    def v_loss(self, data):
        s = data["s"]
        a = data["a"]
        r = data["r"]
        s_ = data["s_"]
        d = data["d"]
        with torch.no_grad():
            target_q = self.q0_target(a, s).detach()
            next_v = self.vf(s_).detach()

        # Update value function
        v = self.vf(s)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, self.tau)
        return v_loss , next_v
        

    def q_loss(self, data,next_v,discount):
        # Update Q function
        s = data["s"]
        a = data["a"]
        r = data["r"]
        d = data["d"]
        targets = r + (1. - d.float()) * discount * next_v.detach()
        qs = self.q0.both(a, s)
        q_loss = sum(torch.nn.functional.mse_loss(q, targets) for q in qs) / len(qs)
        return q_loss  
