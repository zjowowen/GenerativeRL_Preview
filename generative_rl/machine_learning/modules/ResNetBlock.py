import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from generative_rl.machine_learning.modules import get_module

class MLPResNetBlock(nn.Module):
    """MLPResNet block."""
    def __init__(self, features, act, dropout_rate=None, use_layer_norm=False):
        super(MLPResNetBlock, self).__init__()
        self.features = features
        self.act = act
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm

        if self.use_layer_norm:
            self.layer_norm = nn.LayerNorm(features)

        self.fc1 = nn.Linear(features, features * 4)
        self.fc2 = nn.Linear(features * 4, features)
        self.residual = nn.Linear(features, features)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate is not None and dropout_rate > 0.0 else None

    def forward(self, x, training=False):
        residual = x
        if self.dropout is not None:
            x = self.dropout(x)

        if self.use_layer_norm:
            x = self.layer_norm(x)

        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)

        if residual.shape != x.shape:
            residual = self.residual(residual)

        return residual + x

class MLPResNet(nn.Module):
    def __init__(self, num_blocks, input_dim, output_dim, dropout_rate=None, use_layer_norm=False, hidden_dim=256, activations=F.relu):
        super(MLPResNet, self).__init__()
        self.num_blocks = num_blocks
        self.out_dim = output_dim
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm
        self.hidden_dim = hidden_dim
        self.activations = activations

        self.fc = nn.Linear(input_dim+128, self.hidden_dim)

        self.blocks = nn.ModuleList([MLPResNetBlock(self.hidden_dim, self.activations, self.dropout_rate, self.use_layer_norm)
                                     for _ in range(self.num_blocks)])

        self.out_fc = nn.Linear(self.hidden_dim, self.out_dim)

    def forward(self, x):
        x = self.fc(x)

        for block in self.blocks:
            x = block(x)

        x = self.activations(x)
        x = self.out_fc(x)

        return x