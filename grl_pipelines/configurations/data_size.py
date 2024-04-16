"""
Overview:
    This file is used to show the data size examples of the input tensor.
"""

import torch

data_size = 1
data_size = 2
data_size = [2, 2]
data_size = (2, 2)
data_size = torch.Size([2])
data_size = torch.Size([2, 2])
data_size = dict(
    a = 1,
    b = 2,
    c = [2, 2],
    d = (2, 2),
    e = torch.Size([2]),
    f = torch.Size([2, 2]),
    g = dict(
        a = 1,
        b = 2,
        c = [2, 2],
        d = (2, 2),
        e = torch.Size([2]),
        f = torch.Size([2, 2]),
    ),
)