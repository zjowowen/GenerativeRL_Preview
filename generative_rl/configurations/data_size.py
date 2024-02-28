import torch


base_data_size = 1
base_data_size = 2
base_data_size = [2, 2]
base_data_size = (2, 2)
base_data_size = torch.Size([2])
base_data_size = torch.Size([2, 2])
base_data_size = dict(
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