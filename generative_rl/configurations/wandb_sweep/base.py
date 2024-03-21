"""
Overview:
    This file is used to show the configuration for wandb sweep.
"""

from easydict import EasyDict

sweep_config = EasyDict(
    name = "base-sweep",
    metric = dict(
        name = "average_return",
        goal = "maximize",
    ),
    method = "grid",
    parameters = dict(
        a = dict(
            values = [1, 2, 3, 4],
        ),
        b = dict(
            parameters = dict(
                b1 = dict(
                    values = [1, 2,],
                ),
            ),
        ),
    ),
)
