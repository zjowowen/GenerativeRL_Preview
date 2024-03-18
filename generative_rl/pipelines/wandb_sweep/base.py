import wandb
from generative_rl.configurations.wandb_sweep.base import sweep_config
from generative_rl.configurations.base import config
from generative_rl.algorithms.base import BaseAlgorithm
from generative_rl.simulators.base import BaseEnv
from generative_rl.datasets.base import BaseDataset
from generative_rl.utils.log import log


def base_wandb_sweep_pipeline(config, sweep_config):
    """
    Overview:
        The base pipeline for training an algorithm by using weight-and-bias sweep.
    Arguments:
        - config (:obj:`EasyDict`): The configuration.

    .. note::
        This pipeline is for demonstration purposes only.
    """

    def train_func():
        algo = BaseAlgorithm(config)
        algo.train()

    sweep_id = wandb.sweep(config=sweep_config, project="generative_rl")
    wandb.agent(sweep_id, function=train_func)

if __name__ == '__main__':
    log.info("config: \n{}".format(config))
    log.info("sweep config: \n{}".format(sweep_config))
    base_wandb_sweep_pipeline(config, sweep_config)
