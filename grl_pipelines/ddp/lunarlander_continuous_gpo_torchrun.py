# torchrun --nproc_per_node=1 --nnodes=gpu grl_pipelines/ddp/lunarlander_continuous_gpo_torchrun.py
import os

import gym
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from grl.algorithms.ddp.gp import GPAlgorithm
from grl.datasets import GPOCustomizedDataset
from grl.utils.log import log
from grl_pipelines.ddp.configurations.new_lunarlander_continuous_gpo import make_config


def gp_pipeline(config):

    gp = GPAlgorithm(
        config,
        dataset=GPOCustomizedDataset(
            numpy_data_path="./data.npz", device=config.train.device
        ),
    )

    # ---------------------------------------
    # Customized train code ↓
    # ---------------------------------------
    gp.train()
    # ---------------------------------------
    # Customized train code ↑
    # ---------------------------------------

    # ---------------------------------------
    # Customized deploy code ↓
    # ---------------------------------------
    if torch.distributed.get_rank() == 0:
        agent = gp.deploy()
        env = gym.make(config.deploy.env.env_id)
        observation = env.reset()
        for _ in range(config.deploy.num_deploy_steps):
            env.render()
            observation, reward, done, _ = env.step(agent.act(observation))
    # ---------------------------------------
    # Customized deploy code ↑
    # ---------------------------------------


if __name__ == "__main__":
    torch.distributed.init_process_group("nccl")
    device = torch.distributed.get_rank() % torch.cuda.device_count()
    torch.cuda.set_device(device)
    config = make_config(device=device)
    gp_pipeline(config)
