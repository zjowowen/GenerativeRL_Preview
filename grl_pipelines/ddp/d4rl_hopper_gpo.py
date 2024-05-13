import os

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "23333"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
rank_list = [0]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
    [str(rank_list[i]) for i in range(len(rank_list))]
)

import gym
import d4rl

import torch
import torch.multiprocessing as mp

from grl.algorithms.ddp.gpo import GPOAlgorithm
from grl.datasets import GPOCustomizedDataset
from grl.utils.log import log
from grl_pipelines.ddp.configurations.d4rl_hopper_gpo import (
    make_config,
)


def gpo_pipeline(config):

    gpo = GPOAlgorithm(config)

    # ---------------------------------------
    # Customized train code ↓
    # ---------------------------------------
    gpo.train()
    # ---------------------------------------
    # Customized train code ↑
    # ---------------------------------------

    # ---------------------------------------
    # Customized deploy code ↓
    # ---------------------------------------
    if torch.distributed.get_rank() == 0:
        agent = gpo.deploy()
        env = gym.make(config.deploy.env.env_id)
        observation = env.reset()
        for _ in range(config.deploy.num_deploy_steps):
            env.render()
            observation, reward, done, _ = env.step(agent.act(observation))
    # ---------------------------------------
    # Customized deploy code ↑
    # ---------------------------------------


def main(rank, world_size):
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
    device = torch.distributed.get_rank() % torch.cuda.device_count()
    torch.cuda.set_device(device)
    config = make_config(device=device)
    log.info(
        f"Starting rank={torch.distributed.get_rank()}, GPU:[{rank_list[torch.distributed.get_rank()]}], world_size={torch.distributed.get_world_size()}."
    )

    log.info("config: \n{}".format(config))
    gpo_pipeline(config)


def parallel_process(rank, world_size):
    main(rank, world_size)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    world_size = len(rank_list)
    mp.spawn(parallel_process, args=(world_size,), nprocs=world_size, join=True)
