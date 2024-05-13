# torchrun --nproc_per_node=8 this file
import gym
import d4rl
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from grl.algorithms.ddp.gp import GPAlgorithm
from grl_pipelines.ddp.configurations.d4rl_hopper_medium_expert_v2_gpo_vpsde import (
    make_config,
)


def gp_pipeline(config):

    gp = GPAlgorithm(config)

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
