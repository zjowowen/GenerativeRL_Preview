import gym
import d4rl
from grl.algorithms.gp import GPAlgorithm
from grl.datasets import GPOCustomizedDataset
from grl.utils.log import log
from grl_pipelines.configurations.d4rl_walker2d_medium_expert_v2_gpg_gvp import (
    config,
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
    log.info("config: \n{}".format(config))
    gp_pipeline(config)
