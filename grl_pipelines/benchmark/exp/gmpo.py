import gym
import d4rl
import numpy as np

from grl.algorithms.gmpo import GMPOAlgorithm
from grl.utils.log import log
from grl_pipelines.benchmark.gmpo.gvp.halfcheetah_medium_expert import config

def gp_pipeline(config):

    gp = GMPOAlgorithm(config)

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
    total_reward_list = []
    for i in range(100):
        observation = env.reset()
        total_reward = 0
        while True:
            # env.render()
            observation, reward, done, _ = env.step(agent.act(observation))
            total_reward += reward
            if done:
                observation = env.reset()
                print(f"Episode {i}, total_reward: {total_reward}")
                total_reward_list.append(total_reward)
                break

    print(
        f"Average total reward: {np.mean(total_reward_list)}, std: {np.std(total_reward_list)}"
    )

    # ---------------------------------------
    # Customized deploy code ↑
    # ---------------------------------------

log.info("config: \n{}".format(config))
gp_pipeline(config)