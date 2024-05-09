import gym

from grl.algorithms.sac import SACAlgorithm
from grl.utils.log import log
from grl_pipelines.gaussian_policy.configurations.halfcheetah_sac import config


def sac_pipeline(config):

    sac = SACAlgorithm(config)

    # ---------------------------------------
    # Customized train code ↓
    # ---------------------------------------
    sac.train()
    # ---------------------------------------
    # Customized train code ↑
    # ---------------------------------------

    # ---------------------------------------
    # Customized deploy code ↓
    # ---------------------------------------
    agent = sac.deploy()
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
    sac_pipeline(config)
