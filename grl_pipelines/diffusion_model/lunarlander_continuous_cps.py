import gym

from grl.algorithms.cps_tmp import CPSAlgorithm
from grl.utils.log import log
from grl_pipelines.diffusion_model.configurations.lunarlander_continuous_cps import (
    config,
)


def cps_pipeline(config):

    cps = CPSAlgorithm(config)

    # ---------------------------------------
    # Customized train code ↓
    # ---------------------------------------
    cps.train()
    # ---------------------------------------
    # Customized train code ↑
    # ---------------------------------------

    # ---------------------------------------
    # Customized deploy code ↓
    # ---------------------------------------
    agent = cps.deploy()
    env = gym.make(config.deploy.env.env_id)
    env.reset()
    for _ in range(config.deploy.num_deploy_steps):
        env.render()
        env.step(agent.act(env.observation))
    # ---------------------------------------
    # Customized deploy code ↑
    # ---------------------------------------


if __name__ == "__main__":
    log.info("config: \n{}".format(config))
    cps_pipeline(config)
