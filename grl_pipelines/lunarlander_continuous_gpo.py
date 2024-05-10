import gym

from grl.algorithms.gpo import GPOAlgorithm
from grl.datasets import GPOCustomizedDataset
from grl.utils.log import log
from grl_pipelines.configurations.lunarlander_continuous_gpo import (
    config,
)


def gpo_pipeline(config):

    gpo = GPOAlgorithm(
        config,
        dataset=GPOCustomizedDataset(
            numpy_data_path="./data.npz", device=config.train.device
        ),
    )

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
    agent = gpo.deploy()
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
    gpo_pipeline(config)
