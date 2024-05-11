import gym

from grl.algorithms.gpg import GPGAlgorithm
from grl.datasets import GPOCustomizedDataset
from grl.utils.log import log
from grl_pipelines.configurations.lunarlander_continuous_gpg_icfm import (
    config,
)


def gpg_pipeline(config):

    gpg = GPGAlgorithm(
        config,
        dataset=GPOCustomizedDataset(
            numpy_data_path="./data.npz", device=config.train.device
        ),
    )

    # ---------------------------------------
    # Customized train code ↓
    # ---------------------------------------
    gpg.train()
    # ---------------------------------------
    # Customized train code ↑
    # ---------------------------------------

    # ---------------------------------------
    # Customized deploy code ↓
    # ---------------------------------------
    agent = gpg.deploy()
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
    gpg_pipeline(config)
