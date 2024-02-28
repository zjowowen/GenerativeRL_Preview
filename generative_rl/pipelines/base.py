from generative_rl.configurations.base import config
from generative_rl.algorithms.base import BaseAlgorithm
from generative_rl.environments.base import BaseEnv
from generative_rl.datasets.base import BaseDataset
from generative_rl.utils.log import log


def base_pipeline(config):
    """
    Overview:
        The base pipeline for training and deploying an algorithm.
    Arguments:
        - config (:obj:`EasyDict`): The configuration, which must contain the following keys:
            - train (:obj:`EasyDict`): The training configuration.
            - train.env (:obj:`EasyDict`): The training environment configuration.
            - train.dataset (:obj:`EasyDict`): The training dataset configuration.
            - deploy (:obj:`EasyDict`): The deployment configuration.
            - deploy.env (:obj:`EasyDict`): The deployment environment configuration.
            - deploy.num_deploy_steps (:obj:`int`): The number of deployment steps.
    .. note::
        This pipeline is for demonstration purposes only.
    """

    env = BaseEnv(config.train.env)
    dataset = BaseDataset(config.train.dataset)
    algo = BaseAlgorithm(env=env, dataset=dataset)

    #---------------------------------------
    # Customized train code ↓
    #---------------------------------------
    algo.train(config=config.train)
    #---------------------------------------
    # Customized train code ↑
    #---------------------------------------

    #---------------------------------------
    # Customized deploy code ↓
    #---------------------------------------
    agent = algo.deploy(config=config.deploy)
    deploy_env = BaseEnv(config.deploy.env)
    for _ in range(config.deploy.num_deploy_steps):
        deploy_env.render()
        deploy_env.step(agent.act(deploy_env.observation))
    #---------------------------------------
    # Customized deploy code ↑
    #---------------------------------------

def base_pipeline_for_multi_envs(config):
    """
    Overview:
        The base pipeline for training and deploying an algorithm with multiple environments or datasets.
    Arguments:
        - config (:obj:`EasyDict`): The configuration, which must contain the following keys:
            - train (:obj:`EasyDict`): The training configuration.
            - train.env (:obj:`EasyDict`): The training environment configuration.
            - train.dataset (:obj:`EasyDict`): The training dataset configuration.
            - deploy (:obj:`EasyDict`): The deployment configuration.
            - deploy.env (:obj:`EasyDict`): The deployment environment configuration.
            - deploy.num_deploy_steps (:obj:`int`): The number of deployment steps.
    .. note::
        This pipeline is for demonstration purposes only.
    """

    algo = BaseAlgorithm(config)

    #---------------------------------------
    # Customized train code ↓
    #---------------------------------------
    algo.train(create_env_func=BaseEnv, create_dataset_func=BaseDataset)
    #---------------------------------------
    # Customized train code ↑
    #---------------------------------------

    #---------------------------------------
    # Customized deploy code ↓
    #---------------------------------------
    agent = algo.deploy()
    deploy_env = BaseEnv(config.deploy.env)
    for _ in range(config.deploy.num_deploy_steps):
        deploy_env.render()
        deploy_env.step(agent.act(deploy_env.observation))
    #---------------------------------------
    # Customized deploy code ↑
    #---------------------------------------

if __name__ == '__main__':
    log.info("config: \n{}".format(config))
    base_pipeline(config)
