from generative_rl.configurations.halfcheetah_qgpo import config
from generative_rl.algorithms.qgpo import QGPO
from generative_rl.environments.gym_env import GymEnv
from generative_rl.datasets.qgpo import QGPOD4RLDataset
from generative_rl.utils.log import log

def qgpo_pipeline(config):

    qgpo = QGPO(config)

    #---------------------------------------
    # Customized train code ↓
    #---------------------------------------
    qgpo.train(create_env_func=GymEnv, create_dataset_func=QGPOD4RLDataset)
    #---------------------------------------
    # Customized train code ↑
    #---------------------------------------

    #---------------------------------------
    # Customized deploy code ↓
    #---------------------------------------
    agent = qgpo.deploy()
    deploy_env = GymEnv(config.deploy.env)
    for _ in range(config.deploy.num_deploy_steps):
        deploy_env.render()
        deploy_env.step(agent.act(deploy_env.observation))
    #---------------------------------------
    # Customized deploy code ↑
    #---------------------------------------

if __name__ == '__main__':
    log.info("config: \n{}".format(config))
    qgpo_pipeline(config)
