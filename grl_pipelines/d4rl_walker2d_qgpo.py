from grl_pipelines.configurations.walker2d_qgpo import config
from grl.algorithms.qgpo import QGPOAlgorithm
from grl.utils.log import log
import gym

def qgpo_pipeline(config):

    qgpo = QGPOAlgorithm(config)

    #---------------------------------------
    # Customized train code ↓
    #---------------------------------------
    qgpo.train()
    #---------------------------------------
    # Customized train code ↑
    #---------------------------------------

    #---------------------------------------
    # Customized deploy code ↓
    #---------------------------------------
    agent = qgpo.deploy()
    env = gym.make(config.deploy.env.env_id)
    env.reset()
    for _ in range(config.deploy.num_deploy_steps):
        env.render()
        env.step(agent.act(env.observation))
    #---------------------------------------
    # Customized deploy code ↑
    #---------------------------------------

if __name__ == '__main__':
    log.info("config: \n{}".format(config))
    qgpo_pipeline(config)
