from generative_rl.configurations.halfcheetah_srpo import config
from generative_rl.algorithms.srpo import SRPO
from generative_rl.utils.log import log
import gym

def srpo_pipeline(config):

    srpo = SRPO(config)

    #---------------------------------------
    # Customized train code ↓
    #---------------------------------------
    srpo.train()
    #---------------------------------------
    # Customized train code ↑
    #---------------------------------------

    #---------------------------------------
    # Customized deploy code ↓
    #---------------------------------------
    agent = srpo.deploy()
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
    srpo_pipeline(config)