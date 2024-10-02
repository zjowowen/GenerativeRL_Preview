import gym

from grl.algorithms.idql import IDQLAlgorithm
from grl.utils.log import log
from grl_pipelines.benchmark.idql.d4rl_halfcheetah_medium_expert_idql import config
def idql_pipeline(config):

    idql = IDQLAlgorithm(config)

    # ---------------------------------------
    # Customized train code ↓
    # ---------------------------------------
    idql.train()
    # ---------------------------------------
    # Customized train code ↑
    # ---------------------------------------

    # ---------------------------------------
    # Customized deploy code ↓
    # ---------------------------------------
    agent = idql.deploy()
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
    idql_pipeline(config)