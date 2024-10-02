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
    # ---------------------------------------
    # Customized deploy code ↑
    # ---------------------------------------
    
if __name__ == "__main__":
    log.info("config: \n{}".format(config))
    idql_pipeline(config)