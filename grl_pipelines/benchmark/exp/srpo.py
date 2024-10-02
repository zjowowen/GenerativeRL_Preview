import gym

from grl.algorithms.srpo import SRPOAlgorithm
from grl.utils.log import log
from grl_pipelines.benchmark.srpo.d4rl_halfcheetah_medium_expert_srpo import config
def srpo_pipeline(config):

    srpo = SRPOAlgorithm(config)

    # ---------------------------------------
    # Customized train code ↓
    # ---------------------------------------
    srpo.train()
    # ---------------------------------------
    # Customized train code ↑
    # ---------------------------------------

    # ---------------------------------------
    # Customized deploy code ↓
    # ---------------------------------------
    agent = srpo.deploy()
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
    srpo_pipeline(config)