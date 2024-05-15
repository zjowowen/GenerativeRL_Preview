# xvfb-run -s "-screen 0 1400x900x24" python test/train.py
import gym
import numpy as np
from grl.algorithms.gp import GPAlgorithm
from grl.datasets import GPOCustomizedDataset
from grl.utils.log import log
from config import (
    config,
)
import wandb



def run_experiment(env, agent, num_steps, video_path, iteration, **render_args):
    def render_env(env, render_args):
        return env.render(
            **render_args,
        )

    observation = env.reset()
    step = 0
    total_reward = 0
    render_output = []
    done = False
    render_output.append(render_env(env, render_args))
    for step in range(num_steps):
        action = agent.act(observation)
        observation, reward, done, _ = env.step(action)
        render_output.append(render_env(env, render_args))
        total_reward += reward
        # Log step data to wandb
        wandb.log(
            {
                "Step": step,
                "Reward": reward,
                "total_reward": total_reward,
            }
        )
        if done:
            break
    # env.monitor.close()
    render_output.append(render_env(env, render_args))
    video_arry = np.array(render_output).transpose(0, 3, 1, 2)
    video = wandb.Video(video_arry, caption=f"the {iteration}")
    wandb.log(
        {
            "Experiment Total Reward": total_reward,
            "Experiment Video": video,
        }
    )
    return total_reward


def gp_pipeline(config):

    gp = GPAlgorithm(
        config,
        dataset=GPOCustomizedDataset(
            numpy_data_path="./data.npz", device=config.train.device
        ),
    )

    # ---------------------------------------
    # Customized train code ↓
    # ---------------------------------------
    # gp.train()
    # ---------------------------------------
    # Customized train code ↑
    # ---------------------------------------

    # ---------------------------------------
    # Customized deploy code ↓
    # ---------------------------------------
    wandb.init(project="deploy_experiments", dir=f"{config.train.project}")
    agent = gp.deploy()
    env = gym.make(config.deploy.env.env_id)
    num_experiments = 10
    for experiment in range(num_experiments):
        total_reward = run_experiment(
            env,
            agent,
            config.deploy.num_deploy_steps,
            f"{config.train.project}/experiment_{experiment}.mp4",
            iteration=experiment,
            mode="rgb_array",
        )
        log.info(f"Experiment {experiment + 1}: Total Reward = {total_reward}")
    # ---------------------------------------
    # Customized deploy code ↑
    # ---------------------------------------


if __name__ == "__main__":
    log.info("config: \n{}".format(config))
    gp_pipeline(config)
