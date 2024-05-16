import os

# xvfb-run -s "-screen 0 1400x900x24" python train.py
# 设置HTTP和HTTPS代理

import d4rl
import gym
import numpy as np
import torch
import wandb
from easydict import EasyDict
from config import config
from grl.algorithms.gp import GPAlgorithm
from grl.utils.log import log
from typing import Any, Dict, List, Tuple, Union


def QLearningAction(agent, obs):
    if isinstance(obs, np.ndarray):
        obs = torch.from_numpy(obs).float().to(config.train.device)
    elif isinstance(obs, Dict):
        obs = {
            k: torch.from_numpy(v).float().to(config.train.device)
            for k, v in obs.items()
        }
    elif isinstance(obs, torch.Tensor):
        obs = obs.float().to(config.train.device)
    else:
        raise ValueError("observation must be a dict, torch.Tensor, or np.ndarray")
    with torch.no_grad():
        obs = obs.unsqueeze(0)
        obs_batch = torch.repeat_interleave(obs, repeats=4096, dim=0)
        action_batch = agent.model["GuidedPolicy"].sample(
            base_model=agent.model["GPPolicy"].base_model,
            guided_model=agent.model["GPPolicy"].guided_model,
            state=obs_batch,
            t_span=(
                torch.linspace(0.0, 1.0, agent.config.t_span).to(obs.device)
                if agent.config.t_span is not None
                else None
            ),
            guidance_scale=agent.guidance_scale,
        )
        q_value = (
            agent.model["GPPolicy"].critic.q_target(obs_batch, action_batch).flatten()
        )
        idx = torch.multinomial(torch.nn.functional.softmax(q_value), 1)
        return action_batch[idx].squeeze(0).cpu().detach().numpy()


def gp_pipeline(config):

    gp = GPAlgorithm(config)

    # ---------------------------------------
    # Customized train code ↓
    # ---------------------------------------

    # ---------------------------------------
    # Customized train code ↑
    # ---------------------------------------
    render = False
    clip = False
    batch_action = True
    # ---------------------------------------
    # Customized deploy code ↓
    # ---------------------------------------
    # ---------------------------------------
    # Customized deploy code ↑
    # ---------------------------------------

    # ---------------------------------------
    # Customized deploy code ↓
    # ---------------------------------------
    with wandb.init(project="Test_experiments") as wandb_run:

        def run_experiment(
            env,
            agent,
            num_steps,
            video_path,
            iteration,
            clip=True,
            render=True,
            **render_args,
        ):
            def render_env(env, render_args):
                return env.render(
                    **render_args,
                )

            action_shape = env.action_space.shape[0]
            observation = env.reset()
            step = 0
            total_reward = 0
            render_output = []
            done = False
            action_out_of_range_counter = 0
            if render:
                render_output.append(render_env(env, render_args))
            for step in range(num_steps):
                if batch_action:
                    action = QLearningAction(agent=agent, obs=observation)
                else:
                    action = agent.act(observation)
                if clip:
                    action = np.clip(action, -1, 1)
                observation, reward, done, _ = env.step(action)
                if render:
                    render_output.append(render_env(env, render_args))
                total_reward += reward

                action_dict = {f"Action_{i}": action[i] for i in range(action_shape)}
                for i in range(action_shape):
                    if action[i] > 1 or action[i] < -1:
                        action_out_of_range_counter += 1
                        break
                wandb_run.log(data=action_dict, commit=False)

                wandb_run.log(
                    {
                        "Step": step,
                        "Reward": reward,
                        "total_reward": total_reward,
                    },
                    commit=True,
                )
                if done:
                    break
            if render:
                render_output.append(render_env(env, render_args))
                print(np.array(render_output).shape)
                video_arry = np.array(render_output).transpose(0, 3, 1, 2)
                video = wandb.Video(video_arry, caption=f"the {iteration}")
            else:
                video = 0
            wandb_run.log(
                {
                    "Action Out of Range Counter": action_out_of_range_counter,
                    "Experiment Total Reward": total_reward,
                    "Experiment Video": video,
                },
                commit=True,
            )
            return total_reward

        agent = gp.deploy()
        num_experiments = 20
        for experiment in range(num_experiments):
            env = gym.make(config.deploy.env.env_id)
            total_reward = run_experiment(
                env,
                agent,
                config.deploy.num_deploy_steps,
                f"{config.train.project}/experiment_{experiment}.mp4",
                iteration=experiment,
                clip=True,
                render=True,
                mode="rgb_array",
            )
            log.info(f"Experiment {experiment + 1}: Total Reward = {total_reward}")

    # ---------------------------------------
    # Customized deploy code ↑
    # ---------------------------------------


if __name__ == "__main__":
    log.info("config: \n{}".format(config))
    gp_pipeline(config)
