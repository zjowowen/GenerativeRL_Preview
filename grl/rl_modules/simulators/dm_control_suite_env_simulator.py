from typing import Callable, Dict, List, Union

from dm_control import suite
import numpy as np
import torch 

class dm_control_suite_EnvSimulator:
    """
    Overview:
        A simple gym environment simulator in GenerativeRL.
        This simulator is used to collect episodes and steps using a given policy in a gym environment.
        It runs in single process and is suitable for small-scale experiments.
    Interfaces:
        ``__init__``, ``collect_episodes``, ``collect_steps``, ``evaluate``
    """

    def __init__(self, domain_name: str,task_name: str) -> None:
        """
        Overview:
            Initialize the GymEnvSimulator according to the given configuration.
        Arguments:
            env_id (:obj:`str`): The id of the gym environment to simulate.
        """
        self.env_domain_name = domain_name
        self.task_name=task_name
        self.collect_env = suite.load(domain_name, task_name)
        # self.observation_space = self.collect_env.observation_space
        self.action_space = self.collect_env.action_spec()

    # def collect_episodes(
    #     self,
    #     policy: Union[Callable, torch.nn.Module],
    #     num_episodes: int = None,
    #     num_steps: int = None,
    # ) -> List[Dict]:
    #     """
    #     Overview:
    #         Collect several episodes using the given policy. The environment will be reset at the beginning of each episode.
    #         No history will be stored in this method. The collected information of steps will be returned as a list of dictionaries.
    #     Arguments:
    #         policy (:obj:`Union[Callable, torch.nn.Module]`): The policy to collect episodes.
    #         num_episodes (:obj:`int`): The number of episodes to collect.
    #         num_steps (:obj:`int`): The number of steps to collect.
    #     """
    #     assert num_episodes is not None or num_steps is not None
    #     if num_episodes is not None:
    #         data_list = []
    #         with torch.no_grad():
    #             for i in range(num_episodes):
    #                 time_step = self.collect_env.reset()
    #                 done = False
    #                 truncated = False
    #                 while not done and not truncated:
    #                     action = policy(obs)
    #                     next_obs, reward, done, truncated, _ = (
    #                         self.collect_env.step(action)
    #                     )
    #                     data_list.append(
    #                         dict(
    #                             obs=obs,
    #                             action=action,
    #                             reward=reward,
    #                             truncated=truncated,
    #                             done=done,
    #                             next_obs=next_obs,
    #                         )
    #                     )
    #                     obs = next_obs
    #         return data_list
    #     elif num_steps is not None:
    #         data_list = []
    #         with torch.no_grad():
    #             if gym.__version__ >= "0.26.0":
    #                 while len(data_list) < num_steps:
    #                     obs, _ = self.collect_env.reset()
    #                     done = False
    #                     truncated = False
    #                     while not done and not truncated:
    #                         action = policy(obs)
    #                         next_obs, reward, done, truncated, _ = (
    #                             self.collect_env.step(action)
    #                         )
    #                         data_list.append(
    #                             dict(
    #                                 obs=obs,
    #                                 action=action,
    #                                 reward=reward,
    #                                 truncated=truncated,
    #                                 done=done,
    #                                 next_obs=next_obs,
    #                             )
    #                         )
    #                         obs = next_obs
    #             else:
    #                 while len(data_list) < num_steps:
    #                     obs = self.collect_env.reset()
    #                     done = False
    #                     while not done:
    #                         action = policy(obs)
    #                         next_obs, reward, done, _ = self.collect_env.step(action)
    #                         data_list.append(
    #                             dict(
    #                                 obs=obs,
    #                                 action=action,
    #                                 reward=reward,
    #                                 done=done,
    #                                 next_obs=next_obs,
    #                             )
    #                         )
    #                         obs = next_obs
    #         return data_list

    # def collect_steps(
    #     self,
    #     policy: Union[Callable, torch.nn.Module],
    #     num_episodes: int = None,
    #     num_steps: int = None,
    #     random_policy: bool = False,
    # ) -> List[Dict]:
    #     """
    #     Overview:
    #         Collect several steps using the given policy. The environment will not be reset until the end of the episode.
    #         Last observation will be stored in this method. The collected information of steps will be returned as a list of dictionaries.
    #     Arguments:
    #         policy (:obj:`Union[Callable, torch.nn.Module]`): The policy to collect steps.
    #         num_episodes (:obj:`int`): The number of episodes to collect.
    #         num_steps (:obj:`int`): The number of steps to collect.
    #         random_policy (:obj:`bool`): Whether to use a random policy.
    #     """
    #     assert num_episodes is not None or num_steps is not None
    #     if num_episodes is not None:
    #         data_list = []
    #         with torch.no_grad():
    #             if gym.__version__ >= "0.26.0":
    #                 for i in range(num_episodes):
    #                     obs, _ = self.collect_env.reset()
    #                     done = False
    #                     truncated = False
    #                     while not done and not truncated:
    #                         if random_policy:
    #                             action = self.collect_env.action_space.sample()
    #                         else:
    #                             action = policy(obs)
    #                         next_obs, reward, done, truncated, _ = (
    #                             self.collect_env.step(action)
    #                         )
    #                         data_list.append(
    #                             dict(
    #                                 obs=obs,
    #                                 action=action,
    #                                 reward=reward,
    #                                 truncated=truncated,
    #                                 done=done,
    #                                 next_obs=next_obs,
    #                             )
    #                         )
    #                         obs = next_obs
    #                 self.last_state_obs, _ = self.collect_env.reset()
    #                 self.last_state_done = False
    #                 self.last_state_truncated = False
    #             else:
    #                 for i in range(num_episodes):
    #                     obs = self.collect_env.reset()
    #                     done = False
    #                     while not done:
    #                         if random_policy:
    #                             action = self.collect_env.action_space.sample()
    #                         else:
    #                             action = policy(obs)
    #                         next_obs, reward, done, _ = self.collect_env.step(action)
    #                         data_list.append(
    #                             dict(
    #                                 obs=obs,
    #                                 action=action,
    #                                 reward=reward,
    #                                 done=done,
    #                                 next_obs=next_obs,
    #                             )
    #                         )
    #                         obs = next_obs
    #                 self.last_state_obs = self.collect_env.reset()
    #                 self.last_state_done = False
    #         return data_list
    #     elif num_steps is not None:
    #         data_list = []
    #         with torch.no_grad():
    #             if gym.__version__ >= "0.26.0":
    #                 while len(data_list) < num_steps:
    #                     if not self.last_state_done or not self.last_state_truncated:
    #                         if random_policy:
    #                             action = self.collect_env.action_space.sample()
    #                         else:
    #                             action = policy(self.last_state_obs)
    #                         next_obs, reward, done, truncated, _ = (
    #                             self.collect_env.step(action)
    #                         )
    #                         data_list.append(
    #                             dict(
    #                                 obs=self.last_state_obs,
    #                                 action=action,
    #                                 reward=reward,
    #                                 truncated=truncated,
    #                                 done=done,
    #                                 next_obs=next_obs,
    #                             )
    #                         )
    #                         self.last_state_obs = next_obs
    #                         self.last_state_done = done
    #                         self.last_state_truncated = truncated
    #                     else:
    #                         self.last_state_obs, _ = self.collect_env.reset()
    #                         self.last_state_done = False
    #                         self.last_state_truncated = False
    #             else:
    #                 while len(data_list) < num_steps:
    #                     if not self.last_state_done:
    #                         if random_policy:
    #                             action = self.collect_env.action_space.sample()
    #                         else:
    #                             action = policy(self.last_state_obs)
    #                         next_obs, reward, done, _ = self.collect_env.step(action)
    #                         data_list.append(
    #                             dict(
    #                                 obs=self.last_state_obs,
    #                                 action=action,
    #                                 reward=reward,
    #                                 done=done,
    #                                 next_obs=next_obs,
    #                             )
    #                         )
    #                         self.last_state_obs = next_obs
    #                         self.last_state_done = done
    #                     else:
    #                         self.last_state_obs = self.collect_env.reset()
    #                         self.last_state_done = False
    #         return data_list

    def evaluate(
        self,
        policy: Union[Callable, torch.nn.Module],
        num_episodes: int = None,
        render_args: Dict = None,
    ) -> List[Dict]:
        """
        Overview:
            Evaluate the given policy using the environment. The environment will be reset at the beginning of each episode.
            No history will be stored in this method. The evaluation resultswill be returned as a list of dictionaries.
        """
        if num_episodes is None:
            num_episodes = 1

        if render_args is not None:
            render = True
        else:
            render = False

        def render_env(env, render_args):
            # TODO: support different render modes
            render_output = env.render(
                **render_args,
            )
            return render_output

        eval_results = []
        env = suite.load(self.env_domain_name, self.task_name)
        for i in range(num_episodes):
            if render:
                render_output = []
            data_list = []
            with torch.no_grad():
                step = 0
                time_step = env.reset()
                obs=time_step.observation
                if render:
                    render_output.append(render_env(env, render_args))
                done = False
                action_spec = env.action_spec()
                while not done:
                    action = policy(obs)
                    time_step = env.step(action)
                    next_obs = time_step.observation
                    reward = time_step.reward
                    done = time_step.last() 
                    discount = time_step.discount
                    step += 1
                    if render:
                        render_output.append(render_env(env, render_args))
                    data_list.append(
                        dict(
                            obs=obs,
                            action=action,
                            reward=reward,
                            done=done,
                            next_obs=next_obs,
                            discount=discount,
                        )
                    )
                    obs = next_obs
                if render:
                    render_output.append(render_env(env, render_args))

            eval_results.append(
                dict(
                    total_return=sum([d["reward"] for d in data_list]),
                    total_steps=len(data_list),
                    data_list=data_list,
                    render_output=render_output if render else None,
                )
            )

        return eval_results
