from typing import Callable, Dict, List, Union

from dm_control import suite
import numpy as np
import torch 

class DmControlEnvSimulator:
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

# import torch
# import numpy as np
# from tensordict import TensorDict

# file_paths = ["/root/data/dataset_batch_1.pt", "/root/data/dataset_batch_2.pt"]
# state_dicts = {}
# actions_list = []
# next_states_dicts = {}
# rewards_list = []

# for file_path in file_paths:
#     data = torch.load(file_path)
#     obs_keys = list(data[0]["s"].keys())
    
#     for key in obs_keys:
#         if key not in state_dicts:
#             state_dicts[key] = []
#             next_states_dicts[key] = []
        
#         state_values = np.array([item["s"][key] for item in data], dtype=np.float32)
#         next_state_values = np.array([item["s_"][key] for item in data], dtype=np.float32)
        
#         state_dicts[key].append(torch.tensor(state_values))
#         next_states_dicts[key].append(torch.tensor(next_state_values))
    
#     actions_values = np.array([item["a"] for item in data], dtype=np.float32)
#     rewards_values = np.array([item["r"] for item in data], dtype=np.float32).reshape(-1, 1)
    
#     actions_list.append(torch.tensor(actions_values))
#     rewards_list.append(torch.tensor(rewards_values))

# # Concatenate the tensors along the first dimension
# actions = torch.cat(actions_list, dim=0)
# rewards = torch.cat(rewards_list, dim=0)
# state_tensors = {key: torch.cat(state_dicts[key], dim=0) for key in obs_keys}
# next_state_tensors = {key: torch.cat(next_states_dicts[key], dim=0) for key in obs_keys}

# # Create TensorDicts
# state_tensordict = TensorDict(state_tensors, batch_size=[state_tensors[obs_keys[0]].shape[0]])
# next_state_tensordict = TensorDict(next_state_tensors, batch_size=[next_state_tensors[obs_keys[0]].shape[0]])

# # Combine everything into a final TensorDict
# final_tensordict = TensorDict({
#     "actions": actions,
#     "rewards": rewards,
#     "states": state_tensordict,
#     "next_states": next_state_tensordict
# }, batch_size=[actions.shape[0]])