import gym
import torch

class GymEnv:
    def __init__(self, config):
        self.config = config
        self.collect_env = gym.make(config.env_id)

        if gym.__version__ >= '0.26.0':
            self.last_state_obs, _ =self.collect_env.reset()
            self.last_state_done=False
            self.last_state_truncated=False
        else:
            self.last_state_obs=self.collect_env.reset()
            self.last_state_done=False

        self.eval_env = gym.make(config.env_id)
        self.observation_space = self.collect_env.observation_space
        self.action_space = self.collect_env.action_space

    def collect_trajectories(self, policy, num_episodes=None, num_steps=None, device='cpu'):
        assert num_episodes is not None or num_steps is not None
        if num_episodes is not None:
            data_list = []
            with torch.no_grad():
                if gym.__version__ >= '0.26.0':
                    for i in range(num_episodes):
                        obs, _ = self.collect_env.reset()
                        done = False
                        truncated = False
                        while not done or not truncated:
                            action, log_prob = policy(torch.tensor(obs).unsqueeze(0).to(torch.float32).to(device))
                            action = action.squeeze(0).cpu().detach().numpy()
                            next_obs, reward, done, truncated, _ = self.collect_env.step(action)
                            data_list.append((obs, action, reward, done, next_obs))
                            obs = next_obs
                else:
                    for i in range(num_episodes):
                        obs = self.collect_env.reset()
                        done = False
                        while not done:
                            action, log_prob = policy(torch.tensor(obs).unsqueeze(0).to(torch.float32).to(device))
                            action = action.squeeze(0).cpu().detach().numpy()
                            next_obs, reward, done, _ = self.collect_env.step(action)
                            data_list.append((obs, action, reward, done, next_obs))
                            obs = next_obs
            return data_list
        elif num_steps is not None:
            data_list = []
            with torch.no_grad():
                if gym.__version__ >= '0.26.0':
                    while len(data_list) < num_steps:
                        obs, _ = self.collect_env.reset()
                        done = False
                        truncated = False
                        while not done or not truncated:
                            action, log_prob = policy(torch.tensor(obs).unsqueeze(0).to(torch.float32).to(device))
                            action = action.squeeze(0).cpu().detach().numpy()
                            next_obs, reward, done, truncated, _ = self.collect_env.step(action)
                            data_list.append((obs, action, reward, done, next_obs))
                            obs = next_obs
                else:
                    while len(data_list) < num_steps:
                        obs = self.collect_env.reset()
                        done = False
                        while not done:
                            action, log_prob = policy(torch.tensor(obs).unsqueeze(0).to(torch.float32).to(device))
                            action = action.squeeze(0).cpu().detach().numpy()
                            next_obs, reward, done, _ = self.collect_env.step(action)
                            data_list.append((obs, action, reward, done, next_obs))
                            obs = next_obs
            return data_list

    def collect(self, policy, num_episodes=None, num_steps=None, device='cpu'):
        assert num_episodes is not None or num_steps is not None
        if num_episodes is not None:
            data_list = []
            with torch.no_grad():
                if gym.__version__ >= '0.26.0':
                    for i in range(num_episodes):
                        obs, _ = self.collect_env.reset()
                        done = False
                        truncated = False
                        while not done or not truncated:
                            action = policy(torch.tensor(obs).unsqueeze(0).to(torch.float32).to(device))
                            action = action.squeeze(0).cpu().detach().numpy()
                            next_obs, reward, done, truncated, _ = self.collect_env.step(action)
                            data_list.append((obs, action, reward, done, next_obs))
                            obs = next_obs
                    self.last_state_obs, _ = self.collect_env.reset()
                    self.last_state_done = False
                    self.last_state_truncated = False
                else:
                    for i in range(num_episodes):
                        obs = self.collect_env.reset()
                        done = False
                        while not done:
                            action = policy(torch.tensor(obs).unsqueeze(0).to(torch.float32).to(device))
                            action = action.squeeze(0).cpu().detach().numpy()
                            next_obs, reward, done, _ = self.collect_env.step(action)
                            data_list.append((obs, action, reward, done, next_obs))
                            obs = next_obs
                    self.last_state_obs = self.collect_env.reset()
                    self.last_state_done = False
            return data_list
        elif num_steps is not None:
            data_list = []
            with torch.no_grad():
                if gym.__version__ >= '0.26.0':
                    while len(data_list) < num_steps:
                        if not self.last_state_done or not self.last_state_truncated:
                            action = policy(torch.tensor(self.last_state_obs).unsqueeze(0).to(torch.float32).to(device))
                            action = action.squeeze(0).cpu().detach().numpy()
                            next_obs, reward, done, truncated, _ = self.collect_env.step(action)
                            data_list.append((self.last_state_obs, action, reward, done, next_obs))
                            self.last_state_obs = next_obs
                            self.last_state_done = done
                            self.last_state_truncated = truncated
                        else:
                            self.last_state_obs, _ = self.collect_env.reset()
                            self.last_state_done = False
                            self.last_state_truncated = False
                else:
                    while len(data_list) < num_steps:
                        if not self.last_state_done:
                            action = policy(torch.tensor(self.last_state_obs).unsqueeze(0).to(torch.float32).to(device))
                            action = action.squeeze(0).cpu().detach().numpy()
                            next_obs, reward, done, _ = self.collect_env.step(action)
                            data_list.append((self.last_state_obs, action, reward, done, next_obs))
                            self.last_state_obs = next_obs
                            self.last_state_done = done
                        else:
                            self.last_state_obs = self.collect_env.reset()
                            self.last_state_done = False
            return data_list

    def evaluate(self, policy, device='cpu'):
        if gym.__version__ >= '0.26.0':
            obs, _ = self.eval_env.reset()
            done = False
            truncated = False
            return_ = 0
            steps = 1
            with torch.no_grad():
                while not done:
                    steps += 1
                    action = policy(torch.tensor(obs).unsqueeze(0).to(torch.float32).to(device))
                    action = action.squeeze(0).cpu().detach().numpy()
                    next_obs, reward, done, truncated, _ = self.eval_env.step(action)
                    return_ += reward
                    obs = next_obs
            return return_, steps
        else:
            obs = self.eval_env.reset()
            done = False
            return_ = 0
            steps = 1
            with torch.no_grad():
                while not done:
                    steps += 1
                    action = policy(torch.tensor(obs).unsqueeze(0).to(torch.float32).to(device))
                    action = action.squeeze(0).cpu().detach().numpy()
                    next_obs, reward, done, _ = self.eval_env.step(action)
                    return_ += reward
                    obs = next_obs
            return return_, steps

    @staticmethod
    def create(config):
        return GymEnv(config)

if __name__ == "__main__":
    config=dict(
        env_id='Hopper-v3',
    )
    from easydict import EasyDict
    config = EasyDict(config)
    env = GymEnv(config)
    print(env.observation_space)
    print(env.action_space)
    print(env.collect(lambda x: env.action_space.sample(), 1))
    print(env.evaluate(lambda x: env.action_space.sample()))
