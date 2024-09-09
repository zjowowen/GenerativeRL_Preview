import copy
import queue
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from easydict import EasyDict
from gym import spaces
from rich.progress import Progress, track
from tensordict import TensorDict
from torch.distributions import (Distribution, MultivariateNormal,
                                 TransformedDistribution)
from torch.distributions.transforms import TanhTransform
from torch.utils.data import DataLoader
from torchrl.data import LazyTensorStorage, LazyMemmapStorage

import wandb
from grl.datasets import create_dataset
from grl.generative_models.diffusion_model import DiffusionModel
from grl.neural_network import MultiLayerPerceptron
from grl.neural_network.encoders import get_encoder
from grl.rl_modules.simulators import create_simulator
from grl.rl_modules.value_network import DoubleQNetwork
from grl.utils.config import merge_two_dicts_into_newone
from grl.utils.log import log
from grl.rl_modules.replay_buffer import GeneralListBuffer, TensorDictBuffer

DISCRETE_SPACES = (
    spaces.Discrete,
    spaces.MultiBinary,
    spaces.MultiDiscrete,
)
CONTINUOUS_SPACES = (spaces.Box, )


def is_continuous_space(space):
    return isinstance(space, CONTINUOUS_SPACES)


def is_discrete_space(space):
    return isinstance(space, DISCRETE_SPACES)

def heuristic_target_entropy(action_space):
    if is_continuous_space(action_space):
        heuristic_target_entropy = -np.prod(action_space.shape)
    elif is_discrete_space(action_space):
        raise NotImplementedError(
            "TODO(hartikainen): implement for discrete spaces.")
    else:
        raise NotImplementedError((type(action_space), action_space))

    return heuristic_target_entropy

class NonegativeParameter(nn.Module):

    def __init__(self, data=None, requires_grad=True, delta=1e-8):
        super().__init__()
        if data is None:
            data = torch.zeros(1)
        self.log_data = nn.Parameter(torch.log(data + delta), requires_grad=requires_grad)
    
    
    def forward(self):
        return torch.exp(self.log_data)
    
    @property
    def data(self):
        return torch.exp(self.log_data)


class NonegativeFunction(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.model = MultiLayerPerceptron(**config)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.model(x))

class TanhFunction(nn.Module):

    def __init__(self, config):
        super().__init__()
        # self.transform = TanhTransform(cache_size=1)
        self.model = MultiLayerPerceptron(**config)

    def forward(self, x):
        return torch.tanh(self.model(x))
    
class CovarianceMatrix(nn.Module):

    def __init__(self, config, delta=1e-8):
        super().__init__()
        self.dim = config.dim

        self.sigma_lambda = NonegativeFunction(config.sigma_lambda)
        self.sigma_offdiag = TanhFunction(config.sigma_offdiag)

        # register eye matrix
        self.eye = nn.Parameter(torch.eye(self.dim), requires_grad=False)
        self.delta = delta
        self.mask = nn.Parameter(torch.tril(torch.ones(self.dim, self.dim, dtype=torch.bool)).unsqueeze(0), requires_grad=False)
        
    def low_triangle_matrix(self, x):
        low_t_m = self.eye.detach()

        low_t_m=low_t_m.repeat(x.shape[0],1,1)
        low_t_m[torch.concat((torch.reshape(torch.arange(x.shape[0]).repeat(self.dim*(self.dim-1)//2,1).T,(1,-1)),torch.tril_indices(self.dim, self.dim, offset=-1).repeat(1,x.shape[0]))).tolist()]=torch.reshape(self.sigma_offdiag(x),(-1,1)).squeeze(-1)
        #sigma_offdiag=self.sigma_offdiag(x)
        #sigma_offdiag=sigma_offdiag.reshape(-1, self.dim, self.dim)
        #low_t_m = low_t_m + sigma_offdiag.masked_fill(self.mask, 0)
        lambda_ = self.delta + self.sigma_lambda(x)
        low_t_m=torch.einsum("bj,bjk,bk->bjk", lambda_, low_t_m, lambda_)

        return low_t_m

    def forward(self,x):
        ltm = self.low_triangle_matrix(x)
        return torch.matmul(ltm, ltm.T)


class GaussianTanh(nn.Module, Distribution):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if not hasattr(config, "condition_encoder"):
            self.condition_encoder = torch.nn.Identity()
        else:
            self.condition_encoder = get_encoder(config.condition_encoder.type)(**config.condition_encoder.args)
        self.mu_model = MultiLayerPerceptron(**config.mu_model)
        self.cov = CovarianceMatrix(config.cov)

    def dist(self, condition):
        condition = self.condition_encoder(condition)
        mu=self.mu_model(condition)
        scale_tril = self.cov.low_triangle_matrix(condition)
        return TransformedDistribution(
            base_distribution=MultivariateNormal(loc=mu, scale_tril = scale_tril),
            transforms=[TanhTransform(cache_size=1)])

    def log_prob(self, x, condition):
        return self.dist(condition).log_prob(x)

    def sample(self, condition, sample_shape=torch.Size()):
        return self.dist(condition).sample(sample_shape=sample_shape)

    def rsample(self, condition, sample_shape=torch.Size()): 
        return self.dist(condition).rsample(sample_shape=sample_shape)

    def rsample_and_log_prob(self, condition, sample_shape=torch.Size()):
        dist=self.dist(condition)
        x=dist.rsample(sample_shape=sample_shape)
        log_prob=dist.log_prob(x)
        return x, log_prob
    
    def sample_and_log_prob(self, condition, sample_shape=torch.Size()):
        with torch.no_grad():
            return self.rsample_and_log_prob(condition, sample_shape)

    def entropy(self, condition):
        raise NotImplementedError
        # return self.dist(condition).entropy()

    def forward(self, condition):
        dist=self.dist(condition)
        x=dist.rsample()
        log_prob=dist.log_prob(x)
        return x, log_prob


class Gaussian(nn.Module, Distribution):
    def __init__(self, config: EasyDict):
        super().__init__()
        self.config = config
        if not hasattr(config, "condition_encoder"):
            self.condition_encoder = torch.nn.Identity()
        else:
            self.condition_encoder = get_encoder(config.condition_encoder.type)(**config.condition_encoder.args)
        self.mu_model = MultiLayerPerceptron(**config.mu_model)
        self.cov = CovarianceMatrix(config.cov)

    def dist(self, condition):
        condition = self.condition_encoder(condition)
        mu=self.mu_model(condition)
        # repeat the sigma to match the shape of mu
        scale_tril = self.cov.low_triangle_matrix(condition)
        return MultivariateNormal(loc=mu, scale_tril = scale_tril)

    def log_prob(self, x, condition):
        return self.dist(condition).log_prob(x)

    def sample(self, condition, sample_shape=torch.Size()):
        return self.dist(condition).sample(sample_shape=sample_shape)
            
    def rsample(self, condition, sample_shape=torch.Size()): 
        return self.dist(condition).rsample(sample_shape=sample_shape)

    def entropy(self, condition):
        return self.dist(condition).entropy()

    def rsample_and_log_prob(self, condition, sample_shape=torch.Size()):
        dist=self.dist(condition)
        x=dist.rsample(sample_shape=sample_shape)
        log_prob=dist.log_prob(x)
        return x, log_prob

    def sample_and_log_prob(self, condition, sample_shape=torch.Size()):
        with torch.no_grad():
            return self.rsample_and_log_prob(condition, sample_shape)

    def forward(self, condition):
        dist=self.dist(condition)
        x=dist.rsample()
        log_prob=dist.log_prob(x)
        return x, log_prob

class GaussianPolicy(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = GaussianTanh(config.model)

        
    def forward(self, obs):
        action, logp = self.model(obs)
        return action, logp
    
    def log_prob(self, action, obs):
        return self.model.log_prob(action, obs)
    
    def sample(self, obs, sample_shape=torch.Size()):
        return self.model.sample(obs, sample_shape)
    
    def rsample(self, obs, sample_shape=torch.Size()):
        return self.model.rsample(obs, sample_shape)
    
    def entropy(self, obs):
        return self.model.entropy(obs)
    
    def dist(self, obs):
        return self.model.dist(obs)

class ActionStateCritic(nn.Module):

    def __init__(self, config: EasyDict):
        super().__init__()
        self.config = config

        self.q = DoubleQNetwork(config.DoubleQNetwork)
        self.q_target = copy.deepcopy(self.q).requires_grad_(False)

    def compute_mininum_q(
            self,
            action: Union[torch.Tensor, TensorDict],
            state: Union[torch.Tensor, TensorDict],
        ) -> torch.Tensor:
        return self.q.compute_mininum_q(action, state)

    def forward(
            self,
            action: Union[torch.Tensor, TensorDict],
            state: Union[torch.Tensor, TensorDict],
        ) -> torch.Tensor:
        return self.q(action, state)

    def q_loss(
            self,
            action: Union[torch.Tensor, TensorDict],
            state: Union[torch.Tensor, TensorDict],
            reward: Union[torch.Tensor, TensorDict],
            next_states: Union[torch.Tensor, TensorDict],
            done: Union[torch.Tensor, TensorDict],
            policy: nn.Module,
            entropy_coeffi: NonegativeParameter,
            discount_factor: float = 1.0,
        ) -> torch.Tensor:
        with torch.no_grad():
            next_action, next_logp = policy(next_states)
            next_q = self.q_target(next_action, next_states)
            targets = reward + (1. - done.float()) * discount_factor * (next_q - entropy_coeffi.data * next_logp.unsqueeze(-1))
        q0, q1 = self.q.compute_double_q(action, state)
        q_loss = (torch.nn.functional.mse_loss(q0, targets) + torch.nn.functional.mse_loss(q1, targets)) / 2
        return q_loss

class Policy(nn.Module):

    def __init__(self, config: EasyDict):
        super().__init__()
        self.config = config
        self.device = config.device
        # self.policy = GaussianPolicy(config.policy)
        self.policy = DiffusionModel(config.policy)
        self.critic = ActionStateCritic(config.critic)

    def forward(self, state: Union[torch.Tensor, TensorDict]) -> Union[torch.Tensor, TensorDict]:
        """
        Overview:
            Return the output of New policy, which is the action and log probability of action conditioned on the state.
        Arguments:
            state (:obj:`Union[torch.Tensor, TensorDict]`): The input state.
        Returns:
            action (:obj:`Union[torch.Tensor, TensorDict]`): The output action.
        """
        t_span = torch.linspace(0, 1, 100)
        action = self.policy.sample(
            t_span=t_span,
            condition=state,
            with_grad=True
        )
        logp = self.policy.log_prob(
            x=action,
            condition=state,
            using_Hutchinson_trace_estimator=True,
            with_grad=True
        )
        return action, logp
    
    def sample(
            self,
            state: Union[torch.Tensor, TensorDict],
        ) -> Union[torch.Tensor, TensorDict]:
        """
        Overview:
            Return the output of New policy, which is the action conditioned on the state.
        Arguments:
            state (:obj:`Union[torch.Tensor, TensorDict]`): The input state.
        Returns:
            action (:obj:`Union[torch.Tensor, TensorDict]`): The output action.
        """
        t_span = torch.linspace(0, 1, 100)
        action = self.policy.sample(
            t_span=t_span,
            condition=state,
            with_grad=False
        )
        return action

    def action_state_critic_loss(
            self,
            action: Union[torch.Tensor, TensorDict],
            state: Union[torch.Tensor, TensorDict],
            reward: Union[torch.Tensor, TensorDict],
            next_state: Union[torch.Tensor, TensorDict],
            done: Union[torch.Tensor, TensorDict],
            entropy_coeffi: NonegativeParameter,
            discount_factor: float = 1.0,
        ) -> torch.Tensor:

        return self.critic.q_loss(action, state, reward, next_state, done, policy=self, entropy_coeffi=entropy_coeffi, discount_factor=discount_factor)

    def policy_loss(
            self,
            state: Union[torch.Tensor, TensorDict],
            entropy_coeffi: NonegativeParameter,
    ) -> torch.Tensor:
        """
        Overview:
            Calculate the policy loss.
        Arguments:
            state (:obj:`torch.Tensor`): The input state.
            action (:obj:`torch.Tensor`): The input action.
        """

        action, logp = self.forward(state)
        q_value = self.critic.compute_mininum_q(action, state)
        policy_loss = torch.mean(entropy_coeffi.data * logp.unsqueeze(-1) - q_value)
        return policy_loss, torch.mean(logp)

class SACAlgorithm:

    def __init__(
        self,
        config:EasyDict = None,
        simulator = None,
        model: Union[torch.nn.Module, torch.nn.ModuleDict] = None,
    ):
        """
        Overview:
            Initialize the SAC algorithm.
        Arguments:
            config (:obj:`EasyDict`): The configuration , which must contain the following keys:
                train (:obj:`EasyDict`): The training configuration.
                deploy (:obj:`EasyDict`): The deployment configuration.
            simulator (:obj:`object`): The environment simulator.
            model (:obj:`Union[torch.nn.Module, torch.nn.ModuleDict]`): The model.
        Interface:
            ``__init__``, ``train``, ``deploy``
        """
        self.config = config
        self.simulator = simulator

        #---------------------------------------
        # Customized model initialization code ↓
        #---------------------------------------

        self.model = model if model is not None else torch.nn.ModuleDict()

        #---------------------------------------
        # Customized model initialization code ↑
        #---------------------------------------

    def train(
        self,
        config: EasyDict = None
    ):
        """
        Overview:
            Train the model using the given configuration. \
            A weight-and-bias run will be created automatically when this function is called.
        Arguments:
            config (:obj:`EasyDict`): The training configuration.
        """
        
        config = merge_two_dicts_into_newone(
            self.config.train if hasattr(self.config, "train") else EasyDict(),
            config
        ) if config is not None else self.config.train

        with wandb.init(
            project=config.project if hasattr(config, "project") else __class__.__name__,
            **config.wandb if hasattr(config, "wandb") else {}
        ) as wandb_run:
            config=merge_two_dicts_into_newone(EasyDict(wandb_run.config), config)
            wandb_run.config.update(config)
            self.config.train = config

            self.simulator = create_simulator(config.simulator) if hasattr(config, "simulator") else self.simulator
            self.replay_buffer = TensorDictBuffer(config=config.replay_buffer.args)


            #---------------------------------------
            # Customized model initialization code ↓
            #---------------------------------------

            self.entropy_coeffi = NonegativeParameter(torch.tensor(config.parameter.entropy_coeffi))
            self.entropy_coeffi.to(config.model.Policy.device)
            self.target_entropy = config.parameter.target_entropy if hasattr(config.parameter,'target_entropy') \
                else heuristic_target_entropy(self.simulator.env.action_space) * config.parameter.relative_target_entropy_scale if hasattr(config.parameter,'relative_target_entropy_scale') \
                    else heuristic_target_entropy(self.simulator.env.action_space)
            self.target_entropy = torch.tensor(self.target_entropy).to(config.model.Policy.device)
            

            if hasattr(config.model, "Policy"):
                self.model["Policy"] = Policy(config.model.Policy)
                self.model["Policy"].to(config.model.Policy.device)
                if torch.__version__ >= "2.0.0":
                    pass
                    # self.model["Policy"] = torch.compile(self.model["Policy"])

            #---------------------------------------
            # Customized model initialization code ↑
            #---------------------------------------


            #---------------------------------------
            # Customized training code ↓
            #---------------------------------------

            action_state_critic_optimizer = torch.optim.Adam(
                self.model["Policy"].critic.q.parameters(),
                lr=config.parameter.critic.learning_rate,
            )

            policy_optimizer = torch.optim.Adam(
                self.model["Policy"].policy.model.parameters(),
                lr=config.parameter.policy.learning_rate,
            )

            entropy_coeffi_optimizer = torch.optim.Adam(
                self.entropy_coeffi.parameters(),
                lr=config.parameter.entropy.learning_rate,
            )

            def evaluate(model):
                evaluation_results = dict()
                def policy(obs: np.ndarray) -> np.ndarray:
                    obs = torch.tensor(obs, dtype=torch.float32, device=config.model.Policy.device).unsqueeze(0)
                    action = model.sample(obs).squeeze(0).cpu().detach().numpy()
                    return action
                evaluation_results[f"evaluation/total_return"] = self.simulator.evaluate(policy=policy, )[0]["total_return"]
                return evaluation_results
            
            def collect(model, num_steps, random_policy=False, random_ratio=0.0):
                if random_policy:
                    return self.simulator.collect_steps(policy=None, num_steps=num_steps, random_policy=True)
                else:
                    def policy(obs: np.ndarray) -> np.ndarray:
                        obs = torch.tensor(obs, dtype=torch.float32, device=config.model.Policy.device).unsqueeze(0)
                        action = model.sample(obs).squeeze(0).cpu().detach().numpy()
                        # randomly replace some item of action with random action
                        if np.random.rand() < random_ratio:
                            # select random i from 0 to action.shape[0]
                            i = np.random.randint(0, action.shape[0])
                            # randomly select a value from -1 to 1
                            action[i] = np.random.rand() * 2 - 1
                        return action
                    return self.simulator.collect_steps(policy=policy, num_steps=num_steps)

            def collate_to_tensordict(data_list):
                """
                Collates a list of dictionaries into a TensorDict.
                
                Args:
                    data_list (list of dict): A list where each element is a dictionary containing
                                            data such as 'obs', 'action', 'reward', 'done', 'next_obs'.
                
                Returns:
                    tensordict (TensorDict): A TensorDict containing collated tensors.
                """
                
                # Initialize lists for each key
                obs_list = []
                action_list = []
                reward_list = []
                done_list = []
                next_obs_list = []
                
                # Iterate through the list of dictionaries to collect each data
                for item in data_list:
                    obs_list.append(torch.tensor(item['obs'], dtype=torch.float32))
                    action_list.append(torch.tensor(item['action'], dtype=torch.float32))
                    reward_list.append(torch.tensor(item['reward'], dtype=torch.float32))
                    done_list.append(torch.tensor(item['done'], dtype=torch.bool))
                    next_obs_list.append(torch.tensor(item['next_obs'], dtype=torch.float32))
                
                # Stack the lists into tensors
                obs_tensor = torch.stack(obs_list)
                action_tensor = torch.stack(action_list)
                reward_tensor = torch.stack(reward_list)
                done_tensor = torch.stack(done_list)
                next_obs_tensor = torch.stack(next_obs_list)
                
                # Create the TensorDict
                tensordict = TensorDict({
                    'obs': obs_tensor,
                    'action': action_tensor,
                    'reward': reward_tensor,
                    'done': done_tensor,
                    'next_obs': next_obs_tensor
                }, batch_size=[len(data_list)])
                
                return tensordict


            def compute_entropy_coeffi_loss(obs, policy):
                with torch.no_grad():
                    action, logp = policy.forward(obs)
                    average_action_entropy = -torch.mean(logp)
                entropy_coeffi_loss = self.entropy_coeffi.data * (average_action_entropy - self.target_entropy)

                return entropy_coeffi_loss, average_action_entropy

            for online_rl_iteration in track(range(config.parameter.online_rl.iterations), description="Online RL iteration"):

                if online_rl_iteration == 0 or (online_rl_iteration + 1) % config.parameter.evaluation.evaluation_interval == 0:
                    evaluation_results = evaluate(self.model["Policy"])
                    wandb_run.log(data=evaluation_results, commit=False)
                    log.info(f"online_rl_iteration: {online_rl_iteration}:" + "".join([f"{key}: {value}, " for key, value in evaluation_results.items()]))


                if online_rl_iteration > 0:

                    data = collect(self.model["Policy"], num_steps=config.parameter.online_rl.collect_steps, random_policy=False, random_ratio=config.parameter.online_rl.random_ratio)
                    data = collate_to_tensordict(data)
                    self.replay_buffer.add(data)
                else:
                    data = collect(self.model["Policy"], num_steps=config.parameter.online_rl.collect_steps_at_the_beginning, random_policy=True)
                    data = collate_to_tensordict(data)
                    self.replay_buffer.add(data)

                counter=0
                q_loss_sum=0
                q_value_sum=0
                policy_loss_sum=0
                logp_sum=0
                entropy_coeffi_loss_sum=0
                average_action_entropy_sum=0
                critic_gradient_norm_sum=0
                policy_gradient_norm_sum=0
                entropy_gradient_norm_sum=0

                for _ in range(config.parameter.online_rl.update_steps):
                    batch = self.replay_buffer.sample(batch_size=config.parameter.online_rl.batch_size).to(config.model.Policy.device)
                    counter+=1
                    state = batch["obs"]
                    action = batch["action"].unsqueeze(-1) if len(batch["action"].shape) == 1 else batch["action"]
                    reward = batch["reward"].unsqueeze(-1) if len(batch["reward"].shape) == 1 else batch["reward"]
                    next_state = batch["next_obs"]
                    done = batch["done"].unsqueeze(-1) if len(batch["done"].shape) == 1 else batch["done"]

                    action_state_critic_optimizer.zero_grad()
                    
                    q_loss = self.model["Policy"].action_state_critic_loss(action, state, reward, next_state, done, discount_factor=config.parameter.critic.discount_factor, entropy_coeffi=self.entropy_coeffi)
                    q_loss.backward()
                    critic_gradient_norm = torch.nn.utils.clip_grad_norm_(self.model["Policy"].critic.q.parameters(), config.parameter.critic.grad_clip)
                    action_state_critic_optimizer.step()

                    q_loss_sum+=q_loss
                    critic_gradient_norm_sum+=critic_gradient_norm


                    if online_rl_iteration % 10 == 0:
                        q_value = torch.mean(self.model["Policy"].critic.compute_mininum_q(action, state))
                        q_value_sum+=q_value


                    policy_optimizer.zero_grad()
                    policy_loss, logp = self.model["Policy"].policy_loss(state, entropy_coeffi=self.entropy_coeffi)
                    policy_loss.backward()
                    policy_gradient_norm = torch.nn.utils.clip_grad_norm_(self.model["Policy"].policy.model.parameters(), config.parameter.policy.grad_clip)
                    policy_optimizer.step()

                    policy_loss_sum+=policy_loss
                    logp_sum+=logp
                    policy_gradient_norm_sum+=policy_gradient_norm

                    entropy_coeffi_optimizer.zero_grad()
                    entropy_coeffi_loss, average_action_entropy = compute_entropy_coeffi_loss(state, self.model["Policy"])
                    entropy_coeffi_loss.backward()
                    entropy_gradient_norm = torch.nn.utils.clip_grad_norm_(self.entropy_coeffi.parameters(), config.parameter.entropy.grad_clip)
                    entropy_coeffi_optimizer.step()

                    entropy_coeffi_loss_sum+=entropy_coeffi_loss
                    average_action_entropy_sum+=average_action_entropy
                    entropy_gradient_norm_sum+=entropy_gradient_norm


                    with torch.no_grad():
                        for q_target_param, q_param in zip(self.model["Policy"].critic.q_target.parameters(), self.model["Policy"].critic.q.parameters()):
                            q_target_param.data.mul_(config.parameter.critic.update_momentum)
                            q_target_param.data.add_((1-config.parameter.critic.update_momentum)*q_param.data)

                if counter % 100 == 0:
                    log.info(f"online_rl_iteration: {online_rl_iteration}, q_loss: {q_loss_sum/counter}, q_value: {q_value_sum/counter}, policy_loss: {policy_loss_sum/counter}, logp: {logp_sum/counter}, entropy_coeffi_loss: {entropy_coeffi_loss_sum/counter}, average_action_entropy: {average_action_entropy_sum/counter}, critic_gradient_norm: {critic_gradient_norm_sum/counter}, policy_gradient_norm: {policy_gradient_norm_sum/counter}, entropy_gradient_norm: {entropy_gradient_norm_sum/counter}")

                if online_rl_iteration % 10 == 0:
                    wandb.log(
                        data={
                            "q_value": q_value_sum/counter,
                        },
                        commit=False)

                wandb.log(
                    data={
                        "iteration": online_rl_iteration,
                        "q_loss": q_loss_sum/counter,
                        "policy_loss": policy_loss_sum/counter,
                        "logp": logp_sum/counter,
                        "entropy_coeffi_loss": entropy_coeffi_loss_sum/counter,
                        "average_action_entropy": average_action_entropy_sum/counter,
                        "critic_gradient_norm": critic_gradient_norm_sum/counter,
                        "policy_gradient_norm": policy_gradient_norm_sum/counter,
                        "entropy_gradient_norm": entropy_gradient_norm_sum/counter,

                    },
                    commit=True)

            #---------------------------------------
            # Customized training code ↑
            #---------------------------------------

            wandb.finish()


    def deploy(self, config:EasyDict = None):
        
        pass

