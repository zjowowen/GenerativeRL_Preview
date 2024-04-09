from typing import Optional, Tuple, Union, List, Dict, Any, Callable
from easydict import EasyDict
import gym
import d4rl
from rich.progress import Progress
from rich.progress import track
import numpy as np
import torch
import os
from datetime import datetime
from torch.utils.data import DataLoader

import wandb

from generative_rl.rl_modules.policy import SRPOPolicy
from generative_rl.datasets import create_dataset
from generative_rl.datasets.d4rl import D4RLDataset
from generative_rl.simulators import create_simulator
from generative_rl.utils.config import merge_two_dicts_into_newone
from generative_rl.utils.log import log
from generative_rl.agents.srpo import SRPOAgent
from generative_rl.utils import set_seed

class SRPO:

    def __init__(
        self,
        config:EasyDict = None,
        simulator = None,
        dataset: D4RLDataset = None,
        model: Union[torch.nn.Module, torch.nn.ModuleDict] = None,
    ):
        """
        Overview:
            Initialize the QGPO algorithm.
        Arguments:
            - config (:obj:`EasyDict`): The configuration , which must contain the following keys:
                - train (:obj:`EasyDict`): The training configuration.
                - deploy (:obj:`EasyDict`): The deployment configuration.
            - simulator (:obj:`object`): The environment simulator.
            - dataset (:obj:`QGPODataset`): The dataset.
            - model (:obj:`Union[torch.nn.Module, torch.nn.ModuleDict]`): The model.
        Interface:
            ``__init__``, ``train``, ``deploy``
        """
        self.config = config
        self.simulator = simulator
        self.dataset = dataset
        
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
            - config (:obj:`EasyDict`): The training configuration.
        """
        set_seed(self.config.deploy.env["seed"])
        
        config = merge_two_dicts_into_newone(
            self.config.train if hasattr(self.config, "train") else EasyDict(),
            config
        ) if config is not None else self.config.train

        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
        directory_path = os.path.join(f"./{config.project}",formatted_time,)
        os.makedirs(directory_path, exist_ok=True)
        
        with wandb.init(
            project=config.project if hasattr(config, "project") else __class__.__name__,
            **config.wandb if hasattr(config, "wandb") else {}
        ) as wandb_run:
            config=merge_two_dicts_into_newone(EasyDict(wandb_run.config), config)
            
            wandb_run.config.update(config)
            self.config.train = config

            self.simulator = create_simulator(config.simulator) if hasattr(config, "simulator") else self.simulator
            self.dataset = create_dataset(config.dataset) if hasattr(config, "dataset") else self.dataset

            #---------------------------------------
            # Customized model initialization code ↓
            #---------------------------------------
            if hasattr(config.model, "SRPOPolicy"):
                self.model["SRPOPolicy"] = SRPOPolicy(config.model.SRPOPolicy)
                self.model["SRPOPolicy"].to(config.model.SRPOPolicy.device)
                if torch.__version__ >= "2.0.0":
                    self.model["SRPOPolicy"] = torch.compile(self.model["SRPOPolicy"])
            #---------------------------------------
            # test model ↓
            #---------------------------------------
            assert isinstance(self.model, (torch.nn.Module, torch.nn.ModuleDict)), "self.model must be torch.nn.Module or torch.nn.ModuleDict."
            if isinstance(self.model, torch.nn.ModuleDict):
                assert "SRPOPolicy" in self.model and self.model["SRPOPolicy"], "self.model['SRPOPolicy'] cannot be empty."
            else:  # self.model is torch.nn.Module
                assert self.model, "self.model cannot be empty."
            #---------------------------------------
            # Customized model initialization code ↑
            #---------------------------------------


            #---------------------------------------
            # Customized training code ↓
            #---------------------------------------

            def get_train_data(dataloader):
                while True:
                    yield from dataloader

            # def generate_fake_action(model, states, sample_per_state):
            #     # model.eval()
            #     fake_actions_sampled = []
            #     for states in track(np.array_split(states, states.shape[0] // 4096 + 1), description="Generate fake actions"):
            #         #TODO: mkae it batchsize
            #         fake_actions_per_state = []
            #         for _ in range(sample_per_state):
            #             fake_actions_per_state.append(
            #                 model.sample(
            #                     state = states,
            #                     guidance_scale = 0.0,
            #                 )
            #             )
            #         fake_actions_sampled.append(torch.stack(fake_actions_per_state, dim=1))
            #     fake_actions = torch.cat(fake_actions_sampled, dim=0)
            #     return fake_actions

            def pallaral_simple_eval_policy(policy_fn, env_name, seed, eval_episodes=20):
                eval_envs = []
                for i in range(eval_episodes):
                    env = gym.make(env_name)
                    eval_envs.append(env)
                    env.seed(seed + 1001 + i)
                    env.buffer_state = env.reset()
                    env.buffer_return = 0.0
                ori_eval_envs = [env for env in eval_envs]
                import time
                t = time.time()
                while len(eval_envs) > 0:
                    new_eval_envs = []
                    states = np.stack([env.buffer_state for env in eval_envs])
                    states = torch.Tensor(states).to("cuda")
                    with torch.no_grad():
                        actions = policy_fn(states).detach().cpu().numpy()
                    for i, env in enumerate(eval_envs):
                        state, reward, done, info = env.step(actions[i])
                        env.buffer_return += reward
                        env.buffer_state = state
                        if not done:
                            new_eval_envs.append(env)
                    eval_envs = new_eval_envs
                for i in range(eval_episodes):
                    ori_eval_envs[i].buffer_return = d4rl.get_normalized_score(env_name, ori_eval_envs[i].buffer_return)
                mean = np.mean([ori_eval_envs[i].buffer_return for i in range(eval_episodes)])
                std = np.std([ori_eval_envs[i].buffer_return for i in range(eval_episodes)])
                return mean, std
            
            def evaluate(model, train_iter):
                evaluation_results = dict()
                def policy(obs: np.ndarray) -> np.ndarray:
                    obs = torch.tensor(obs, dtype=torch.float32, device=config.model.QGPOPolicy.device).unsqueeze(0)
                    action = model.deter_policy.select_actions(obs).detach().cpu().numpy()
                    return action
                evaluation_results[f"evaluation/guidance_scale:[{guidance_scale}]/total_return"] = self.simulator.evaluate(policy=policy, )[0]["total_return"]
                log.info(f"Train iter: {train_iter}, guidance_scale: {guidance_scale}, total_return: {evaluation_results[f'evaluation/guidance_scale:[{guidance_scale}]/total_return']}")

                return evaluation_results
                

            data_generator = get_train_data(DataLoader(
                self.dataset,
                batch_size=config.parameter.behaviour_policy.batch_size,
                shuffle=True,
                collate_fn=None,
            ))

            behaviour_model_optimizer = torch.optim.Adam(
                self.model["SRPOPolicy"].diffusion_model.model.parameters(),
                lr=config.parameter.behaviour_policy.learning_rate,
            )

            for train_iter in track(range(config.parameter.behaviour_policy.iterations), description="Behaviour policy training"):
                data=next(data_generator)
                #data["s"].shape  torch.Size([2048, 17])   data["a"].shape torch.Size([2048, 6])  data["r"].shape torch.Size([2048, 1])
                behaviour_model_training_loss = self.model["SRPOPolicy"].behaviour_policy_loss(data['a'], data['s'])
                behaviour_model_optimizer.zero_grad()
                behaviour_model_training_loss.backward()
                behaviour_model_optimizer.step()

                # if train_iter == 0 or (train_iter + 1) % config.parameter.evaluation.evaluation_interval == 0:
                #     evaluation_results = evaluate(self.model["SRPOPolicy"], train_iter=train_iter)
                #     wandb_run.log(data=evaluation_results, commit=False)

                wandb_run.log(
                    data=dict(
                        train_iter=train_iter,
                        behaviour_model_training_loss=behaviour_model_training_loss.item(),
                    ),
                    commit=True)
                
                if train_iter == config.parameter.behaviour_policy.iterations-1:
                    file_path = os.path.join(directory_path, f"checkpoint_diffusion_{train_iter+1}.pt")
                    torch.save(
                        dict(
                            diffusion_model=self.model["SRPOPolicy"].diffusion_model.state_dict(),
                            behaviour_model_optimizer=behaviour_model_optimizer.state_dict(),
                            diffusion_iteration=train_iter+1,
                        ),
                        f=file_path,
            )


            # make optimizer for more action
            q_optimizer = torch.optim.Adam(
                self.model["SRPOPolicy"].critic.q0.parameters(), 
                lr=config.parameter.critic.learning_rate)
            v_optimizer = torch.optim.Adam(
                self.model["SRPOPolicy"].critic.vf.parameters(),
                lr=config.parameter.critic.learning_rate)
            
            data_generator = get_train_data(DataLoader(
                self.dataset,
                batch_size=config.parameter.critic.batch_size,
                shuffle=True,
                collate_fn=None,
            ))

            for train_iter in track(range(config.parameter.critic.iterations), description="Critic training"):
                data = next(data_generator)
                
                v_loss ,next_v = self.model["SRPOPolicy"].v_loss(
                    data,
                )
                v_optimizer.zero_grad(set_to_none=True)
                v_loss.backward()
                v_optimizer.step()

                q_loss = self.model["SRPOPolicy"].q_loss(
                    data,
                    next_v,
                    config.parameter.critic.discount_factor,
                )
                q_optimizer.zero_grad(set_to_none=True)
                q_loss.backward()
                q_optimizer.step()
                
                # Update target
                for param, target_param in zip(self.model["SRPOPolicy"].critic.q0.parameters(), self.model["SRPOPolicy"].critic.q0.parameters()):
                    target_param.data.copy_(config.parameter.critic.tau * param.data + (1 - config.parameter.critic.tau) * target_param.data)
                
                wandb_run.log(
                    data=dict(
                        train_iter=train_iter,
                        q_loss=q_loss.item(),
                        v_loss=v_loss.item(),
                    ),
                    commit=True)
                
                if train_iter == config.parameter.critic.iterations-1:
                    file_path = os.path.join(directory_path, f"checkpoint_policy_{train_iter+1}.pt")
                    torch.save(
                        dict(
                            q_model=self.model["SRPOPolicy"].critic.q0.state_dict(),
                            v_model=self.model["SRPOPolicy"].critic.vf.state_dict(),
                            q_optimizer=q_optimizer.state_dict(),
                            v_optimizer=v_optimizer.state_dict(),
                            policy_iteration=train_iter+1,
                        ),
                        f=file_path,)
                
            SRPO_policy_optimizer = torch.optim.Adam(self.model["SRPOPolicy"].deter_policy.parameters(), lr=3e-4)
            for train_iter in track(range(config.parameter.actor.iterations), description="actor training"):
                data = next(data_generator)
                actor_loss = self.model["SRPOPolicy"].srpo_actor_loss(data)
                actor_loss = actor_loss.sum(-1).mean()
                SRPO_policy_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                SRPO_policy_optimizer.step()
                wandb_run.log(
                    data=dict(
                        train_iter=train_iter,
                        actor_loss=actor_loss.item(),
                    ),
                    commit=True)
                
                if train_iter == 0 or (train_iter + 1) % config.parameter.evaluation.evaluation_interval == 0:
                    mean, std = pallaral_simple_eval_policy(self.model["SRPOPolicy"],config.dataset.args.env_id,00)
                    wandb_run.log(
                        data=dict(
                            train_iter=train_iter,
                            rew=mean,
                            std=std,
                        ),
                        commit=True)
                
                if train_iter == config.parameter.critic.iterations-1:
                    file_path = os.path.join(directory_path, f"checkpoint_actor_{train_iter+1}.pt")
                    torch.save(
                        dict(
                            actor_model=self.model["SRPOPolicy"].deter_policy.state_dict(),
                            actor_optimizer=SRPO_policy_optimizer.state_dict(),
                            policy_iteration=train_iter+1,
                        ),
                        f=file_path,)
                
                
            #---------------------------------------
            # Customized training code ↑
            #---------------------------------------

        wandb.finish()


    def deploy(self, config:EasyDict = None) -> SRPOAgent:
        
        if config is not None:
            config = merge_two_dicts_into_newone(self.config.deploy, config)
        else:
            config = self.config.deploy

        return SRPOAgent(
            config=config,
            model=torch.nn.ModuleDict({
                "QGPOPolicy": self.model,
            })
        )
