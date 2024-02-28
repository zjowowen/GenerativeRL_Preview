from typing import Optional, Tuple, Union, List, Dict, Any, Callable
from easydict import EasyDict

import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader

import wandb

from generative_rl.rl_modules.policy import QGPOPolicy
from generative_rl.datasets.qgpo import QGPODataset
from generative_rl.utils.config import merge_two_dicts_into_newone
from generative_rl.agents.qgpo import QGPOAgent

class QGPO:

    def __init__(
        self,
        config:EasyDict = None,
        env = None,
        dataset: QGPODataset = None,
        model: Union[torch.nn.Module, torch.nn.ModuleDict] = None,
    ):
        """
        Overview:
            Initialize the QGPO algorithm.
        Arguments:
            - config (:obj:`EasyDict`): The configuration , which must contain the following keys:
                - train (:obj:`EasyDict`): The training configuration.
                - deploy (:obj:`EasyDict`): The deployment configuration.
            - env (:obj:`Env`): The environment.
            - dataset (:obj:`QGPODataset`): The dataset.
            - model (:obj:`Union[torch.nn.Module, torch.nn.ModuleDict]`): The model.
        Interface:
            ``__init__``, ``train``, ``deploy``
        """
        self.config = config
        self.env = env
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
        config: EasyDict = None,
        create_env_func: Union[Callable, object] = None, 
        create_dataset_func: Union[Callable, object] = None,
    ):
        """
        Overview:
            Train the model using the given configuration. \
            A weight-and-bias run will be created automatically when this function is called.
        Arguments:
            - config (:obj:`EasyDict`): The training configuration.
            - create_env_func (:obj:`Union[Callable, object]`): The function to create the environment.
            - create_dataset_func (:obj:`Union[Callable, object]`): The function to create the dataset.
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

            self.env = create_env_func(config.env) if create_env_func is not None and hasattr(config, "env") else self.env
            self.dataset = create_dataset_func(config.dataset) if create_dataset_func is not None and hasattr(config, "dataset") else self.dataset

            #---------------------------------------
            # Customized model initialization code ↓
            #---------------------------------------

            self.model["QGPOPolicy"] = QGPOPolicy(config.model.QGPOPolicy) if hasattr(config.model, "QGPOPolicy") else self.model.get("QGPOPolicy", None)
            
            #---------------------------------------
            # Customized model initialization code ↑
            #---------------------------------------


            #---------------------------------------
            # Customized training code ↓
            #---------------------------------------

            def get_train_data(dataloader):
                while True:
                    yield from dataloader

            def generate_fake_action(model, states, sample_per_state, diffusion_steps):
                fake_actions_sampled = []
                for states in tqdm.tqdm(np.array_split(states, states.shape[0] // 4096 + 1)):
                    fake_actions_sampled.append(
                        model.sample(
                            states,
                            sample_per_state=sample_per_state,
                            diffusion_steps=diffusion_steps,
                            guidance_scale=0.0,
                        )
                    )
                fake_actions = np.concatenate(fake_actions_sampled)
                return torch.Tensor(fake_actions.astype(np.float32)).to(config.device)

            dataloader = DataLoader(
                self.dataset,
                batch_size=self.config.parameter.behaviour_policy.batch_size,
                shuffle=True,
                collate_fn=None,
            )

            behavior_model_optimizer = torch.optim.Adam(
                self.model["QGPOPolicy"].behaviour_policy.parameters(),
                lr=config.parameter.behaviour_policy.learning_rate,
            )

            for train_iter in range(config.parameter.behaviour_policy.iterations):
                data=get_train_data(dataloader)
                behavior_model_training_loss = self.model["QGPOPolicy"].behaviour_policy_loss(data['a'], data['s'])
                behavior_model_optimizer.zero_grad()
                behavior_model_training_loss.backward()
                behavior_model_optimizer.step()

                wandb_run.log(
                    data=dict(
                        train_iter=train_iter,
                        behavior_model_training_loss=behavior_model_training_loss.item(),
                    ),
                    commit=True)

            self.dataset.fake_actions = generate_fake_action(
                self.model["QGPOPolicy"],
                self.dataset.states[:].cpu().numpy(),
                config.parameter.sample_per_state,
                config.parameter.diffusion_steps)
            self.dataset.fake_next_actions = generate_fake_action(
                self.model["QGPOPolicy"],
                self.dataset.next_states[:].cpu().numpy(),
                config.parameter.sample_per_state,
                config.parameter.diffusion_steps)

            #TODO add notation
            dataloader = DataLoader(
                self.dataset,
                batch_size=self.config.parameter.energy_guided_policy.batch_size,
                shuffle=True,
                collate_fn=None,
            )

            q_optimizer = torch.optim.Adam(
                self.model["QGPOPolicy"].critic.q0.parameters(),
                lr=config.parameter.critic.q0.learning_rate,
            )

            qt_optimizer = torch.optim.Adam(
                self.model["QGPOPolicy"].critic.qt.parameters(),
                lr=config.parameter.critic.qt.learning_rate,
            )
            
            for train_iter in range(config.parameter.critic.iterations):
                data=get_train_data(dataloader)
                if train_iter < config.parameter.critic.q0.stop_training_iterations:
                    q0_loss = self.model["QGPOPolicy"].q_loss(
                        data['a'], data['s'], data['r'], data['s_'], data['d'], data['fake_a_'],
                        discount=config.parameter.critic.q0.discount
                    )

                    q_optimizer.zero_grad()
                    q0_loss.backward()
                    q_optimizer.step()
                    
                    # Update target
                    for param, target_param in zip(self.model["QGPOPolicy"].critic.q0.parameters(), self.model["QGPOPolicy"].critic.q0_target.parameters()):
                        target_param.data.copy_(
                            config.parameter.critic.qt.update_momentum * param.data + (1 - config.parameter.critic.qt.update_momentum) * target_param.data
                        )

                    wandb_run.log(data=dict(q0_loss=q0_loss.item()), commit=False)

                qt_loss = self.model["QGPOPolicy"].qt_loss(data['s'], data['fake_a'])
                qt_optimizer.zero_grad()
                qt_loss.backward()
                qt_optimizer.step()

                wandb_run.log(
                    data=dict(
                        train_iter=train_iter,
                        qt_loss=qt_loss.item(),
                    ),
                    commit=True)

            #---------------------------------------
            # Customized training code ↑
            #---------------------------------------

            wandb.finish()


    def deploy(self, config:EasyDict = None) -> QGPOAgent:
        
        if config is not None:
            config = merge_two_dicts_into_newone(self.config.deploy, config)

        return QGPOAgent(
            config=config,
            model=torch.nn.ModuleDict({
                "QGPOPolicy": self.model,
            })
        )
