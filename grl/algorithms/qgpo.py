from typing import Optional, Tuple, Union, List, Dict, Any, Callable
from easydict import EasyDict

from rich.progress import Progress
from rich.progress import track
import numpy as np
import torch
from torch.utils.data import DataLoader

import wandb

from grl.rl_modules.policy import QGPOPolicy
from grl.datasets import create_dataset
from grl.datasets.qgpo import QGPODataset
from grl.rl_modules.simulators import create_simulator
from grl.utils.config import merge_two_dicts_into_newone
from grl.utils.log import log
from grl.agents.qgpo import QGPOAgent

class QGPOAlgorithm:

    def __init__(
        self,
        config:EasyDict = None,
        simulator = None,
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
            self.dataset = create_dataset(config.dataset) if hasattr(config, "dataset") else self.dataset

            #---------------------------------------
            # Customized model initialization code ↓
            #---------------------------------------

            self.model["QGPOPolicy"] = QGPOPolicy(config.model.QGPOPolicy) if hasattr(config.model, "QGPOPolicy") else self.model.get("QGPOPolicy", None)
            self.model["QGPOPolicy"].to(config.model.QGPOPolicy.device)
            if torch.__version__ >= "2.0.0":
                self.model["QGPOPolicy"] = torch.compile(self.model["QGPOPolicy"])
            
            #---------------------------------------
            # Customized model initialization code ↑
            #---------------------------------------


            #---------------------------------------
            # Customized training code ↓
            #---------------------------------------

            def get_train_data(dataloader):
                while True:
                    yield from dataloader

            def generate_fake_action(model, states, sample_per_state):
                # model.eval()
                fake_actions_sampled = []
                for states in track(np.array_split(states, states.shape[0] // 4096 + 1), description="Generate fake actions"):
                    #TODO: mkae it batchsize
                    fake_actions_per_state = []
                    for _ in range(sample_per_state):
                        fake_actions_per_state.append(
                            model.sample(
                                state = states,
                                guidance_scale = 0.0,
                            )
                        )
                    fake_actions_sampled.append(torch.stack(fake_actions_per_state, dim=1))
                fake_actions = torch.cat(fake_actions_sampled, dim=0)
                return fake_actions

            def evaluate(model, train_iter):
                evaluation_results = dict()
                for guidance_scale in config.parameter.evaluation.guidance_scale:
                    def policy(obs: np.ndarray) -> np.ndarray:
                        obs = torch.tensor(obs, dtype=torch.float32, device=config.model.QGPOPolicy.device).unsqueeze(0)
                        action = model.sample(
                            state = obs,
                            guidance_scale=guidance_scale).squeeze(0).cpu().detach().numpy()
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
                self.model["QGPOPolicy"].diffusion_model.model.parameters(),
                lr=config.parameter.behaviour_policy.learning_rate,
            )

            for train_iter in track(range(config.parameter.behaviour_policy.iterations), description="Behaviour policy training"):
                data=next(data_generator)
                behaviour_model_training_loss = self.model["QGPOPolicy"].behaviour_policy_loss(data['a'], data['s'])
                behaviour_model_optimizer.zero_grad()
                behaviour_model_training_loss.backward()
                behaviour_model_optimizer.step()

                if train_iter == 0 or (train_iter + 1) % config.parameter.evaluation.evaluation_interval == 0:
                    evaluation_results = evaluate(self.model["QGPOPolicy"], train_iter=train_iter)
                    wandb_run.log(data=evaluation_results, commit=False)

                wandb_run.log(
                    data=dict(
                        train_iter=train_iter,
                        behaviour_model_training_loss=behaviour_model_training_loss.item(),
                    ),
                    commit=True)

            self.dataset.fake_actions = generate_fake_action(
                self.model["QGPOPolicy"],
                self.dataset.states[:],
                config.parameter.sample_per_state)
            self.dataset.fake_next_actions = generate_fake_action(
                self.model["QGPOPolicy"],
                self.dataset.next_states[:],
                config.parameter.sample_per_state)

            #TODO add notation
            data_generator = get_train_data(DataLoader(
                self.dataset,
                batch_size=config.parameter.energy_guided_policy.batch_size,
                shuffle=True,
                collate_fn=None,
            ))

            q_optimizer = torch.optim.Adam(
                self.model["QGPOPolicy"].critic.q.parameters(),
                lr=config.parameter.critic.learning_rate,
            )

            energy_guidance_optimizer = torch.optim.Adam(
                self.model["QGPOPolicy"].diffusion_model.energy_guidance.parameters(),
                lr=config.parameter.energy_guidance.learning_rate,
            )
            
            with Progress() as progress:
                critic_training = progress.add_task("Critic training", total=config.parameter.critic.stop_training_iterations)
                energy_guidance_training = progress.add_task("Energy guidance training", total=config.parameter.energy_guidance.iterations)

                for train_iter in range(config.parameter.energy_guidance.iterations):
                    data=next(data_generator)
                    if train_iter < config.parameter.critic.stop_training_iterations:
                        q_loss = self.model["QGPOPolicy"].q_loss(
                            data['a'], data['s'], data['r'], data['s_'], data['d'], data['fake_a_'],
                            discount_factor=config.parameter.critic.discount_factor
                        )

                        q_optimizer.zero_grad()
                        q_loss.backward()
                        q_optimizer.step()
                        
                        # Update target
                        for param, target_param in zip(self.model["QGPOPolicy"].critic.parameters(), self.model["QGPOPolicy"].critic.q_target.parameters()):
                            target_param.data.copy_(
                                config.parameter.critic.update_momentum * param.data + (1 - config.parameter.critic.update_momentum) * target_param.data
                            )

                        wandb_run.log(data=dict(q_loss=q_loss.item()), commit=False)
                        progress.update(critic_training, advance=1)

                    energy_guidance_loss = self.model["QGPOPolicy"].energy_guidance_loss(data['s'], data['fake_a'])
                    energy_guidance_optimizer.zero_grad()
                    energy_guidance_loss.backward()
                    energy_guidance_optimizer.step()

                    if train_iter == 0 or (train_iter + 1) % config.parameter.evaluation.evaluation_interval == 0:
                        evaluation_results = evaluate(self.model["QGPOPolicy"], train_iter=train_iter)
                        wandb_run.log(data=evaluation_results, commit=False)

                    wandb_run.log(
                        data=dict(
                            train_iter=train_iter,
                            energy_guidance_loss=energy_guidance_loss.item(),
                        ),
                        commit=True)
                    progress.update(energy_guidance_training, advance=1)

            #---------------------------------------
            # Customized training code ↑
            #---------------------------------------

            wandb.finish()


    def deploy(self, config:EasyDict = None) -> QGPOAgent:
        
        if config is not None:
            config = merge_two_dicts_into_newone(self.config.deploy, config)
        else:
            config = self.config.deploy

        return QGPOAgent(
            config=config,
            model=torch.nn.ModuleDict({
                "QGPOPolicy": self.model,
            })
        )
