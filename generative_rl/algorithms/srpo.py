from typing import Optional, Tuple, Union, List, Dict, Any, Callable
from easydict import EasyDict

from rich.progress import Progress
from rich.progress import track
import numpy as np
import torch
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

            # def evaluate(model, train_iter):
            #     evaluation_results = dict()
            #     for guidance_scale in config.parameter.evaluation.guidance_scale:
            #         def policy(obs: np.ndarray) -> np.ndarray:
            #             obs = torch.tensor(obs, dtype=torch.float32, device=config.model.QGPOPolicy.device).unsqueeze(0)
            #             action = model.sample(
            #                 state = obs,
            #                 guidance_scale=guidance_scale).squeeze(0).cpu().detach().numpy()
            #             return action
            #         evaluation_results[f"evaluation/guidance_scale:[{guidance_scale}]/total_return"] = self.simulator.evaluate(policy=policy, )[0]["total_return"]
            #         log.info(f"Train iter: {train_iter}, guidance_scale: {guidance_scale}, total_return: {evaluation_results[f'evaluation/guidance_scale:[{guidance_scale}]/total_return']}")

            #     return evaluation_results
                

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


            # self.dataset.fake_actions = generate_fake_action(
            #     self.model["QGPOPolicy"],
            #     self.dataset.states[:],
            #     config.parameter.sample_per_state)
            # self.dataset.fake_next_actions = generate_fake_action(
            #     self.model["QGPOPolicy"],
            #     self.dataset.next_states[:],
            #     config.parameter.sample_per_state)

            #TODO add notation


            # data_generator = get_train_data(DataLoader(
            #     self.dataset,
            #     batch_size=config.parameter.energy_guided_policy.batch_size,
            #     shuffle=True,
            #     collate_fn=None,
            # ))

            q_optimizer = torch.optim.Adam(
                self.model["SRPOPolicy"].critic.q.parameters(),
                lr=config.parameter.critic.learning_rate, #关注这个值
            )

            data_generator = get_train_data(DataLoader(
                self.dataset,
                batch_size=config.parameter.critic_policy.batch_size,
                shuffle=True,
                collate_fn=None,
            ))

            for train_iter in track(range(config.critic.stop_training_iterations), description="Critic training"):
                data = next(data_generator)
                v_loss = self.model["SRPOPolicy"].v_loss(
                    data,
                )
                self.v_optimizer.zero_grad(set_to_none=True)
                v_loss.backward()
                self.v_optimizer.step()

                q_loss = self.model["SRPOPolicy"].q_loss(
                    data,
                )
                self.q_optimizer.zero_grad(set_to_none=True)
                q_loss.backward()
                self.q_optimizer.step()
                
                # Update target
                for param, target_param in zip(self.model["SGPOPolicy"].critic.q0.parameters(), self.model["QGPOPolicy"].critic.q0_target.parameters()):
                            target_param.data.copy_(
                                config.parameter.critic.update_momentum * param.data + (1 - config.parameter.critic.update_momentum) * target_param.data
                            )

                wandb_run.log(data=dict(q_loss=q_loss.item()), commit=False)


        self.q[0].update_q0(data)
        
        # evaluate iql policy part, can be deleted
        with torch.no_grad():
            target_q = self.q[0].q0_target(a, s).detach()
            v = self.q[0].vf(s).detach()
        adv = target_q - v
        temp = 10.0 if "maze" in self.args.env else 3.0
        exp_adv = torch.exp(temp * adv.detach()).clamp(max=100.0)

        policy_out = self.deter_policy(s)
        bc_losses = torch.sum((policy_out - a)**2, dim=1)
        policy_loss = torch.mean(exp_adv.squeeze() * bc_losses)
        self.deter_policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()

        self.deter_policy_optimizer.step()
        self.deter_policy_lr_scheduler.step()
        self.policy_loss = policy_loss


            # with Progress() as progress:
            #     critic_training = progress.add_task("Critic training", total=config.parameter.critic.stop_training_iterations)

            #     for train_iter in  range(config.parameter.critic.stop_training_iterations):
            #         data = next(data_generator)
            #         q_loss = self.model["SRPOPolicy"].q_loss(
            #             data['a'], data['s'], data['r'], data['s_'], data['d'], data['fake_a_'],
            #             discount_factor=config.parameter.critic.discount_factor
            #         )
            #         q_optimizer.zero_grad()
            #         q_loss.backward()
            #         q_optimizer.step()
        
                        
            #             # Update target
            #             for param, target_param in zip(self.model["QGPOPolicy"].critic.parameters(), self.model["QGPOPolicy"].critic.q_target.parameters()):
            #                 target_param.data.copy_(
            #                     config.parameter.critic.update_momentum * param.data + (1 - config.parameter.critic.update_momentum) * target_param.data
            #                 )

            #             wandb_run.log(data=dict(q_loss=q_loss.item()), commit=False)
            #             progress.update(critic_training, advance=1)

            #         energy_guidance_loss = self.model["QGPOPolicy"].energy_guidance_loss(data['s'], data['fake_a'])
            #         energy_guidance_optimizer.zero_grad()
            #         energy_guidance_loss.backward()
            #         energy_guidance_optimizer.step()

            #         if train_iter == 0 or (train_iter + 1) % config.parameter.evaluation.evaluation_interval == 0:
            #             evaluation_results = evaluate(self.model["QGPOPolicy"], train_iter=train_iter)
            #             wandb_run.log(data=evaluation_results, commit=False)

            #         wandb_run.log(
            #             data=dict(
            #                 train_iter=train_iter,
            #                 energy_guidance_loss=energy_guidance_loss.item(),
            #             ),
            #             commit=True)
            #         progress.update(energy_guidance_training, advance=1)

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
