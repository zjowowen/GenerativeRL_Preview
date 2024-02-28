from typing import Optional, Tuple, Union, List, Dict, Any, Callable
from easydict import EasyDict
import torch
from torch.utils.data import DataLoader
import wandb

from generative_rl.rl_modules.policy.base import BasePolicy
from generative_rl.datasets.base import BaseDataset
from generative_rl.utils.config import merge_two_dicts_into_newone
from generative_rl.agents.base import BaseAgent

class BaseAlgorithm:

    def __init__(
        self,
        config:EasyDict = None,
        env = None,
        dataset: BaseDataset = None,
        model: Union[torch.nn.Module, torch.nn.ModuleDict] = None,
    ):
        """
        Overview:
            Initialize the base algorithm.
        Arguments:
            - config (:obj:`EasyDict`): The configuration , which must contain the following keys:
                - train (:obj:`EasyDict`): The training configuration.
                - deploy (:obj:`EasyDict`): The deployment configuration.
            - env (:obj:`Env`): The environment.
            - dataset (:obj:`BaseDataset`): The dataset.
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
        create_env_func: Callable = None, 
        create_dataset_func: Callable = None,
    ):
        """
        Overview:
            Train the model using the given configuration. \
            A weight-and-bias run will be created automatically when this function is called.
        Arguments:
            - config (:obj:`EasyDict`): The training configuration.
            - create_env_func (:obj:`Callable`): The function for creating the environment.
            - create_dataset_func (:obj:`Callable`): The function for creating the dataset.
        """

        if config is not None:
            config = merge_two_dicts_into_newone(
                self.config.train if hasattr(self.config, "train") else EasyDict(),
                config
            )

        with wandb.init(
            project=config.project if hasattr(config, "project") else __class__.__name__,
            **config.wandb if hasattr(config, "wandb") else {}
        ) as wandb_run:
            config.update(EasyDict(wandb_run.config))
            wandb_run.config.update(config)
            self.config.train = config

            self.env = create_env_func(config.env) if create_env_func is not None and hasattr(config, "env") else self.env
            self.dataset = create_dataset_func(config.dataset) if create_dataset_func is not None and hasattr(config, "dataset") else self.dataset

            #---------------------------------------
            # Customized model initialization code ↓
            #---------------------------------------

            self.model["BasePolicy"] = BasePolicy(config.model.BasePolicy) if hasattr(config.model, "BasePolicy") else self.model.get("BasePolicy", None)
            
            #---------------------------------------
            # Customized model initialization code ↑
            #---------------------------------------


            #---------------------------------------
            # Customized training code ↓
            #---------------------------------------

            def get_train_data(dataloader):
                while True:
                    yield from dataloader

            dataloader = DataLoader(
                self.dataset,
                batch_size=self.config.parameter.batch_size,
                shuffle=True,
                collate_fn=None,
            )

            optimizer = torch.optim.Adam(
                self.model["BasePolicy"].parameters(),
                lr=config.parameter.learning_rate,
            )

            for train_iter in range(config.parameter.behaviour_policy.iterations):
                data=get_train_data(dataloader)
                model_training_loss = self.model["BasePolicy"](data)
                optimizer.zero_grad()
                model_training_loss.backward()
                optimizer.step()

                wandb_run.log(
                    data=dict(
                        train_iter=train_iter,
                        model_training_loss=model_training_loss.item(),
                    ),
                    commit=True)

            #---------------------------------------
            # Customized training code ↑
            #---------------------------------------

            wandb.finish()


    def deploy(self, config:EasyDict = None) -> BaseAgent:
        
        if config is not None:
            config = merge_two_dicts_into_newone(self.config.deploy, config)

        return BaseAgent(
            config=config,
            model=torch.nn.ModuleDict({
                "BasePolicy": self.model,
            })
        )
