# Generative Reinforcement Learning (GRL)
    
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

English | [简体中文(Simplified Chinese)](https://github.com/zjowowen/GenerativeRL_Preview/blob/main/README.zh.md)

GenerativeRL, short for Generative Reinforcement Learning, is a Python library for solving reinforcement learning (RL) problems using generative models, such as diffusion models and flow models. This library aims to provide a framework for combining the power of generative models with the decision-making capabilities of reinforcement learning algorithms.

## Features

- Integration of diffusion models and flow models for state representation, action representation or policy learning in RL
- Implementation of popular RL algorithms tailored for generative models, such as Q-guided policy optimization (QGPO)
- Support for various RL environments and benchmarks
- Easy-to-use API for training and evaluation

## Installation

```bash
pip install grl
```

Or, if you want to install from source:

```bash
git clone https://github.com/zjowowen/GenerativeRL_Preview.git
cd generative-rl
pip install -e .
```

## Quick Start

Here is an example of how to train a diffusion model for Q-guided policy optimization (QGPO) in the MuJoCo environment:

```python
from grl_pipelines.configurations.halfcheetah_qgpo import config
from grl.algorithms import QGPOAlgorithm
from grl.utils.log import log
import gym

def qgpo_pipeline(config):

    qgpo = QGPOAlgorithm(config)
    qgpo.train()

    agent = qgpo.deploy()
    env = gym.make(config.deploy.env.env_id)
    observation = env.reset()
    for _ in range(config.deploy.num_deploy_steps):
        env.render()
        observation, reward, done, _ = env.step(agent.act(observation))

if __name__ == '__main__':
    log.info("config: \n{}".format(config))
    qgpo_pipeline(config)
```

For more detailed examples and documentation, please refer to the GenerativeRL documentation.

## Contributing

We welcome contributions to GenerativeRL! If you are interested in contributing, please refer to the [Contributing Guide](CONTRIBUTING.md).

## License

GenerativeRL is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for more details.
