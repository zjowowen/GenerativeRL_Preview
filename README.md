# GenerativeRL

GenerativeRL is a Python library for solving reinforcement learning (RL) problems using generative models, such as diffusion models and flow models. This library aims to provide a framework for combining the power of generative models with the decision-making capabilities of reinforcement learning algorithms.

## Features

- Integration of diffusion models and flow models for state representation and policy learning in RL
- Implementation of popular RL algorithms tailored for generative models
- Support for various RL environments and benchmarks
- Easy-to-use API for training and evaluation

## Installation

```bash
pip install grl
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
    env.reset()
    for _ in range(config.deploy.num_deploy_steps):
        env.render()
        env.step(agent.act(env.observation))

if __name__ == '__main__':
    log.info("config: \n{}".format(config))
    qgpo_pipeline(config)
```

For more detailed examples and documentation, please refer to the GenerativeRL documentation.

## Contributing

We welcome contributions to GenerativeRL! If you are interested in contributing, please refer to the [Contributing Guide](CONTRIBUTING.md).

## License

GenerativeRL is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for more details.
