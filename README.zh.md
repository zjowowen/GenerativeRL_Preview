# 生成式强化学习
    
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

[英语 (English)](https://github.com/zjowowen/GenerativeRL_Preview/blob/main/README.md) | 简体中文

GenerativeRL 是一个使用生成式模型解决强化学习问题的算法库，支持扩散模型和流模型等不同类型的生成式模型。这个库旨在提供一个框架，将生成式模型的能力与强化学习算法的决策能力相结合。

## 特性

- 集成了多种扩散模型和流模型，用于强化学习中的状态与动作表示和策略学习
- 实现了多种基于生成式模型的强化学习算法
- 支持多种强化学习环境和基准
- 易于使用的训练和评估 API

## 安装

```bash
pip install grl
```

或者，如果你想从源码安装：

```bash
git clone https://github.com/zjowowen/GenerativeRL_Preview.git
cd generative-rl
pip install -e .
```

## 启动

这是一个在 MuJoCo 环境中训练 Q-guided policy optimization (QGPO) 的扩散模型的示例：

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

更多详细示例和文档，请参考 GenerativeRL 文档。

## 开源支持

我们欢迎所有对 GenerativeRL 的贡献和支持！请参考 [开源贡献指南](CONTRIBUTING.md)。

## 开源协议

GenerativeRL 开源协议为 Apache License 2.0。更多信息和文档，请参考 [开源协议](LICENSE)。
