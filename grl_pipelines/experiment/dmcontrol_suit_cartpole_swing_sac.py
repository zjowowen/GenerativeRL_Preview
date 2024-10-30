import gym

from grl.algorithms.sac import SACAlgorithm
from grl.utils.log import log
import torch
from easydict import EasyDict

action_size = 1
state_size = 5
domain_name = "cartpole"
task_name = "swingup"
algorithm_type = "SAC_Step"
env_id = f"{domain_name}-{task_name}"
project_name = f"{env_id}-{algorithm_type}"
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
t_embedding_dim = 32
t_encoder = dict(
    type="GaussianFourierProjectionTimeEncoder",
    args=dict(
        embed_dim=t_embedding_dim,
        scale=30.0,
    ),
)
solver_type = "ODESolver"

config = EasyDict(
    train=dict(
        project=project_name,
        simulator=dict(
            type="DeepMindControlEnvSimulator",
            args=dict(
                domain_name=domain_name,
                task_name=task_name,
                dict_return=False,
            ),
        ),
        replay_buffer=dict(
            type="TensorDictBuffer",
            args=dict(
                size=100000,
                batch_size=2000,
            ),
        ),
        dataset=dict(
            type="Dataset",
            args=dict(
                device=device,
            ),
        ),
        model=dict(
            Policy=dict(
                device=device,
                critic=dict(
                    DoubleQNetwork=dict(
                        # action_encoder=dict(
                        #     type="GaussianFourierProjectionEncoder",
                        #     args=dict(
                        #         embed_dim=32,
                        #         x_shape=[action_size],
                        #         scale=0.1,
                        #     ),
                        # ),
                        # state_encoder=dict(
                        #     type="GaussianFourierProjectionEncoder",
                        #     args=dict(
                        #         embed_dim=32,
                        #         x_shape=[state_size],
                        #         scale=0.1,
                        #     ),
                        # ),
                        backbone=dict(
                            type="ConcatenateMLP",
                            args=dict(
                                hidden_sizes=[state_size + action_size, 256, 256],
                                output_size=1,
                                activation="relu",
                            ),
                        ),
                    ),
                ),
                policy=dict(
                    model=dict(
                        # condition_encoder=dict(
                        #     type="GaussianFourierProjectionEncoder",
                        #     args=dict(
                        #         embed_dim=32,
                        #         x_shape=[state_size],
                        #         scale=0.1,
                        #     ),
                        # ),
                        mu_model=dict(
                            hidden_sizes=[state_size, 128, 128],
                            output_size=action_size,
                            activation="relu",
                            final_activation="tanh",
                            scale=5.0,
                        ),
                        cov=dict(
                            dim=action_size,
                            sigma_lambda=dict(
                                hidden_sizes=[state_size, 128, 128],
                                output_size=action_size,
                                activation="relu",
                            ),
                            sigma_offdiag=dict(
                                hidden_sizes=[state_size, 128, 128],
                                output_size=action_size * (action_size - 1) // 2,
                                activation="relu",
                            ),
                        ),
                    ),
                ),
            )
        ),
        parameter=dict(
            entropy_coeffi=0.2,
            target_entropy=-1,
            online_rl=dict(
                iterations=100000,
                collect_steps=1,
                collect_steps_at_the_beginning=1000,
                update_steps=1,
                drop_ratio=0.00001,
                batch_size=2000,
                random_ratio=0.0,
            ),
            policy=dict(
                learning_rate=3e-4,
                grad_clip=5.0,
            ),
            critic=dict(
                learning_rate=3e-4,
                discount_factor=0.99,
                update_momentum=0.005,
                grad_clip=200.0,
            ),
            entropy=dict(
                learning_rate=3e-4,
                grad_clip=1.0,
            ),
            evaluation=dict(
                evaluation_interval=1000,
            ),
        ),
    ),
    deploy=dict(
        device=device,
        env=dict(
            env_id=env_id,
            seed=0,
        ),
        num_deploy_steps=1000,
    ),
)


def sac_pipeline(config):

    sac = SACAlgorithm(config)

    # ---------------------------------------
    # Customized train code ↓
    # ---------------------------------------
    sac.train()
    # ---------------------------------------
    # Customized train code ↑
    # ---------------------------------------

    # ---------------------------------------
    # Customized deploy code ↓
    # ---------------------------------------
    agent = sac.deploy()
    env = gym.make(config.deploy.env.env_id)
    observation = env.reset()
    for _ in range(config.deploy.num_deploy_steps):
        env.render()
        observation, reward, done, _ = env.step(agent.act(observation))
    # ---------------------------------------
    # Customized deploy code ↑
    # ---------------------------------------


if __name__ == "__main__":
    log.info("config: \n{}".format(config))
    sac_pipeline(config)
