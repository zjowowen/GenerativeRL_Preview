import torch
from easydict import EasyDict

action_size = 6
state_size = 17
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
        project="HalfCheetah-v2-SAC-Online",
        simulator=dict(
            type="GymEnvSimulator",
            args=dict(
                env_id="HalfCheetah-v2",
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
                        action_encoder=dict(
                            type="GaussianFourierProjectionEncoder",
                            args=dict(
                                embed_dim=32,
                                x_shape=[action_size],
                            ),
                        ),
                        state_encoder=dict(
                            type="GaussianFourierProjectionEncoder",
                            args=dict(
                                embed_dim=32,
                                x_shape=[state_size],
                            ),
                        ),
                        backbone=dict(
                            type="ConcatenateMLP",
                            args=dict(
                                hidden_sizes=[320, 512],
                                output_size=1,
                                activation="silu",
                            ),
                        ),
                    ),
                ),
                policy=dict(
                    model=dict(
                        condition_encoder=dict(
                            type="GaussianFourierProjectionEncoder",
                            args=dict(
                                embed_dim=32,
                                x_shape=[state_size],
                            ),
                        ),
                        mu_model=dict(
                            hidden_sizes=[256, 512],
                            output_size=action_size,
                            activation="silu",
                        ),
                        cov=dict(
                            dim=action_size,
                            sigma_lambda=dict(
                                hidden_sizes=[256, 512],
                                output_size=action_size,
                                activation="silu",
                            ),
                            sigma_offdiag=dict(
                                hidden_sizes=[256, 512],
                                output_size=action_size * action_size,
                                activation="silu",
                            ),
                        ),
                    ),
                ),
            )
        ),
        parameter=dict(
            entropy_coeffi=0.2,
            target_entropy=-20,
            online_rl=dict(
                iterations=100000,
                collect_steps=1,
                collect_steps_at_the_beginning=10000,
                drop_ratio=0.00001,
                batch_size=2000,
                random_ratio=0.0,
            ),
            policy=dict(
                learning_rate=1e-4,
                grad_clip=5.0,
            ),
            critic=dict(
                learning_rate=1e-4,
                discount_factor=0.99,
                update_momentum=0.995,
                grad_clip=200.0,
            ),
            entropy=dict(
                learning_rate=1e-4,
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
            env_id="HalfCheetah-v2",
            seed=0,
        ),
        num_deploy_steps=1000,
    ),
)
