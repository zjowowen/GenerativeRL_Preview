import torch
from easydict import EasyDict

action_size = 6
state_size = 17
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

t_embedding_dim = 16  # CHANGE
t_encoder = dict(
    type="SinusoidalPosEmb",
    args=dict(dim=t_embedding_dim),
)

config = EasyDict(
    train=dict(
        project="d4rl-halfcheetah-cps",
        simulator=dict(
            type="GymEnvSimulator",
            args=dict(
                env_id="HalfCheetah-v2",
            ),
        ),
        dataset=dict(
            type="D4RLDataset",
            args=dict(
                env_id="halfcheetah-medium-expert-v2",
                device=device,
            ),
        ),
        model=dict(
            CPSPolicy=dict(
                device=device,
                policy_model=dict(
                    state_dim=state_size,
                    action_dim=action_size,
                    layer=2,
                ),
                LA=1.0,
                LA_min=0,
                LA_max=100,
                target_kl=0.04,
                critic=dict(
                    device=device,
                    adim=action_size,
                    sdim=state_size,
                    DoubleQNetwork=dict(
                        backbone=dict(
                            type="ConcatenateMLP",
                            args=dict(
                                hidden_sizes=[action_size + state_size, 256, 256, 256],
                                output_size=1,
                                activation="mish",
                            ),
                        ),
                    ),
                ),
                diffusion_model=dict(
                    device=device,
                    x_size=action_size,
                    alpha=1.0,
                    beta=0.01,
                    solver=dict(
                        # type = "ODESolver",
                        # args = dict(
                        #     library="torchdyn",
                        # ),
                        type="DPMSolver",
                        args=dict(
                            order=2,
                            device=device,
                            steps=17,
                        ),
                    ),
                    path=dict(
                        type="linear_vp_sde",
                        beta_0=0.1,
                        beta_1=20.0,
                    ),
                    model=dict(
                        type="noise_function",
                        args=dict(
                            t_encoder=t_encoder,
                            backbone=dict(
                                type="CONCATMLP",
                                args=dict(
                                    state_dim=state_size,
                                    action_dim=action_size,
                                    t_dim=t_embedding_dim,
                                ),
                            ),
                        ),
                    ),
                ),
            )
        ),
        parameter=dict(
            behaviour_policy=dict(
                batch_size=256,
                iterations=2000000,
                learning_rate=3e-4,
                lr_learning_rate=3e-5,
                update_momentum=0.005,
                update_target_every=5,
                update_policy_every=2,
                update_lr_every=1000,
                step_start_target=1000,
                grad_norm=7.0,
                t_max=2000,
            ),
            sample_per_state=16,
            critic=dict(
                batch_size=256,
                learning_rate=3e-4,
                discount_factor=0.99,
                update_momentum=0.005,
                grad_norm=7.0,
                max_action=1.0,
                t_max=2000,
            ),
            actor=dict(
                batch_size=256,
                iterations=1000000,
                learning_rate=3e-4,
            ),
            evaluation=dict(
                evaluation_interval=1,
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