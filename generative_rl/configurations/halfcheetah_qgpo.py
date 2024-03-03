import torch
from easydict import EasyDict

action_size = 6
state_size = 17
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
t_embedding_dim = 32
t_encoder = dict(
    type = "GaussianFourierProjectionTimeEncoder",
    args = dict(
        embed_dim = t_embedding_dim,
        scale = 30.0,
    ),
)

config = EasyDict(
    train = dict(
        project = 'd4rl-halfcheetah-v2-qgpo',
        env = dict(
            env_id = 'HalfCheetah-v2',
            seed = 0,
        ),
        dataset = dict(
            env_id = "halfcheetah-medium-expert-v2",
            device = device,
        ),
        model = dict(
            QGPOPolicy = dict(
                device = device,
                critic = dict(
                    device = device,
                    q_alpha = 1.0,
                    DoubleQNetwork = dict(
                        backbone = dict(
                            type = "ConcatenateMLP",
                            args = dict(
                                hidden_sizes = [action_size + state_size, 256, 256],
                                output_size = 1,
                                activation = "relu",
                            ),
                        ),
                    ),
                ),
                diffusion_model = dict(
                    device = device,
                    x_size = action_size,
                    alpha = 1.0,
                    solver = dict(
                        type = "ODESolver",
                        args = dict(
                            library="torchdyn",
                        ),
                    ),
                    gaussian_conditional_probability_path = dict(
                        type = "linear_vp_sde",
                        beta_0 = 0.1,
                        beta_1 = 20.0,
                    ),
                    diffusion_process = "VPSDE",
                    score_function = dict(
                        type = "noise_function",
                        model = dict(
                            t_encoder = t_encoder,
                            backbone = dict(
                                type = "TemporalSpatialResidualNet",
                                args = dict(
                                    hidden_sizes = [512, 256, 128],
                                    output_dim = action_size,
                                    t_dim = t_embedding_dim,
                                    condition_dim = state_size,
                                    condition_hidden_dim = 32,
                                    t_condition_hidden_dim = 128,
                                ),
                            ),
                        ),
                    ),
                    energy_guidance = dict(
                        t_encoder = t_encoder,
                        backbone = dict(
                            type = "ConcatenateMLP",
                            args = dict(
                                hidden_sizes = [action_size + state_size + t_embedding_dim, 256, 256],
                                output_size = 1,
                                activation = "silu",
                            ),
                        ),
                    ),
                ),
            )
        ),
        parameter = dict(
            behaviour_policy = dict(
                batch_size = 4096,
                learning_rate = 1e-4,
                iterations = 600000,
            ),
            sample_per_state = 16,
            energy_guided_policy = dict(
                batch_size = 256,
            ),
            critic = dict(
                stop_training_iterations = 500000,
                learning_rate = 3e-4,
                discount_factor = 0.99,
                update_momentum = 0.995,
            ),
            energy_guidance = dict(
                iterations = 600000,
                learning_rate = 3e-4,
            ),
        ),
    ),
    deploy = dict(
        num_deploy_steps = 1000,
    ),
)

