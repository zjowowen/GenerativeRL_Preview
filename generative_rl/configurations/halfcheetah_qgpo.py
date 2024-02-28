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
                    noise_schedule = "linear",
                    solver = "DPMSolver",
                    marginal_prob_std = "marginal_prob_std",
                    score_model = dict(
                        marginal_prob_std = "marginal_prob_std",
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
            policy_lr = 1e-3,
            q_lr = 1e-3,
            q_t_lr = 1e-3,
            num_train_steps_per_epoch = 1000,
            num_train_epochs = 100,
            num_eval_steps_per_epoch = 1000,
            num_eval_epochs = 10,
        ),
    ),
    deploy = dict(
        num_deploy_steps = 1000,
    ),
)

