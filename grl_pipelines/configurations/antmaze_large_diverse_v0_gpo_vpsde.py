import torch
from easydict import EasyDict

action_size = 8
state_size = 29
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
t_embedding_dim = 32
t_encoder = dict(
    type="GaussianFourierProjectionTimeEncoder",
    args=dict(
        embed_dim=t_embedding_dim,
        scale=30.0,
    ),
)
algorithm_type = "GPO"
solver_type = "ODESolver"
model_type = "DiffusionModel"
project_name = "antmaze-large-diverse-v0-GPO-VPSDE"
model = dict(
    device=device,
    x_size=action_size,
    solver=(
        dict(
            type="DPMSolver",
            args=dict(
                order=2,
                device=device,
                steps=17,
            ),
        )
        if solver_type == "DPMSolver"
        else (
            dict(
                type="ODESolver",
                args=dict(
                    library="torchdiffeq",
                ),
            )
            if solver_type == "ODESolver"
            else dict(
                type="SDESolver",
                args=dict(
                    library="torchsde",
                ),
            )
        )
    ),
    path=dict(
        type="linear_vp_sde",
        beta_0=0.1,
        beta_1=20.0,
    ),
    reverse_path=dict(
        type="linear_vp_sde",
        beta_0=0.1,
        beta_1=20.0,
    ),
    model=dict(
        type="noise_function",
        args=dict(
            t_encoder=t_encoder,
            backbone=dict(
                type="TemporalSpatialResidualNet",
                args=dict(
                    hidden_sizes=[512, 256, 128],
                    output_dim=action_size,
                    t_dim=t_embedding_dim,
                    condition_dim=state_size,
                    condition_hidden_dim=32,
                    t_condition_hidden_dim=128,
                ),
            ),
        ),
    ),
)

config = EasyDict(
    train=dict(
        project=project_name,
        device=device,
        simulator=dict(
            type="GymEnvSimulator",
            args=dict(
                env_id="antmaze-large-diverse-v0",
            ),
        ),
        dataset=dict(
            type="GPOD4RLDataset",
            args=dict(
                env_id="antmaze-large-diverse-v0",
                device=device,
            ),
        ),
        model=dict(
            GPOPolicy=dict(
                device=device,
                model_type=model_type,
                model_loss_type="score_matching",
                model=model,
                critic=dict(
                    device=device,
                    q_alpha=1.0,
                    DoubleQNetwork=dict(
                        backbone=dict(
                            type="ConcatenateMLP",
                            args=dict(
                                hidden_sizes=[action_size + state_size, 256, 256],
                                output_size=1,
                                activation="relu",
                            ),
                        ),
                    ),
                ),
            ),
            GuidedPolicy=dict(
                model_type=model_type,
                model=model,
            ),
        ),
        parameter=dict(
            algorithm_type=algorithm_type,
            behaviour_policy=dict(
                batch_size=2048,
                learning_rate=1e-4,
                epochs=10000,
            ),
            sample_per_state=16,
            fake_data_t_span=None if solver_type == "DPMSolver" else 32,
            critic=dict(
                batch_size=2048,
                epochs=10000,
                learning_rate=3e-4,
                discount_factor=0.99,
                update_momentum=0.005,
            ),
            guided_policy=dict(
                batch_size=2048,
                epochs=10000,
                learning_rate=1e-4,
            ),
            evaluation=dict(
                repeat=5,
                evaluation_interval=500,
                guidance_scale=[0.0, 1.0, 2.0],
            ),
            checkpoint_path=f"./{project_name}/checkpoint",
            checkpoint_freq=100,
        ),
    ),
    deploy=dict(
        device=device,
        env=dict(
            env_id="antmaze-large-diverse-v0",
            seed=0,
        ),
        num_deploy_steps=1000,
        t_span=None if solver_type == "DPMSolver" else 32,
    ),
)
