import torch
from easydict import EasyDict

action_size = 2
state_size = 8
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
model_type = "SchrodingerBridgeConditionalFlowModel"
assert model_type in [
    "OptimalTransportConditionalFlowModel",
    "DiffusionModel",
    "SchrodingerBridgeConditionalFlowModel",
    "IndependentConditionalFlowModel",
]
project_name = "LunarLanderContinuous-v2-GPG-Sbridge"
method = "fine_tune"

if model_type == "DiffusionModel":
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
                        library="torchdiffeq_adjoint",
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
            type="gvp",
            beta_0=0.1,
            beta_1=20.0,
        ),
        model=dict(
            type="velocity_function",
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
elif model_type == "OptimalTransportConditionalFlowModel":
    model = dict(
        device=device,
        x_size=action_size,
        solver=dict(
            type="ODESolver",
            args=dict(
                library="torchdiffeq_adjoint",
            ),
        ),
        path=dict(
            sigma=0.1,
        ),
        model=dict(
            type="velocity_function",
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
elif model_type == "SchrodingerBridgeConditionalFlowModel":
    model = dict(
        device=device,
        x_size=action_size,
        alpha=1.0,
        solver=dict(
            type="ODESolver",
            args=dict(
                library="torchdyn",
            ),
        ),
        path=dict(
            sigma=1.0,
        ),
        velocity_model=dict(
            type="velocity_function",
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
        score_model=dict(
            type="score_function",
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
    method=method,
    train=dict(
        project=project_name,
        device=device,
        simulator=dict(
            type="GymEnvSimulator",
            args=dict(
                env_id="LunarLanderContinuous-v2",
            ),
        ),
        # dataset = dict(
        #     type = "GPOCustomizedDataset",
        #     args = dict(
        #         env_id = "LunarLanderContinuous-v2",
        #         device = device,
        #         numpy_data_path = "./data.npz",
        #     ),
        # ),
        model=dict(
            GPGPolicy=dict(
                device=device,
                model_type=model_type,
                model_loss_type="flow_matching",
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
            behaviour_policy=dict(
                batch_size=2048,
                learning_rate=1e-4,
                epochs=500,
            ),
            sample_per_state=16,
            fake_data_t_span=None if solver_type == "DPMSolver" else 32,
            critic=dict(
                batch_size=2048,
                epochs=500,
                learning_rate=1e-4,
                discount_factor=0.99,
                update_momentum=0.005,
            ),
            guided_policy=dict(
                copy_frome_basemodel=True,
                batch_size=2048,
                epochs=500,
                learning_rate=1e-4,
            ),
            evaluation=dict(
                evaluation_interval=2,
                guidance_scale=[0.0, 1.0, 2.0],
            ),
            checkpoint_path=f"./{project_name}/checkpoint",
            checkpoint_freq=10,
        ),
    ),
    deploy=dict(
        device=device,
        env=dict(
            env_id="LunarLanderContinuous-v2",
            seed=0,
        ),
        num_deploy_steps=1000,
        t_span=None if solver_type == "DPMSolver" else 32,
    ),
)
