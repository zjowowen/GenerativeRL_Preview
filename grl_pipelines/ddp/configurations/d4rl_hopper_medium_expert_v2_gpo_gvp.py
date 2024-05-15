import torch
from easydict import EasyDict


def make_config(device):

    action_size = 3
    state_size = 11

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
    project_name = "d4rl-hopper-medium-expert-v2-GPO-GVP-DDP"

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
            type="gvp",
        ),
        reverse_path=dict(
            type="gvp",
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

    return EasyDict(
        train=dict(
            project=project_name,
            device=device,
            simulator=dict(
                type="GymEnvSimulator",
                args=dict(
                    env_id="hopper-medium-expert-v2",
                ),
            ),
            dataset=dict(
                type="GPOD4RLDataset",
                args=dict(
                    env_id="hopper-medium-expert-v2",
                    device=device,
                ),
            ),
            model=dict(
                GPPolicy=dict(
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
                    batch_size=4096,
                    learning_rate=1e-4,
                    epochs=10000,
                    # new add below
                    lr_decy=False,
                ),
                sample_per_state=16,
                fake_data_t_span=None if solver_type == "DPMSolver" else 32,
                critic=dict(
                    batch_size=4096,
                    epochs=20000,
                    learning_rate=3e-4,
                    discount_factor=0.99,
                    update_momentum=0.005,
                    # new add below
                    lr_decy=False,
                ),
                guided_policy=dict(
                    batch_size=4096,
                    epochs=20000,
                    learning_rate=1e-4,
                    # new add below
                    copy_from_basemodel=False,
                    lr_decy=False,
                ),
                evaluation=dict(
                    eval=True,
                    evaluation_behavior_policy_interval=200,
                    evaluation_guided_policy_interval=200,
                    guidance_scale=[0.0, 1.0, 2.0],
                ),
                checkpoint_path=f"./{project_name}/checkpoint",
                checkpoint_freq=100,
            ),
        ),
        deploy=dict(
            device=device,
            env=dict(
                env_id="hopper-medium-expert-v2",
                seed=0,
            ),
            num_deploy_steps=1000,
            t_span=None if solver_type == "DPMSolver" else 32,
        ),
    )
