import torch
import torch.nn as nn
from easydict import EasyDict
from grl.neural_network.encoders import register_encoder


class fish_swim(nn.Module):
    def __init__(self):
        super(fish_swim, self).__init__()
        self.joint_angles = nn.Sequential(
            nn.Linear(7, 14),
            nn.ReLU(),
            nn.Linear(14, 14),
            nn.LayerNorm(14),
        )

        self.upright = nn.Sequential(
            nn.Linear(1, 2), nn.ReLU(), nn.Linear(2, 2), nn.LayerNorm(2)
        )

        self.target = nn.Sequential(
            nn.Linear(3, 6), nn.ReLU(), nn.Linear(6, 6), nn.LayerNorm(6)
        )

        self.velocity = nn.Sequential(
            nn.Linear(13, 26), nn.ReLU(), nn.Linear(26, 26), nn.LayerNorm(26)
        )

    def forward(self, x: dict) -> torch.Tensor:
        if x["upright"].dim() == 1:
            upright = x["upright"].unsqueeze(-1)
        else:
            upright = x["upright"]
        joint_angles = self.joint_angles(x["joint_angles"])
        upright = self.upright(upright)
        target = self.target(x["target"])
        velocity = self.velocity(x["velocity"])
        combined_output = torch.cat([joint_angles, upright, target, velocity], dim=-1)
        return combined_output


register_encoder(fish_swim, "fish_swim")

data_path = ""
domain_name = "fish"
task_name = "swim"
env_id = f"{domain_name}-{task_name}"
action_size = 5
state_size = 24

algorithm_type = "QGPO"
solver_type = "DPMSolver"
action_augment_num = 16
model_type = "DiffusionModel"
generative_model_type = "VPSDE"
path = dict(
    type="linear_vp_sde",
    beta_0=0.1,
    beta_1=20.0,
)
model_loss_type = "score_matching"
project_name = f"{env_id}-{algorithm_type}-{generative_model_type}"

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
t_embedding_dim = 32
t_encoder = dict(
    type="GaussianFourierProjectionTimeEncoder",
    args=dict(
        embed_dim=t_embedding_dim,
        scale=30.0,
    ),
)

config = EasyDict(
    train=dict(
        project=project_name,
        device=device,
        wandb=dict(project=f"IQL-{env_id}-{algorithm_type}-{generative_model_type}"),
        simulator=dict(
            type="DeepMindControlEnvSimulator",
            args=dict(
                domain_name=domain_name,
                task_name=task_name,
            ),
        ),
        dataset=dict(
            type="QGPODeepMindControlTensorDictDataset",
            args=dict(
                path=data_path,
                action_augment_num=action_augment_num,
            ),
        ),
        model=dict(
            QGPOPolicy=dict(
                device=device,
                critic=dict(
                    device=device,
                    q_alpha=1.0,
                    DoubleQNetwork=dict(
                        backbone=dict(
                            type="ConcatenateMLP",
                            args=dict(
                                hidden_sizes=[action_size + state_size * 2, 256, 256],
                                output_size=1,
                                activation="relu",
                            ),
                        ),
                        state_encoder=dict(
                            type="fish_swim",
                            args=dict(),
                        ),
                    ),
                ),
                diffusion_model=dict(
                    device=device,
                    x_size=action_size,
                    alpha=1.0,
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
                                    library="torchdyn",
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
                    path=path,
                    reverse_path=path,
                    model=dict(
                        type="noise_function",
                        args=dict(
                            t_encoder=t_encoder,
                            condition_encoder=dict(
                                type="fish_swim",
                                args=dict(),
                            ),
                            backbone=dict(
                                type="TemporalSpatialResidualNet",
                                args=dict(
                                    hidden_sizes=[512, 256, 128],
                                    output_dim=action_size,
                                    t_dim=t_embedding_dim,
                                    condition_dim=state_size * 2,
                                    condition_hidden_dim=32,
                                    t_condition_hidden_dim=128,
                                ),
                            ),
                        ),
                    ),
                    energy_guidance=dict(
                        t_encoder=t_encoder,
                        condition_encoder=dict(
                            type="fish_swim",
                            args=dict(),
                        ),
                        backbone=dict(
                            type="ConcatenateMLP",
                            args=dict(
                                hidden_sizes=[
                                    action_size + state_size * 2 + t_embedding_dim,
                                    256,
                                    256,
                                ],
                                output_size=1,
                                activation="silu",
                            ),
                        ),
                    ),
                ),
            ),
        ),
        parameter=dict(
            behaviour_policy=dict(
                batch_size=4096,
                learning_rate=1e-4,
                epochs=10000,
            ),
            action_augment_num=action_augment_num,
            fake_data_t_span=None if solver_type == "DPMSolver" else 32,
            energy_guided_policy=dict(
                batch_size=256,
            ),
            critic=dict(
                stop_training_epochs=8000,
                learning_rate=3e-4,
                discount_factor=0.99,
                update_momentum=0.005,
            ),
            energy_guidance=dict(
                epochs=10000,
                learning_rate=3e-4,
            ),
            evaluation=dict(
                evaluation_interval=200,
                guidance_scale=[0.0, 1.0],
            ),
            checkpoint_path=f"./{env_id}-{algorithm_type}",
        ),
    ),
    deploy=dict(
        device=device,
        env=dict(
            env_id=env_id,
            seed=0,
        ),
        t_span=None if solver_type == "DPMSolver" else 32,
    ),
)


if __name__ == "__main__":

    import gym

    from grl.algorithms.qgpo import QGPOAlgorithm
    from grl.utils.log import log

    def qgpo_pipeline(config):

        qgpo = QGPOAlgorithm(config)

        # ---------------------------------------
        # Customized train code ↓
        # ---------------------------------------
        qgpo.train()
        # ---------------------------------------
        # Customized train code ↑
        # ---------------------------------------

        # ---------------------------------------
        # Customized deploy code ↓
        # ---------------------------------------
        agent = qgpo.deploy()
        env = gym.make(config.deploy.env.env_id)
        observation = env.reset()
        for _ in range(config.deploy.num_deploy_steps):
            env.render()
            observation, reward, done, _ = env.step(agent.act(observation))
        # ---------------------------------------
        # Customized deploy code ↑
        # ---------------------------------------

    log.info("config: \n{}".format(config))
    qgpo_pipeline(config)
