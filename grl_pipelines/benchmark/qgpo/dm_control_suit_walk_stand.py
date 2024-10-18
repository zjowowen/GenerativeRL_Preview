import torch
from easydict import EasyDict
from grl.neural_network.encoders import register_encoder
import torch.nn as nn
class walker_encoder(nn.Module):
    def __init__(self):
        super(walker_encoder, self).__init__()
        self.orientation_mlp = nn.Sequential(
            nn.Linear(14, 28),
            nn.ReLU(),
            nn.Linear(28, 28),
            nn.LayerNorm(28),
        )

        self.velocity_mlp = nn.Sequential(
            nn.Linear(9, 18),
            nn.ReLU(),
            nn.Linear(18, 18),
            nn.LayerNorm(18)
        )
        
        self.height_mlp = nn.Sequential(
            nn.Linear(1, 2),
            nn.ReLU(),
            nn.Linear(2, 2),
            nn.LayerNorm(2),
        )
    
    def forward(self, x: dict) -> torch.Tensor:
        orientation_output = self.orientation_mlp(x['orientations'])
        velocity_output = self.velocity_mlp(x['velocity'])
        height=x["height"]
        if height.dim() == 1:  
            height = height.unsqueeze(-1)  
        height_output = self.height_mlp(height)  
        combined_output = torch.cat([orientation_output, velocity_output, height_output], dim=-1)
        return combined_output
register_encoder(walker_encoder,"walker_encoder")  
data_path=""
algorithm="QGPO"
action_size = 6
state_size = 48
domain_name="walker"
task_name="stand"
env_id=f"{domain_name}-{task_name}"
project_name =  f"{env_id}-{algorithm}"
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
t_embedding_dim = 32
t_encoder = dict(
    type="GaussianFourierProjectionTimeEncoder",
    args=dict(
        embed_dim=t_embedding_dim,
        scale=30.0,
    ),
)
solver_type = "DPMSolver"
action_augment_num = 16
config = EasyDict(
    train=dict(
        project=project_name,
        simulator=dict(
            type="DeepMindControlEnvSimulator",
            args=dict(
                domain_name=domain_name,
                task_name=task_name,
            ),
        ),
        dataset=dict(
            type="QGPODMcontrolTensorDictDataset",
            args=dict(
                path=data_path,
                action_augment_num=action_augment_num,
                dcit_calulate=True
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
                                hidden_sizes=[action_size + state_size, 256, 256],
                                output_size=1,
                                activation="relu",
                            ),
                        ),
                        state_encoder=dict(
                            type="walker_encoder",
                            args=dict(
                            ),
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
                            condition_encoder=dict(
                                type="walker_encoder",
                                args=dict(
                                            ),
                            ),
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
                    energy_guidance=dict(
                        t_encoder=t_encoder,
                        condition_encoder=dict(
                                type="walker_encoder",
                                args=dict(
                                            ),
                            ),
                        backbone=dict(
                            type="ConcatenateMLP",
                            args=dict(
                                hidden_sizes=[
                                    action_size + state_size + t_embedding_dim,
                                    256,
                                    256,
                                ],
                                output_size=1,
                                activation="silu",
                            ),
                        ),
                    ),
                ),
            )
        ),
        parameter=dict(
            behaviour_policy=dict(
                batch_size=4096,
                learning_rate=1e-4,
                iterations=4000,
            ),
            action_augment_num=action_augment_num,
            fake_data_t_span=None if solver_type == "DPMSolver" else 32,
            energy_guided_policy=dict(
                batch_size=256,
            ),
            critic=dict(
                stop_training_iterations=4000,
                learning_rate=3e-4,
                discount_factor=0.99,
                update_momentum=0.005,
            ),
            energy_guidance=dict(
                iterations=4000,
                learning_rate=3e-4,
            ),
            evaluation=dict(
                evaluation_interval=200,
                guidance_scale=[0.0, 1.0],
            ),
            checkpoint_path=f"./{env_id}-{algorithm}",
        ),
    ),
    deploy=dict(
        device=device,
        env=dict(
            env_id="Walker2d-v2",
            seed=0,
        ),
        num_deploy_steps=1000,
        t_span=None if solver_type == "DPMSolver" else 32,
    ),
)

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


if __name__ == "__main__":
    log.info("config: \n{}".format(config))
    qgpo_pipeline(config)
