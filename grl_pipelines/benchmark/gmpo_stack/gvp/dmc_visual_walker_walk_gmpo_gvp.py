import torch
from easydict import EasyDict

data_path = ""
domain_name = "walker"
task_name = "walk"
policy_type = "medium"
pixel_size = 64
env_id = f"{domain_name}_{task_name}"
action_size = 6
stack_frames = 4
state_size = [stack_frames, 64, 64, 3]
state_condition_hidden_dim = 128
algorithm_type = "GMPO"
solver_type = "ODESolver"
model_type = "DiffusionModel"
generative_model_type = "GVP"
path = dict(type="gvp")
model_loss_type = "flow_matching"
project_name = f"{domain_name}-{task_name}-{policy_type}-{pixel_size}-{algorithm_type}-{generative_model_type}"
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, output_dim, stack_frames=4):
        super(Encoder, self).__init__()

        # Define the CNN layers
        self.conv1 = nn.Conv2d(3 * stack_frames, 32, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)

        # Flatten and fully connected layer
        self.fc = nn.Linear(32 * 27 * 27, output_dim)

        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.float() / 255.0 - 0.5
        x = x.permute(0, 4, 1, 2, 3)
        x = x.reshape(x.size(0), x.size(1) * x.size(2), x.size(3), x.size(4))
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x


from grl.neural_network.encoders import register_encoder

register_encoder(Encoder, "Encoder")

t_embedding_dim = 32
t_encoder = dict(
    type="GaussianFourierProjectionTimeEncoder",
    args=dict(
        embed_dim=t_embedding_dim,
        scale=30.0,
    ),
)
model = dict(
    device=device,
    x_size=action_size,
    solver=dict(
        type="ODESolver",
        args=dict(
            library="torchdiffeq",
        ),
    ),
    path=path,
    reverse_path=path,
    model=dict(
        type="velocity_function",
        args=dict(
            t_encoder=t_encoder,
            condition_encoder=dict(
                type="Encoder",
                args=dict(
                    output_dim=state_condition_hidden_dim,
                    stack_frames=stack_frames,
                ),
            ),
            backbone=dict(
                type="TemporalSpatialResidualNet",
                args=dict(
                    hidden_sizes=[512, 256, 128],
                    output_dim=action_size,
                    t_dim=t_embedding_dim,
                    condition_dim=state_condition_hidden_dim,
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
        wandb=dict(
            project=f"IQL-{domain_name}-{task_name}-{policy_type}-{pixel_size}-{algorithm_type}-{generative_model_type}"
        ),
        simulator=dict(
            type="DeepMindControlVisualEnvSimulator2",
            args=dict(
                domain_name=domain_name,
                task_name=task_name,
                stack_frames=stack_frames,
            ),
        ),
        dataset=dict(
            type="GPDeepMindControlVisualTensorDictDataset",
            args=dict(
                env_id=env_id,
                policy_type=policy_type,
                pixel_size=pixel_size,
                path=data_path,
                stack_frames=stack_frames,
            ),
        ),
        model=dict(
            GPPolicy=dict(
                device=device,
                model_type=model_type,
                model_loss_type=model_loss_type,
                model=model,
                critic=dict(
                    device=device,
                    q_alpha=1.0,
                    DoubleQNetwork=dict(
                        backbone=dict(
                            type="ConcatenateMLP",
                            args=dict(
                                hidden_sizes=[
                                    action_size + state_condition_hidden_dim,
                                    256,
                                    256,
                                ],
                                output_size=1,
                                activation="relu",
                            ),
                        ),
                        state_encoder=dict(
                            type="Encoder",
                            args=dict(
                                output_dim=state_condition_hidden_dim,
                                stack_frames=stack_frames,
                            ),
                        ),
                    ),
                    VNetwork=dict(
                        backbone=dict(
                            type="MultiLayerPerceptron",
                            args=dict(
                                hidden_sizes=[state_condition_hidden_dim, 256, 256],
                                output_size=1,
                                activation="relu",
                            ),
                        ),
                        state_encoder=dict(
                            type="Encoder",
                            args=dict(
                                output_dim=state_condition_hidden_dim,
                                stack_frames=stack_frames,
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
            number_stack_time_step=stack_frames,
            behaviour_policy=dict(
                batch_size=4096,
                learning_rate=1e-4,
                epochs=0,
            ),
            t_span=32,
            critic=dict(
                batch_size=4096,
                epochs=2000,
                learning_rate=3e-4,
                discount_factor=0.99,
                update_momentum=0.005,
                tau=0.7,
                method="iql",
            ),
            guided_policy=dict(
                batch_size=4096,
                epochs=10000,
                learning_rate=1e-4,
                beta=1.0,
                weight_clamp=100,
            ),
            evaluation=dict(
                eval=True,
                repeat=10,
                epoch_interval=100,
            ),
            checkpoint_path=f"./{project_name}/checkpoint",
            checkpoint_freq=1000,
        ),
    ),
    deploy=dict(
        device=device,
        env=dict(
            env_id=env_id,
            seed=0,
        ),
        t_span=32,
    ),
)


if __name__ == "__main__":

    import gym
    import numpy as np

    from grl.algorithms.gmpo_stack import GMPOAlgorithm
    from grl.utils.log import log

    def gp_pipeline(config):

        gp = GMPOAlgorithm(config)

        # ---------------------------------------
        # Customized train code ↓
        # ---------------------------------------
        gp.train()
        # ---------------------------------------
        # Customized train code ↑
        # ---------------------------------------

        # ---------------------------------------
        # Customized deploy code ↓
        # ---------------------------------------

        agent = gp.deploy()
        env = gym.make(config.deploy.env.env_id)
        total_reward_list = []
        for i in range(100):
            observation = env.reset()
            total_reward = 0
            while True:
                # env.render()
                observation, reward, done, _ = env.step(agent.act(observation))
                total_reward += reward
                if done:
                    observation = env.reset()
                    print(f"Episode {i}, total_reward: {total_reward}")
                    total_reward_list.append(total_reward)
                    break

        print(
            f"Average total reward: {np.mean(total_reward_list)}, std: {np.std(total_reward_list)}"
        )

        # ---------------------------------------
        # Customized deploy code ↑
        # ---------------------------------------

    log.info("config: \n{}".format(config))
    gp_pipeline(config)
