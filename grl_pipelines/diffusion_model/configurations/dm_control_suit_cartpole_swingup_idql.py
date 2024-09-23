import torch
from easydict import EasyDict

path="/mnt/nfs3/zhangjinouwen/dataset/dm_control/my_dm_control_suite/cartpole_swingup.npy"
domain_name="cartpole"
task_name="swingup"
env_id=f"{domain_name}-{task_name}"
algorithm="IDQL"
action_size = 1
state_size = 5

project_name =  f"{env_id}-{algorithm}"
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
t_embedding_dim = 64
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
                dict_return=False,
            ),
        ),
        dataset=dict(
            type="QGPODMcontrolTensorDictDataset",
            args=dict(
                path=path,
                action_augment_num=action_augment_num,
            ),
        ),
        model=dict(
            IDQLPolicy=dict(
                device=device,
                critic=dict(
                    device=device,
                    adim=action_size,
                    sdim=state_size,
                    layers=2,
                    update_momentum=0.95,
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
                    VNetwork=dict(
                        backbone=dict(
                            type="MultiLayerPerceptron",
                            args=dict(
                                hidden_sizes=[state_size, 256, 256],
                                output_size=1,
                                activation="relu",
                            ),
                        ),
                    ),
                ),
                diffusion_model=dict(
                    device=device,
                    x_size=action_size,
                    alpha=1.0,
                    beta=0.1,
                    solver=dict(
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
                                type="AllCatMLP",
                                args=dict(
                                    input_dim=state_size + action_size,
                                    output_dim=action_size,
                                    num_blocks=3,
                                ),
                            ),
                        ),
                    ),
                ),
            )
        ),
        parameter=dict(
            behaviour_policy=dict(
                batch_size=2048,
                learning_rate=3e-4,
                iterations=2000,
            ),
            critic=dict(
                batch_size=4096,
                iterations=2000,
                learning_rate=3e-4,
                discount_factor=0.99,
                tau=0.7,
                update_momentum=0.005,
                checkpoint_freq=100,
            ),
            evaluation=dict(
                evaluation_interval=100,
                repeat=5,
            ),
            checkpoint_path=f"./{env_id}-{algorithm}",
        ),
    ),
    deploy=dict(
        device=device,
        env=dict(
            env_id=env_id,
            seed=0,
        ),
        num_deploy_steps=1000,
    ),
)

import gym

from grl.algorithms.idql import IDQLAlgorithm
from grl.utils.log import log

def idql_pipeline(config):

    idql = IDQLAlgorithm(config)

    # ---------------------------------------
    # Customized train code ↓
    # ---------------------------------------
    idql.train()
    # ---------------------------------------
    # Customized train code ↑
    # ---------------------------------------

    # ---------------------------------------
    # Customized deploy code ↓
    # ---------------------------------------
    agent = idql.deploy()
    env = gym.make(config.deploy.env.env_id)
    env.reset()
    for _ in range(config.deploy.num_deploy_steps):
        env.render()
        env.step(agent.act(env.observation))
    # ---------------------------------------
    # Customized deploy code ↑
    # ---------------------------------------


if __name__ == "__main__":
    log.info("config: \n{}".format(config))
    idql_pipeline(config)

