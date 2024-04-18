import multiprocessing as mp
import os
import signal
import sys

import matplotlib
import numpy as np
from easydict import EasyDict
from rich.progress import track
from sklearn.datasets import make_swiss_roll
from torch.utils.data import DataLoader, Dataset

matplotlib.use("Agg")
import gym
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
from easydict import EasyDict
from matplotlib import animation
from torchvision import transforms
from torchvision.datasets import ImageFolder

import wandb
from grl.datasets.value_test import ReplayMemoryDataset, SampleData
from grl.generative_models.diffusion_model.diffusion_model import \
    DiffusionModel
from grl.utils import set_seed
from grl.utils.config import merge_two_dicts_into_newone
from grl.utils.log import log


def get_train_data(dataloader):
    while True:
        yield from dataloader


train_mode = "single_card"
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
image_size = 32
channel_size = 3
video_length = 2
x_size = (video_length, channel_size, image_size, image_size)
patch_block_size = (video_length, image_size, image_size)
patch_size = (1, 4, 4)
patch_grid_num = (
    video_length // patch_size[0],
    image_size // patch_size[1],
    image_size // patch_size[2],
)
num_heads = 6
hidden_size = np.sum(patch_grid_num) * 2 * num_heads * 2
assert (
    hidden_size % (np.sum(patch_grid_num) * 2 * num_heads) == 0
), f"hidden_size must be divisible by patch_grid_num * 2 * num_heads."
projectname = "dit-3D-inpating-video"

config = EasyDict(
    dict(
        device=device,
        project=projectname,
        diffusion_model=dict(
            device=device,
            x_size=x_size,
            alpha=1.0,
            solver=dict(
                type="ODESolver",
                args=dict(
                    library="torchdyn",
                ),
            ),
            path=dict(
                type="linear_vp_sde",
                beta_0=0.1,
                beta_1=20.0,
            ),
            model=dict(
                type="velocity_function",
                args=dict(
                    backbone=dict(
                        type="DiT_3D",
                        args=dict(
                            patch_block_size=patch_block_size,
                            patch_size=patch_size,
                            in_channels=channel_size,
                            hidden_size=hidden_size,
                            depth=6,
                            num_heads=num_heads,
                            learn_sigma=False,
                        ),
                    ),
                ),
            ),
        ),
        parameter=dict(
            training_loss_type="flow_matching",
            train_mode=train_mode,
            batch_size=2700,
            eval_freq=1000,
            learning_rate=5e-4,
            iterations=200000,
            clip_grad_norm=1.0,
            checkpoint_freq=1000,
            checkpoint_path="./checkpoint",
            video_save_path="./videos",
            dataset_folder="./value_function_memories_data",
            dataset_num_episodes=1,
            dataset_max_num_steps_per_episode=13000,
            num_timesteps=video_length,
        ),
    )
)

if __name__ == "__main__":
    seed_value = set_seed()
    if not os.path.exists("value_function_memories_data/actions.memmap.npy"):
        env = gym.make(
            "ALE/Othello-v5",
            obs_type="rgb",  # ram | rgb | grayscale
            frameskip=4,  # frame skip
            mode=None,  # game mode, see Machado et al. 2018
            difficulty=None,  # game difficulty, see Machado et al. 2018
            repeat_action_probability=0.25,  # Sticky action probability
            full_action_space=False,  # Use all actions
            render_mode="rgb_array",  # None | human | rgb_array
        )

        value_test = SampleData(env, config.parameter)
        value_test.start_smple()
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
            ),
        ]
    )
    data_generator0 = get_train_data(
        DataLoader(
            ReplayMemoryDataset(config.parameter),
            batch_size=config.parameter.batch_size,
            shuffle=True,
        )
    )
    data_generator1 = get_train_data(
        DataLoader(
            ReplayMemoryDataset(config.parameter),
            batch_size=12,
            shuffle=True,
        )
    )
    with wandb.init(
        project=f"{config.project if hasattr(config, 'project') else 'fourier-dit-3D-mc-video'}",
        **config.wandb if hasattr(config, "wandb") else {},
    ) as wandb_run:
        config = merge_two_dicts_into_newone(EasyDict(wandb_run.config), config)
        wandb_run.config.update(config)
        log.info(f"start exp with seed value {seed_value}.")
        diffusion_model = DiffusionModel(config=config.diffusion_model).to(
            config.diffusion_model.device
        )
        diffusion_model = torch.compile(diffusion_model)
        optimizer = torch.optim.Adam(
            diffusion_model.parameters(),
            lr=config.parameter.learning_rate,
        )
        if config.parameter.checkpoint_path is not None:
            if (
                not os.path.exists(config.parameter.checkpoint_path)
                or len(os.listdir(config.parameter.checkpoint_path)) == 0
            ):
                log.warning(
                    f"Checkpoint path {config.parameter.checkpoint_path} does not exist"
                )
                last_iteration = -1
            else:
                checkpoint_files = [
                    f
                    for f in os.listdir(config.parameter.checkpoint_path)
                    if f.endswith(".pt")
                ]
                checkpoint_files = sorted(
                    checkpoint_files, key=lambda x: int(x.split("_")[-1].split(".")[0])
                )
                checkpoint = torch.load(
                    os.path.join(
                        config.parameter.checkpoint_path, checkpoint_files[-1]
                    ),
                    map_location="cpu",
                )
                diffusion_model.load_state_dict(checkpoint["model"])
                optimizer.load_state_dict(checkpoint["optimizer"])
                last_iteration = checkpoint["iteration"]
        else:
            last_iteration = -1

        gradient_sum = 0.0
        loss_sum = 0.0
        counter = 0

        def render_video(
            data_list, video_save_path, iteration, fps=100, dpi=100, info=None
        ):
            if not os.path.exists(video_save_path):
                os.makedirs(video_save_path)
            fig = plt.figure(figsize=(6, 6))
            plt.xlim([-10, 10])
            plt.ylim([-10, 10])
            ims = []
            colors = np.linspace(0, 1, len(data_list))

            for i, data in enumerate(data_list):
                # image alpha frm 0 to 1
                im = plt.scatter(data[:, 0], data[:, 1], s=1)
                ims.append([im])
            ani = animation.ArtistAnimation(fig, ims, interval=0.1, blit=True)
            if info is not None:
                ani.save(
                    os.path.join(video_save_path, f"iteration_{iteration}_{info}.mp4"),
                    fps=fps,
                    dpi=dpi,
                )
            else:
                ani.save(
                    os.path.join(video_save_path, f"iteration_{iteration}.mp4"),
                    fps=fps,
                    dpi=dpi,
                )
            # clean up
            plt.close(fig)
            plt.clf()

        def save_checkpoint(model, optimizer, iteration):
            if not os.path.exists(config.parameter.checkpoint_path):
                os.makedirs(config.parameter.checkpoint_path)
            torch.save(
                dict(
                    model=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    iteration=iteration,
                ),
                f=os.path.join(
                    config.parameter.checkpoint_path, f"checkpoint_{iteration}.pt"
                ),
            )

        history_iteration = [-1]

        def save_checkpoint_on_exit(model, optimizer, iterations):
            def exit_handler(signal, frame):
                log.info("Saving checkpoint when exit...")
                save_checkpoint(model, optimizer, iteration=iterations[-1])
                log.info("Done.")
                sys.exit(0)

            signal.signal(signal.SIGINT, exit_handler)

        save_checkpoint_on_exit(diffusion_model, optimizer, history_iteration)

        subprocess_list = []

        for iteration in track(
            range(config.parameter.iterations), description="Training"
        ):

            if iteration <= last_iteration:
                continue

            if iteration > 0 and iteration % config.parameter.eval_freq == 0:
                diffusion_model.eval()
                t_span = torch.linspace(0.0, 1.0, 1000)
                data = next(data_generator1)[4].to(device)
                fixed_x = data.unsqueeze(1).expand(-1, 2, -1, -1, -1)
                fixed_mask = torch.ones(
                    fixed_x.shape, dtype=torch.float32, device=fixed_x.device
                )
                fixed_mask[:, 0, :, :, :] = 0
                x_t = (
                    diffusion_model.sample_forward_process_with_fixed_x(
                        fixed_x=fixed_x,
                        fixed_mask=fixed_mask,
                        t_span=t_span,
                        batch_size=12,
                    )
                    .cpu()
                    .detach()
                )
                video_x_1 = (
                    x_t[-1][0]
                    .squeeze(0)
                    .mul(0.5)
                    .add(0.5)
                    .mul(255)
                    .clamp(0, 255)
                    .permute(0, 2, 3, 1)
                    .to("cpu", torch.uint8)
                )
                if not os.path.exists(config.parameter.video_save_path):
                    os.makedirs(config.parameter.video_save_path)
                torchvision.io.write_video(
                    filename=os.path.join(
                        config.parameter.video_save_path, f"iteration_{iteration}.mp4"
                    ),
                    video_array=video_x_1,
                    fps=20,
                )
                video_x_1 = video_x_1.permute(0, 3, 1, 2).numpy()
                video = wandb.Video(video_x_1, caption=f"iteration {iteration}")
                wandb_run.log(
                    data=dict(
                        video=video,
                    ),
                    commit=False,
                )
            batch_data = next(data_generator0)
            s = batch_data[0].to(config.device)
            s = s.to(config.device)

            # plot2d(batch_data.cpu().numpy())
            diffusion_model.train()

            if config.parameter.training_loss_type == "flow_matching":
                loss = diffusion_model.flow_matching_loss(s)
            elif config.parameter.training_loss_type == "score_matching":
                loss = diffusion_model.score_matching_loss(s)
            else:
                raise NotImplementedError("Unknown loss type")

            optimizer.zero_grad()
            loss.backward()
            gradien_norm = torch.nn.utils.clip_grad_norm_(
                diffusion_model.parameters(), config.parameter.clip_grad_norm
            )
            optimizer.step()
            gradient_sum += gradien_norm.item()
            loss_sum += loss.item()
            counter += 1
            wandb_run.log(
                data=dict(
                    iteration=iteration,
                    loss=loss.item(),
                    average_loss=loss_sum / counter,
                    average_gradient=gradient_sum / counter,
                ),
                commit=True,
            )

            if iteration == config.parameter.iterations - 1:
                diffusion_model.eval()
                t_span = torch.linspace(0.0, 1.0, 1000)
                data = next(data_generator1)[4].to(device)
                fixed_x = data.unsqueeze(1).expand(-1, 2, -1, -1, -1)
                fixed_mask = torch.ones(
                    fixed_x.shape, dtype=torch.float32, device=fixed_x.device
                )
                fixed_mask[:, 0, :, :, :] = 0
                x_t = diffusion_model.sample_forward_process_with_fixed_x(
                    fixed_x=fixed_x,
                    fixed_mask=fixed_mask,
                    t_span=t_span,
                    batch_size=12,
                )
                video_x_1 = (
                    x_t[-1][0]
                    .squeeze(0)
                    .mul(0.5)
                    .add(0.5)
                    .mul(255)
                    .clamp(0, 255)
                    .permute(0, 2, 3, 1)
                    .to("cpu", torch.uint8)
                )
                if not os.path.exists(config.parameter.video_save_path):
                    os.makedirs(config.parameter.video_save_path)
                torchvision.io.write_video(
                    filename=os.path.join(
                        config.parameter.video_save_path, f"iteration_{iteration}.mp4"
                    ),
                    video_array=video_x_1,
                    fps=20,
                )
                video_x_1 = video_x_1.permute(0, 3, 1, 2).numpy()
                video = wandb.Video(video_x_1, caption=f"iteration {iteration}")
                wandb_run.log(
                    data=dict(
                        video=video,
                    ),
                    commit=True,
                )

            if (iteration + 1) % config.parameter.checkpoint_freq == 0:
                save_checkpoint(diffusion_model, optimizer, iteration)

    wandb_run.finish()
