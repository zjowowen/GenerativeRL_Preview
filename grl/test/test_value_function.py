import os
import multiprocessing as mp
import signal
import sys
from easydict import EasyDict
from rich.progress import track
import numpy as np
from sklearn.datasets import make_swiss_roll
import matplotlib
from torch.utils.data import Dataset, DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from grl.utils.config import merge_two_dicts_into_newone
from matplotlib import animation
from easydict import EasyDict
import torch
import torch.nn as nn
from grl.generative_models.diffusion_model.diffusion_model import DiffusionModel
from grl.utils.log import log
from grl.utils import set_seed
from grl.datasets.value_test import ReplayMemoryDataset
import wandb


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

config = EasyDict(
    dict(
        device=device,
        project="value_function",
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
                            convolved=True,
                        ),
                    ),
                ),
            ),
        ),
        parameter=dict(
            training_loss_type="flow_matching",
            train_mode=train_mode,
            batch_size=2,
            eval_freq=1000,
            learning_rate=5e-4,
            weight_decay=1e-4,
            iterations=1,
            clip_grad_norm=1.0,
            checkpoint_freq=10000000000,
            checkpoint_path="./checkpoint",
            video_save_path="./videos",
            dataset=dict(
                dataset_folder="./replay_memories_data",
                num_timesteps=2,
            ),
        ),
    )
)

if __name__ == "__main__":
    seed_value = set_seed()
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

        data_generator = get_train_data(
            DataLoader(
                ReplayMemoryDataset(**config.parameter.dataset),
                batch_size=config.parameter.batch_size,
                shuffle=True,
            )
        )
        optimizer = torch.optim.Adam(
            diffusion_model.parameters(),
            lr=config.parameter.learning_rate,
            weight_decay=config.parameter.weight_decay,
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
                data = next(data_generator)
                next_state = data[4].to(device).permute(0, 1, 4, 2, 3)
                next_state = next_state.unsqueeze(1)
                next_state_padded = next_state.expand(-1, 5, -1, -1, -1)
                fixed_x = next_state
                fixed_mask = torch.tensor([0.0, 1.0]).to(config.device)
                fixed_mask = fixed_mask.repeat(fixed_x[0], 1)
                x_t = (
                    diffusion_model.sample_forward_process_with_fixed_x(
                        fixed_x=fixed_x,
                        fixed_mask=fixed_mask,
                        t_span=t_span,
                        batch_size=500,
                    )
                    .cpu()
                    .detach()
                )
                x_t = [
                    x.squeeze(0)
                    for x in torch.split(x_t, split_size_or_sections=1, dim=0)
                ]
                # render_video(x_t, config.parameter.video_save_path, iteration, fps=100, dpi=100)
                p1 = mp.Process(
                    target=render_video,
                    args=(
                        x_t,
                        config.parameter.video_save_path,
                        iteration,
                        100,
                        100,
                        "fixed_x",
                    ),
                )
                p1.start()
                subprocess_list.append(p1)
                x_t = (
                    diffusion_model.sample_forward_process(
                        t_span=t_span, batch_size=500
                    )
                    .cpu()
                    .detach()
                )
                x_t = [
                    x.squeeze(0)
                    for x in torch.split(x_t, split_size_or_sections=1, dim=0)
                ]
                # render_video(x_t, config.parameter.video_save_path, iteration, fps=100, dpi=100)
                p2 = mp.Process(
                    target=render_video,
                    args=(x_t, config.parameter.video_save_path, iteration, 100, 100),
                )
                p2.start()
                subprocess_list.append(p2)

            batch_data = next(data_generator)
            s = batch_data[0].to(config.device)
            s = s.to(config.device)
            s = s.permute(0, 1, 4, 2, 3)

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

            log.info(
                f"iteration {iteration}, gradient {gradient_sum/counter}, loss {loss_sum/counter}"
            )
            history_iteration.append(iteration)

            if iteration == config.parameter.iterations - 1:
                diffusion_model.eval()
                t_span = torch.linspace(0.0, 1.0, 1000)
                data = next(data_generator)
                next_state = data[4].to(device).permute(0, 3, 1, 2)
                next_state = next_state.unsqueeze(1)
                next_state_padded = next_state.expand(-1, 5, -1, -1, -1)
                fixed_x = next_state_padded
                b, n, c, h, w = fixed_x.shape
                fixed_mask = torch.ones(
                    (b, n, c, h, w), dtype=torch.float32, device=fixed_x.device
                )
                fixed_mask[:, 0, :, :, :] = 0
                x_t = (
                    diffusion_model.sample_forward_process_with_fixed_x(
                        fixed_x=fixed_x,
                        fixed_mask=fixed_mask,
                        t_span=t_span,
                        batch_size=500,
                    )
                    .cpu()
                    .detach()
                )
                x_t = [
                    x.squeeze(0)
                    for x in torch.split(x_t, split_size_or_sections=1, dim=0)
                ]
                # render_video(x_t, config.parameter.video_save_path, iteration, fps=100, dpi=100)
                p1 = mp.Process(
                    target=render_video,
                    args=(
                        x_t,
                        config.parameter.video_save_path,
                        iteration,
                        100,
                        100,
                        "fixed_x",
                    ),
                )
                p1.start()
                subprocess_list.append(p1)
                x_t = (
                    diffusion_model.sample_forward_process(
                        t_span=t_span, batch_size=500
                    )
                    .cpu()
                    .detach()
                )
                x_t = [
                    x.squeeze(0)
                    for x in torch.split(x_t, split_size_or_sections=1, dim=0)
                ]
                # render_video(x_t, config.parameter.video_save_path, iteration, fps=100, dpi=100)
                p2 = mp.Process(
                    target=render_video,
                    args=(x_t, config.parameter.video_save_path, iteration, 100, 100),
                )
                p2.start()
                subprocess_list.append(p2)

            if (iteration + 1) % config.parameter.checkpoint_freq == 0:
                save_checkpoint(diffusion_model, optimizer, iteration)

        for p in subprocess_list:
            p.join()
    wandb_run.finish()
