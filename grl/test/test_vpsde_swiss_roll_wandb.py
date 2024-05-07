import os
import signal
import sys

import matplotlib
import numpy as np
from easydict import EasyDict
from rich.progress import track
from sklearn.datasets import make_swiss_roll

import wandb

matplotlib.use("Agg")
from datetime import datetime

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from matplotlib import animation

from grl.generative_models.diffusion_model.diffusion_model import DiffusionModel
from grl.utils import set_seed
from grl.utils.config import merge_two_dicts_into_newone
from grl.utils.log import log

sweep_config = EasyDict(
    name="base-sweep-gvp-lossadding",
    metric=dict(
        name="loss",
        goal="minimize",
    ),
    method="grid",
    parameters=dict(
        diffusion_model=dict(
            parameters=dict(
                path=dict(
                    parameters=dict(
                        type=dict(
                            values=["gvp"],
                        ),
                    ),
                ),
                model=dict(
                    parameters=dict(
                        type=dict(
                            values=[
                                "velocity_function",
                            ],
                        ),
                    ),
                ),
            ),
        ),
        parameter=dict(
            parameters=dict(
                training_loss_type=dict(
                    values=["flow_matching"],
                ),
                lr=dict(
                    values=[2e-3, 3e-3, 4e-3, 5e-3],
                ),
                accumulation_steps=dict(
                    values=[1, 2, 3, 4],
                ),
            ),
        ),
    ),
)


def train_func():
    x_size = 2
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    t_embedding_dim = 32
    t_encoder = dict(
        type="GaussianFourierProjectionTimeEncoder",
        args=dict(
            embed_dim=t_embedding_dim,
            scale=30.0,
        ),
    )

    origin_config = EasyDict(
        dict(
            device=device,
            project_name="gvp_lossadding_lr",
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
                        t_encoder=t_encoder,
                        backbone=dict(
                            type="TemporalSpatialResidualNet",
                            args=dict(
                                hidden_sizes=[512, 256, 128],
                                output_dim=x_size,
                                t_dim=t_embedding_dim,
                            ),
                        ),
                    ),
                ),
            ),
            parameter=dict(
                training_loss_type="flow_matching",
                lr=5e-3,
                data_num=10000,
                iterations=1000,
                batch_size=2048,
                clip_grad_norm=1.0,
                eval_freq=500,
                accumulation_steps=3,
                # checkpoint_freq=5000,
                # checkpoint_path="./checkpoint",
                video_save_path="./video",
                device=device,
            ),
        )
    )

    with wandb.init(
        project="swiss_roll_diffusion_model",
        config=origin_config,
    ) as wandb_run:
        # config=merge_two_dicts_into_newone(EasyDict(wandb_run.config), origin_config)
        # wandb_run.config.update(config)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config = EasyDict(wandb.config)
        run_name = f"{config.diffusion_model.path.type}-{config.diffusion_model.model.type}-{config.parameter.training_loss_type}--{config.parameter.accumulation_steps}--{config.parameter.lr}"
        wandb.run.name = run_name
        wandb.run.save()
        diffusion_model = DiffusionModel(config=config.diffusion_model).to(
            config.diffusion_model.device
        )
        # diffusion_model = torch.compile(diffusion_model)

        # get data
        data = make_swiss_roll(n_samples=config.parameter.data_num, noise=0.01)[
            0
        ].astype(np.float32)[:, [0, 2]]
        # transform data
        data[:, 0] = data[:, 0] / np.max(np.abs(data[:, 0]))
        data[:, 1] = data[:, 1] / np.max(np.abs(data[:, 1]))
        data = (data - data.min()) / (data.max() - data.min())
        data = data * 10 - 5

        optimizer = torch.optim.Adam(
            diffusion_model.parameters(),
            lr=config.parameter.lr,
        )

        last_iteration = -1

        data_loader = torch.utils.data.DataLoader(
            data, batch_size=config.parameter.batch_size, shuffle=True
        )

        def get_train_data(dataloader):
            while True:
                yield from dataloader

        data_generator = get_train_data(data_loader)

        gradient_sum = 0.0
        loss_sum = 0.0
        counter = 0
        iteration = 0

        def plot2d(data):

            plt.scatter(data[:, 0], data[:, 1])
            plt.show()

        def render_video(data_list, video_save_path, iteration, fps=100, dpi=100):
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
            ani.save(
                os.path.join(video_save_path, f"iteration_{iteration}.mp4"),
                fps=fps,
                dpi=dpi,
            )

            wandb_run.log(
                data=dict(
                    video=wandb.Video(
                        os.path.join(video_save_path, f"iteration_{iteration}.mp4"),
                        fps=fps,
                        format="mp4",
                    ),
                ),
                commit=False,
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

        for iteration in track(
            range(config.parameter.iterations), description="Training"
        ):

            if iteration <= last_iteration:
                continue

            batch_data = next(data_generator)
            batch_data = batch_data.to(config.device)
            # plot2d(batch_data.cpu().numpy())
            diffusion_model.train()
            with torch.cuda.amp.autocast():
                if config.parameter.training_loss_type == "flow_matching":
                    loss = diffusion_model.flow_matching_loss(batch_data)
                elif config.parameter.training_loss_type == "score_matching":
                    loss = diffusion_model.score_matching_loss(batch_data)
                else:
                    raise NotImplementedError("Unknown loss type")
            loss.backward()

            gradien_norm = torch.nn.utils.clip_grad_norm_(
                diffusion_model.parameters(), config.parameter.clip_grad_norm
            )
            if (iteration + 1) % config.parameter.accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            gradient_sum += gradien_norm.item()
            loss_sum += loss.item()
            counter += 1
            history_iteration.append(iteration)

            if (iteration == config.parameter.iterations - 1) or (
                iteration > 0 and iteration % config.parameter.eval_freq == 0
            ):
                diffusion_model.eval()
                t_span = torch.linspace(0.0, 1.0, 1000)
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
                render_video(
                    x_t, config.parameter.video_save_path, iteration, fps=100, dpi=100
                )

            wandb_run.log(
                data=dict(
                    iteration=iteration,
                    gradient=gradient_sum / counter,
                    loss=loss.item(),
                ),
                commit=True,
            )


if __name__ == "__main__":
    seed_value = set_seed()
    sweep_id = wandb.sweep(
        sweep=sweep_config, project=f"test_vpsede_swiss-{seed_value}"
    )
    wandb.agent(sweep_id, function=train_func)
