import multiprocessing as mp
import os
import signal
import sys

import matplotlib
import numpy as np
from easydict import EasyDict
from rich.progress import Progress, track
from sklearn.datasets import make_swiss_roll

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from easydict import EasyDict
from matplotlib import animation

from grl.generative_models.diffusion_model import (
    DiffusionModel,
    EnergyConditionalDiffusionModel,
)
from grl.generative_models.diffusion_model.guided_diffusion_model import (
    GuidedDiffusionModel,
)
from grl.rl_modules.value_network.one_shot_value_function import OneShotValueFunction
from grl.utils import set_seed
from grl.utils.log import log

x_size = 2

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
t_embedding_dim = 32
t_encoder = dict(
    type="GaussianFourierProjectionTimeEncoder",
    args=dict(
        embed_dim=t_embedding_dim,
        scale=30.0,
    ),
)
data_num = 10000
config = EasyDict(
    project="diffusion_model_swiss_roll_important_sampling_flow_matching",
    dataset=dict(
        data_num=data_num,
        noise=0.6,
        temperature=0.1,
    ),
    model=dict(
        device=device,
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
                type="noise_function",
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
    ),
    parameter=dict(
        diffusion_model=dict(
            batch_size=2048,
            learning_rate=5e-5,
            clip_grad_norm=1.0,
            iterations=50000,
        ),
        support_size=data_num,
        sample_per_data=100,
        value_function_model=dict(
            batch_size=2048,
            stop_training_iterations=50000,
            learning_rate=5e-4,
            discount_factor=0.99,
            update_momentum=0.995,
        ),
        evaluation=dict(
            eval_freq=5000,
            video_save_path="./video-swiss-roll-diffusion-model-important-sampling-flow-matching",
            model_save_path="./model-swiss-roll-diffusion-model-important-sampling-flow-matching",
            guidance_scale=[0, 1, 2, 4, 8, 16],
        ),
    ),
)


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
        os.path.join(video_save_path, f"iteration_{iteration}.mp4"), fps=fps, dpi=dpi
    )
    # clean up
    plt.close(fig)
    plt.clf()


def save_checkpoint(model, iteration, path):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(
        dict(
            model=model.state_dict(),
            iteration=iteration,
        ),
        f=os.path.join(path, f"checkpoint_{iteration}.pt"),
    )


def save_checkpoint(diffusion_model, diffusion_model_iteration, path):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(
        dict(
            diffusion_model=diffusion_model.state_dict(),
            diffusion_model_iteration=diffusion_model_iteration,
        ),
        f=os.path.join(path, f"checkpoint_{diffusion_model_iteration}.pt"),
    )


if __name__ == "__main__":
    seed_value = set_seed()
    log.info(f"start exp with seed value {seed_value}.")

    diffusion_model = DiffusionModel(config.model.diffusion_model).to(
        config.model.diffusion_model.device
    )
    diffusion_model_important_sampling = DiffusionModel(
        config.model.diffusion_model
    ).to(config.model.diffusion_model.device)
    guidance_model = GuidedDiffusionModel(config.model.diffusion_model)
    diffusion_model = torch.compile(diffusion_model)
    diffusion_model_important_sampling = torch.compile(
        diffusion_model_important_sampling
    )

    if config.parameter.evaluation.model_save_path is not None:

        if not os.path.exists(config.parameter.evaluation.model_save_path):
            log.warning(
                f"Checkpoint path {config.parameter.evaluation.model_save_path} does not exist"
            )
            diffusion_model_iteration = 0
        else:
            checkpoint_files = [
                f
                for f in os.listdir(config.parameter.evaluation.model_save_path)
                if f.endswith(".pt")
            ]
            if len(checkpoint_files) == 0:
                log.warning(
                    f"No checkpoint files found in {config.parameter.evaluation.model_save_path}"
                )
                diffusion_model_iteration = 0
            else:
                checkpoint_files = sorted(
                    checkpoint_files, key=lambda x: int(x.split("_")[-1].split(".")[0])
                )
                checkpoint = torch.load(
                    os.path.join(
                        config.parameter.evaluation.model_save_path,
                        checkpoint_files[-1],
                    ),
                    map_location="cpu",
                )
                diffusion_model.load_state_dict(checkpoint["diffusion_model"])
                diffusion_model_important_sampling.load_state_dict(
                    checkpoint["diffusion_model"]
                )
                diffusion_model_iteration = checkpoint.get(
                    "diffusion_model_iteration", 0
                )

    else:
        diffusion_model_iteration = 0

    # get data
    x_and_t = make_swiss_roll(
        n_samples=config.dataset.data_num, noise=config.dataset.noise
    )
    t = x_and_t[1].astype(np.float32)
    value = ((t - np.min(t)) / (np.max(t) - np.min(t)) - 0.5) * 5 - 1.0
    x = x_and_t[0].astype(np.float32)[:, [0, 2]]
    # transform data
    x[:, 0] = x[:, 0] / np.max(np.abs(x[:, 0]))
    x[:, 1] = x[:, 1] / np.max(np.abs(x[:, 1]))
    x = (x - x.min()) / (x.max() - x.min())
    x = x * 10 - 5

    # plot data with color of value
    plt.scatter(x[:, 0], x[:, 1], c=value, vmin=-5, vmax=3)
    plt.colorbar()
    if not os.path.exists(config.parameter.evaluation.video_save_path):
        os.makedirs(config.parameter.evaluation.video_save_path)
    plt.savefig(
        os.path.join(
            config.parameter.evaluation.video_save_path, f"swiss_roll_data.png"
        )
    )
    plt.clf()

    # zip x and value
    data = np.concatenate([x, value[:, None]], axis=1)

    def get_train_data(dataloader):
        while True:
            yield from dataloader

    data_loader = torch.utils.data.DataLoader(
        data, batch_size=config.parameter.diffusion_model.batch_size, shuffle=True
    )
    data_generator = get_train_data(data_loader)

    diffusion_model_optimizer = torch.optim.Adam(
        diffusion_model.model.parameters(),
        lr=config.parameter.diffusion_model.learning_rate,
    )

    diffusion_model_important_sampling_optimizer = torch.optim.Adam(
        diffusion_model_important_sampling.model.parameters(),
        lr=config.parameter.diffusion_model.learning_rate,
    )

    moving_average_loss = 0.0

    subprocess_list = []

    for train_iter in track(
        range(config.parameter.diffusion_model.iterations),
        description="diffusion model training",
    ):
        if train_iter < diffusion_model_iteration:
            continue

        train_data = next(data_generator).to(config.model.diffusion_model.device)
        train_x, train_value = train_data[:, :x_size], train_data[:, x_size]
        diffusion_model_training_loss = diffusion_model.score_matching_loss(train_x)

        diffusion_model_optimizer.zero_grad()
        diffusion_model_training_loss.backward()
        gradien_norm = torch.nn.utils.clip_grad_norm_(
            diffusion_model.parameters(),
            config.parameter.diffusion_model.clip_grad_norm,
        )
        diffusion_model_optimizer.step()

        diffusion_model_important_sampling_training_loss = (
            diffusion_model_important_sampling.score_matching_loss(
                train_x, average=False
            )
        )
        diffusion_model_important_sampling_training_loss = torch.mean(
            diffusion_model_important_sampling_training_loss * torch.exp(train_value)
        )

        diffusion_model_important_sampling_optimizer.zero_grad()
        diffusion_model_important_sampling_training_loss.backward()
        gradien_norm = torch.nn.utils.clip_grad_norm_(
            diffusion_model_important_sampling.parameters(),
            config.parameter.diffusion_model.clip_grad_norm,
        )
        diffusion_model_important_sampling_optimizer.step()
        moving_average_loss = (
            0.99 * moving_average_loss
            + 0.01 * diffusion_model_important_sampling_training_loss.item()
            if train_iter > 0
            else diffusion_model_important_sampling_training_loss.item()
        )

        if train_iter % 100 == 0:
            log.info(
                f"iteration {train_iter}, gradient norm {gradien_norm}, unconditional model loss {diffusion_model_training_loss.item()}, model loss {diffusion_model_important_sampling_training_loss.item()}, moving average loss {moving_average_loss}"
            )

        diffusion_model_iteration = train_iter

        if (
            train_iter == 0
            or (train_iter + 1) % config.parameter.evaluation.eval_freq == 0
        ):
            diffusion_model.eval()

            t_span = torch.linspace(0.0, 1.0, 1000)
            x_t = (
                diffusion_model.sample_forward_process(t_span=t_span, batch_size=500)
                .cpu()
                .detach()
            )
            x_t = [
                x.squeeze(0) for x in torch.split(x_t, split_size_or_sections=1, dim=0)
            ]
            p = mp.Process(
                target=render_video,
                args=(
                    x_t,
                    config.parameter.evaluation.video_save_path,
                    f"unconditional_diffusion_model_iteration_{diffusion_model_iteration}",
                    100,
                    100,
                ),
            )
            p.start()
            subprocess_list.append(p)

            x_t = (
                diffusion_model_important_sampling.sample_forward_process(
                    t_span=t_span, batch_size=500
                )
                .cpu()
                .detach()
            )
            x_t = [
                x.squeeze(0) for x in torch.split(x_t, split_size_or_sections=1, dim=0)
            ]
            p1 = mp.Process(
                target=render_video,
                args=(
                    x_t,
                    config.parameter.evaluation.video_save_path,
                    f"conditional_diffusion_model_iteration_{diffusion_model_iteration}",
                    100,
                    100,
                ),
            )
            p1.start()
            subprocess_list.append(p1)

            for guidance_scale in config.parameter.evaluation.guidance_scale:
                x_t = (
                    guidance_model.sample_forward_process(
                        base_model=diffusion_model.model,
                        guided_model=diffusion_model_important_sampling.model,
                        t_span=t_span,
                        batch_size=500,
                        guidance_scale=guidance_scale,
                    )
                    .cpu()
                    .detach()
                )
                x_t = [
                    x.squeeze(0)
                    for x in torch.split(x_t, split_size_or_sections=1, dim=0)
                ]
                p2 = mp.Process(
                    target=render_video,
                    args=(
                        x_t,
                        config.parameter.evaluation.video_save_path,
                        f"guidance_diffusion_model_iteration_{diffusion_model_iteration}_scale_{guidance_scale}",
                        100,
                        100,
                    ),
                )
                p2.start()
                subprocess_list.append(p2)

            # save_checkpoint(diffusion_model=diffusion_model, diffusion_model_iteration=diffusion_model_iteration, path=config.parameter.evaluation.model_save_path)

    for p in subprocess_list:
        p.join()
