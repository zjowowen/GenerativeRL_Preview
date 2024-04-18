import os
import signal
import sys

import matplotlib
import numpy as np
from easydict import EasyDict
from rich.progress import track

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
from diffusers.models import AutoencoderKL
from easydict import EasyDict
from matplotlib import animation
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder

import wandb
from grl.generative_models.diffusion_model.diffusion_model import \
    DiffusionModel
from grl.utils import set_seed
from grl.utils.log import log

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
vae_hidden_size = 8
vae_channel_size = 4
x_size = (4, vae_hidden_size, vae_hidden_size)
image_size = 64
train_mode = "single_card"
assert train_mode in ["single_card", "ddp"]

config = EasyDict(
    dict(
        device = device,
        data=dict(
            image_size=image_size,
            data_path="./minecraft",
        ),
        model=dict(
            diffusion_model = dict(
                device = device,
                x_size = x_size,
                alpha = 1.0,
                solver = dict(
                    type = "ODESolver",
                    args = dict(
                        library="torchdyn",
                    ),
                ),
                path = dict(
                    type = "linear_vp_sde",
                    beta_0 = 0.1,
                    beta_1 = 20.0,
                ),
                model = dict(
                    type = "noise_function",
                    args = dict(
                        backbone = dict(
                            type = "DiT",
                            args = dict(
                                input_size = vae_hidden_size,
                                patch_size = 2,
                                in_channels = vae_channel_size,
                                hidden_size = 384,
                                depth = 12,
                                num_heads = 6,
                                learn_sigma = False,
                            ),
                        ),
                    ),
                ),
            ),
        ),
        parameter=dict(
            training_loss_type = "flow_matching",
            train_mode=train_mode,
            batch_size=256,
            eval_freq=100,
            learning_rate=5e-4,
            iterations=20000,
            clip_grad_norm=1.0,
            checkpoint_freq=100,
            checkpoint_path="./checkpoint",
            image_save_path="./images",
        ),
    )
)

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


if __name__ == "__main__":
    seed_vale=set_seed()
    with wandb.init(
        project=f"{config.project if hasattr(config, "project") else "dit-minecraft"}-{seed_vale}",
        **config.wandb if hasattr(config, "wandb") else {}
    ) as wandb_run:

        if config.parameter.train_mode == "ddp":
            torch.distributed.init_process_group("nccl")
            assert config.parameter.batch_size % torch.distributed.get_world_size() == 0, f"Batch size must be divisible by world size."
            device = torch.distributed.get_rank() % torch.cuda.device_count()
            torch.cuda.set_device(device)
            log.info(f"Starting rank={torch.distributed.get_rank()}, world_size={torch.distributed.get_world_size()}.")

        diffusion_model = DiffusionModel(config=config.model.diffusion_model)
        diffusion_model = torch.compile(diffusion_model)

        vae = AutoencoderKL.from_pretrained(f"/home/zjow/huggingface/sd-vae-ft-ema").to(device)

        if config.parameter.train_mode == "ddp":
            diffusion_model = nn.parallel.DistributedDataParallel(diffusion_model.to(config.model.diffusion_model.device), device_ids=[torch.distributed.get_rank()])
        else:
            diffusion_model = diffusion_model.to(config.model.diffusion_model.device)

        transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, config.data.image_size)),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])

        dataset = ImageFolder(config.data.data_path, transform=transform)

        if config.parameter.train_mode == "single_card":
            sampler = torch.utils.data.RandomSampler(dataset, replacement = False)
        else:
            sampler = torch.utils.data.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=True,
            )

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.parameter.batch_size if config.parameter.train_mode == "single_card" else int(config.parameter.batch_size // torch.distributed.get_world_size()),
            shuffle=False,
            sampler=sampler,
            num_workers=config.parameter.num_workers if hasattr(config.parameter, "num_workers") else 2,
            pin_memory=True,
            drop_last=True,
        )
        def get_train_data(dataloader):
            while True:
                yield from dataloader
        data_generator = get_train_data(data_loader)

        optimizer = torch.optim.Adam(
            diffusion_model.parameters(), 
            lr=config.parameter.learning_rate,
            )

        if config.parameter.checkpoint_path is not None:

            if not os.path.exists(config.parameter.checkpoint_path):
                log.warning(f"Checkpoint path {config.parameter.checkpoint_path} does not exist")
                last_iteration = -1
            else:
                checkpoint_files = [f for f in os.listdir(config.parameter.checkpoint_path) if f.endswith(".pt")]
                checkpoint_files = sorted(checkpoint_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
                checkpoint = torch.load(os.path.join(config.parameter.checkpoint_path, checkpoint_files[-1]), map_location="cpu")
                diffusion_model.load_state_dict(checkpoint["model"])
                optimizer.load_state_dict(checkpoint["optimizer"])
                last_iteration = checkpoint["iteration"]
        else:
            last_iteration = -1

        gradient_sum=0
        loss_sum=0
        counter=0

        def save_checkpoint(model, optimizer, iteration):
            if not os.path.exists(config.parameter.checkpoint_path):
                os.makedirs(config.parameter.checkpoint_path)
            torch.save(
                dict(
                    model=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    iteration=iteration,
                ),f=os.path.join(config.parameter.checkpoint_path, f"checkpoint_{iteration}.pt"))

        history_iteration = [-1]
        def save_checkpoint_on_exit(model, optimizer, iterations):
            def exit_handler(signal, frame):
                log.info("Saving checkpoint when exit...")
                save_checkpoint(model, optimizer, iteration=iterations[-1])
                log.info("Done.")
                sys.exit(0)
            signal.signal(signal.SIGINT, exit_handler)
        save_checkpoint_on_exit(diffusion_model, optimizer, history_iteration)

        for iteration in track(range(config.parameter.iterations), description="Training"):
            sampler.set_epoch(iteration)

            if iteration <= last_iteration:
                continue

            if iteration >= 0 and iteration % config.parameter.eval_freq == 0:
                diffusion_model.eval()
                t_span=torch.linspace(0.0, 1.0, 1000)
                x_t = diffusion_model.sample_forward_process(t_span=t_span, batch_size=1).detach()
                x_t=[x.squeeze(0) for x in torch.split(x_t, split_size_or_sections=1, dim=0)]
                # render_video(x_t, config.parameter.video_save_path, iteration, fps=100, dpi=100)
                # image_x_t = [vae.decode(z / 0.18215).sample for z in x_t]
                image_x_1 = vae.decode(x_t[-1] / 0.18215).sample
                if not os.path.exists(config.parameter.image_save_path):
                    os.makedirs(config.parameter.image_save_path)
                torchvision.utils.save_image(image_x_1, os.path.join(config.parameter.image_save_path, f"iteration_{iteration}.png"),  nrow=4, normalize=True, value_range=(-1, 1))
                image_x_1 = image_x_1[0].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
                image = wandb.Image(image_x_1, caption=f"iteration {iteration}")
                wandb_run.log(
                    data=dict(
                        image = image,
                    ),
                    commit=False)

            batch_data, label = next(data_generator)
            batch_data = batch_data.to(config.device)

            batch_data = vae.encode(batch_data).latent_dist.sample().mul_(0.18215)
            
            diffusion_model.train()
            if config.parameter.training_loss_type=="flow_matching":
                loss=diffusion_model.flow_matching_loss(batch_data)
            elif config.parameter.training_loss_type=="score_matching":
                loss=diffusion_model.score_matching_loss(batch_data)
            else:
                raise NotImplementedError("Unknown loss type")
            optimizer.zero_grad()
            loss.backward()
            gradien_norm = torch.nn.utils.clip_grad_norm_(diffusion_model.parameters(), config.parameter.clip_grad_norm)
            optimizer.step()
            gradient_sum+=gradien_norm.item()
            loss_sum+=loss.item()
            counter+=1

            log.info(f"iteration {iteration}, gradient {gradient_sum/counter}, loss {loss_sum/counter}")
            history_iteration.append(iteration)

            if iteration == config.parameter.iterations-1:
                diffusion_model.eval()
                t_span=torch.linspace(0.0, 1.0, 1000)
                x_t = diffusion_model.sample_forward_process(t_span=t_span, batch_size=1).detach()
                x_t=[x.squeeze(0) for x in torch.split(x_t, split_size_or_sections=1, dim=0)]
                # render_video(x_t, config.parameter.video_save_path, iteration, fps=100, dpi=100)
                # image_x_t = [vae.decode(z / 0.18215).sample for z in x_t]
                image_x_1 = vae.decode(x_t[-1] / 0.18215).sample
                if not os.path.exists(config.parameter.image_save_path):
                    os.makedirs(config.parameter.image_save_path)
                torchvision.utils.save_image(image_x_1, os.path.join(config.parameter.image_save_path, f"iteration_{iteration}.png"),  nrow=4, normalize=True, value_range=(-1, 1))
                image_x_1 = image_x_1[0].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
                image = wandb.Image(image_x_1, caption=f"iteration {iteration}")
                wandb_run.log(
                    data=dict(
                        image = image,
                    ),
                    commit=False)

            wandb_run.log(
                data=dict(
                    iteration=iteration,
                    loss=loss.item(),
                    average_loss=loss_sum/counter,
                    average_gradient=gradient_sum/counter,
                ),
                commit=True)

            if (iteration+1) % config.parameter.checkpoint_freq == 0:
                save_checkpoint(diffusion_model, optimizer, iteration)

        wandb.finish()
