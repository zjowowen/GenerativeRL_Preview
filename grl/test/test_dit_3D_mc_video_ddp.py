import os

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '23333'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
rank_list = [0,1]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(rank_list[i]) for i in range(len(rank_list))])
import signal
import sys

import matplotlib
import numpy as np
from easydict import EasyDict
from rich.progress import track

matplotlib.use('Agg')
import time

import matplotlib.pyplot as plt
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torchvision
from diffusers.models import AutoencoderKL
from easydict import EasyDict
from matplotlib import animation
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder

import wandb
from grl.datasets.minecraft import MineRLVideoDataset
from grl.generative_models.diffusion_model.diffusion_model import \
    DiffusionModel
from grl.utils import set_seed
from grl.utils.config import merge_two_dicts_into_newone
from grl.utils.log import log


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


def main(rank, world_size):
    seed_value=set_seed()
    train_mode = "ddp"
    assert train_mode in ["single_card", "ddp"]

    batch_size_per_card = 2
    batch_size = len(rank_list) * batch_size_per_card

    if train_mode == "ddp":
        torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
        assert batch_size % torch.distributed.get_world_size() == 0, f"Batch size must be divisible by world size."
        device = torch.distributed.get_rank() % torch.cuda.device_count()
        torch.cuda.set_device(device)
        log.info(f"Starting rank={torch.distributed.get_rank()}, GPU:[{rank_list[torch.distributed.get_rank()]}], world_size={torch.distributed.get_world_size()}.")
    else:
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    image_size = 64
    channel_size = 3
    video_length = 10
    x_size = (video_length, channel_size, image_size, image_size)
    patch_block_size = (video_length, image_size, image_size)
    patch_size = (2, 4, 4)
    patch_grid_num = (video_length//patch_size[0], image_size//patch_size[1], image_size//patch_size[2])
    num_heads = 6
    hidden_size = np.sum(patch_grid_num) * 2 * num_heads * 2
    assert hidden_size % (np.sum(patch_grid_num) * 2 * num_heads) == 0, f"hidden_size must be divisible by patch_grid_num * 2 * num_heads."

    config = EasyDict(
        dict(
            device = device,
            project = 'dit-3D-mc-video-ddp',
            data=dict(
                image_size=image_size,
                data_path="./minecraft/MineRLBasaltFindCave-v0-100000",
                video_length=video_length,
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
                        type = "velocity_function",
                        args = dict(
                            backbone = dict(
                                type = "DiT_3D",
                                args = dict(
                                    patch_block_size = patch_block_size,
                                    patch_size = patch_size,
                                    in_channels = channel_size,
                                    hidden_size = hidden_size,
                                    depth = 6,
                                    num_heads = num_heads,
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
                batch_size=batch_size,
                eval_freq=100,
                learning_rate=5e-4,
                iterations=200000,
                clip_grad_norm=1.0,
                checkpoint_freq=100,
                checkpoint_path="./checkpoint",
                video_save_path="./videos",
            ),
        )
    )

    with wandb.init(
        project=config.project if hasattr(config, "project") else "dit-3D-mc-video-ddp",
        group=f"DDP-{time.time()}",
        **config.wandb if hasattr(config, "wandb") else {}
    ) as wandb_run:
        config=merge_two_dicts_into_newone(EasyDict(wandb_run.config), config)
        wandb_run.config.update(config)

        diffusion_model = DiffusionModel(config=config.model.diffusion_model)
        diffusion_model = torch.compile(diffusion_model)

        if config.parameter.train_mode == "ddp":
            diffusion_model = nn.parallel.DistributedDataParallel(diffusion_model.to(config.model.diffusion_model.device), device_ids=[torch.distributed.get_rank()], find_unused_parameters=True)
        else:
            diffusion_model = diffusion_model.to(config.model.diffusion_model.device)

        transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, config.data.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])

        dataset = MineRLVideoDataset(config=config.data, transform=transform)

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
            num_workers=config.parameter.num_workers if hasattr(config.parameter, "num_workers") else 0,
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
        if torch.distributed.get_rank() == 0:
            save_checkpoint_on_exit(diffusion_model, optimizer, history_iteration)

        for iteration in track(range(config.parameter.iterations), description="Training"):
            sampler.set_epoch(iteration)

            if iteration <= last_iteration:
                continue

            if iteration > 0 and iteration % config.parameter.eval_freq == 0:
                diffusion_model.eval()
                t_span=torch.linspace(0.0, 1.0, 1000)
                if config.parameter.train_mode == "ddp":
                    x_t = diffusion_model.module.sample_forward_process(t_span=t_span, batch_size=1).detach()
                else:
                    x_t = diffusion_model.sample_forward_process(t_span=t_span, batch_size=1).detach()
                video_x_1 = x_t[-1].squeeze(0).mul(0.5).add(0.5).mul(255).clamp(0, 255).permute(0, 2, 3, 1).to("cpu", torch.uint8)
                if not os.path.exists(config.parameter.video_save_path):
                    os.makedirs(config.parameter.video_save_path)
                torchvision.io.write_video(filename=os.path.join(config.parameter.video_save_path, f"iteration_{iteration}.mp4"), video_array=video_x_1, fps=20)
                video_x_1 = video_x_1.permute(0, 3, 1, 2).numpy()
                video = wandb.Video(video_x_1, caption=f"iteration {iteration}")
                wandb_run.log(
                    data=dict(
                        video = video,
                    ),
                    commit=False)

            batch_data = next(data_generator)
            batch_data = batch_data.to(config.device)

            diffusion_model.train()
            if config.parameter.train_mode == "ddp":
                if config.parameter.training_loss_type=="flow_matching":
                    loss=diffusion_model.module.flow_matching_loss(batch_data)
                elif config.parameter.training_loss_type=="score_matching":
                    loss=diffusion_model.module.score_matching_loss(batch_data)
                else:
                    raise NotImplementedError("Unknown loss type")
            else:
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
                if config.parameter.train_mode == "ddp":
                    x_t = diffusion_model.module.sample_forward_process(t_span=t_span, batch_size=1).detach()
                else:
                    x_t = diffusion_model.sample_forward_process(t_span=t_span, batch_size=1).detach()
                video_x_1 = x_t[-1].squeeze(0).mul(0.5).add(0.5).mul(255).clamp(0, 255).permute(0, 2, 3, 1).to("cpu", torch.uint8)
                if not os.path.exists(config.parameter.video_save_path):
                    os.makedirs(config.parameter.video_save_path)
                torchvision.io.write_video(filename=os.path.join(config.parameter.video_save_path, f"iteration_{iteration}.mp4"), video_array=video_x_1, fps=20)
                video_x_1 = video_x_1.permute(0, 3, 1, 2).numpy()
                video = wandb.Video(video_x_1, caption=f"iteration {iteration}")
                wandb_run.log(
                    data=dict(
                        video = video,
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

            if (iteration+1) % config.parameter.checkpoint_freq == 0 and torch.distributed.get_rank() == 0:
                save_checkpoint(diffusion_model, optimizer, iteration)
                torch.distributed.barrier()
            else:
                torch.distributed.barrier()

        wandb.finish()



def parallel_process(rank, world_size):
    main(rank, world_size)
    torch.distributed.destroy_process_group()

if __name__ == "__main__":
    world_size = len(rank_list)
    mp.spawn(parallel_process,
              args=(world_size,),
              nprocs=world_size,
              join=True)
