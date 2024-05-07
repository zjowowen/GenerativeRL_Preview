import os
import signal
import sys
from typing import Optional, Union

import matplotlib
import numpy as np
from easydict import EasyDict
from rich.progress import track

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import treetensor
from diffusers.models import AutoencoderKL
from easydict import EasyDict
from matplotlib import animation
from PIL import Image
from tensordict import TensorDict
from timm.models.vision_transformer import Attention, Mlp, PatchEmbed
from torchvision import transforms
from torchvision.datasets import ImageFolder

import wandb
from grl.datasets.minecraft import MineRLImageDataset, MineRLVideoDataset
from grl.generative_models.diffusion_model.diffusion_model import DiffusionModel
from grl.generative_models.variational_autoencoder import VariationalAutoencoder
from grl.neural_network import TemporalSpatialResidualNet, register_module
from grl.neural_network.encoders import ExponentialFourierProjectionTimeEncoder
from grl.neural_network.transformers.dit import (
    DiTBlock,
    FinalLayer,
    get_2d_sincos_pos_embed,
)
from grl.utils import set_seed
from grl.utils.log import log


class DiT_Encoder(nn.Module):

    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
    ):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels * 2
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(
            input_size, patch_size, in_channels, hidden_size, bias=True
        )
        self.t_embedder = ExponentialFourierProjectionTimeEncoder(hidden_size)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, hidden_size), requires_grad=False
        )

        self.blocks = nn.ModuleList(
            [
                DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()
        # average pooling to shrink the spatial dimension by 4x:
        self.contraction = nn.AvgPool2d(4)

    def initialize_weights(self):
        """
        Overview:
            Initialize the weights of the model.
        """

        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.x_embedder.num_patches**0.5)
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        # nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        Overview:
            Unpatchify the input tensor.
        Arguments:
            x (:obj:`torch.Tensor`): The input tensor.
        Returns:
            imgs (:obj:`torch.Tensor`): The output tensor.
        Shapes:
            x (:obj:`torch.Tensor`): (N, T, patch_size**2 * C)
            imgs (:obj:`torch.Tensor`): (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(
        self,
        x: Union[dict, treetensor.torch.Tensor, TensorDict],
        condition: Union[
            torch.Tensor, dict, treetensor.torch.Tensor, TensorDict
        ] = None,
    ) -> Union[dict, treetensor.torch.Tensor, TensorDict]:
        """
        Overview:
            Forward pass of DiT.
        Arguments:
            t (:obj:`torch.Tensor`): Tensor of diffusion timesteps.
            x (:obj:`torch.Tensor`): Tensor of spatial inputs (images or latent representations of images).
            condition (:obj:`Union[torch.Tensor, TensorDict]`, optional): The input condition, such as class labels.
        """

        x = (
            self.x_embedder(x) + self.pos_embed
        )  # (N, T, D), where T = H * W / patch_size ** 2
        # t = self.t_embedder(t)                   # (N, D)

        # if condition is not None:
        # TODO: polish this part
        # c = t
        # y = self.y_embedder(condition, self.training)    # (N, D)
        # c = t + y                                # (N, D)
        # else:
        # c = t

        for i, block in enumerate(self.blocks):
            index = torch.tensor(
                i / len(self.blocks), device=x.device, dtype=torch.float32
            )
            c = self.t_embedder(index).repeat(x.shape[0], 1)
            x = block(x, c)  # (N, T, D)
        x = self.final_layer(x, c)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        x = self.contraction(x)  # (N, out_channels, H // 4, W // 4)
        x, x_log_var = torch.split(x, self.in_channels, dim=1)
        return x, x_log_var


class DiT_Decoder(nn.Module):

    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 1,
        in_channels: int = 4,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
    ):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(
            input_size, patch_size, in_channels, hidden_size, bias=True
        )
        self.t_embedder = ExponentialFourierProjectionTimeEncoder(hidden_size)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, hidden_size), requires_grad=False
        )

        self.blocks = nn.ModuleList(
            [
                DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()
        # extension the spatial dimension by 4x:
        self.extension = nn.Upsample(scale_factor=4, mode="nearest")

    def initialize_weights(self):
        """
        Overview:
            Initialize the weights of the model.
        """

        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.x_embedder.num_patches**0.5)
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        # nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        Overview:
            Unpatchify the input tensor.
        Arguments:
            x (:obj:`torch.Tensor`): The input tensor.
        Returns:
            imgs (:obj:`torch.Tensor`): The output tensor.
        Shapes:
            x (:obj:`torch.Tensor`): (N, T, patch_size**2 * C)
            imgs (:obj:`torch.Tensor`): (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(
        self,
        x: Union[dict, treetensor.torch.Tensor, TensorDict],
        condition: Union[
            torch.Tensor, dict, treetensor.torch.Tensor, TensorDict
        ] = None,
    ) -> Union[dict, treetensor.torch.Tensor, TensorDict]:

        x = self.extension(x)  # (N, out_channels, H * 4, W * 4)

        x = (
            self.x_embedder(x) + self.pos_embed
        )  # (N, T, D), where T = H * W / patch_size ** 2
        # t = self.t_embedder(t)                   # (N, D)

        # if condition is not None:
        # TODO: polish this part
        # c = t
        # y = self.y_embedder(condition, self.training)    # (N, D)
        # c = t + y                                # (N, D)
        # else:
        # c = t

        for i, block in enumerate(self.blocks):
            i = torch.tensor(i, device=x.device, dtype=torch.float32)
            c = self.t_embedder(i)
            x = block(x, c)  # (N, T, D)
        x = self.final_layer(x, c)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)

        return x


register_module(DiT_Encoder, "DiT_Encoder")
register_module(DiT_Decoder, "DiT_Decoder")


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

image_size = 64
x_size = (3, image_size, image_size)

z_size = (3, 16, 16)

train_mode = "single_card"
assert train_mode in ["single_card", "ddp"]

config = EasyDict(
    dict(
        project="dit-minecraft-vae",
        device=device,
        data=dict(
            image_size=image_size,
            data_path="./minerl_images/minerl-5000",
        ),
        vae=dict(
            device=device,
            input_dim=(3, 64, 64),
            output_dim=(3, 16, 16),
            encoder=dict(
                backbone=dict(
                    type="DiT_Encoder",
                    args=dict(
                        input_size=64,
                        patch_size=2,
                        in_channels=3,
                        hidden_size=384,
                        depth=6,
                        num_heads=6,
                    ),
                ),
            ),
            decoder=dict(
                backbone=dict(
                    type="DiT_Decoder",
                    args=dict(
                        input_size=64,
                        patch_size=1,
                        in_channels=3,
                        hidden_size=384,
                        depth=6,
                        num_heads=6,
                    ),
                ),
            ),
        ),
        model=dict(
            diffusion_model=dict(
                device=device,
                x_size=z_size,
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
                        backbone=dict(
                            type="DiT",
                            args=dict(
                                input_size=16,
                                patch_size=2,
                                in_channels=3,
                                hidden_size=384,
                                depth=12,
                                num_heads=6,
                                learn_sigma=False,
                            ),
                        ),
                    ),
                ),
            ),
        ),
        parameter=dict(
            training_loss_type="score_matching",
            train_mode=train_mode,
            batch_size=10,
            eval_freq=100,
            learning_rate=5e-4,
            iterations=20000,
            clip_grad_norm=1.0,
            checkpoint_freq=100,
            checkpoint_path="./checkpoint-vae",
            image_save_path="./images-vae",
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
    return Image.fromarray(
        arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
    )


if __name__ == "__main__":
    seed_vale = set_seed()
    with wandb.init(
        project=config.project, **config.wandb if hasattr(config, "wandb") else {}
    ) as wandb_run:

        if config.parameter.train_mode == "ddp":
            torch.distributed.init_process_group("nccl")
            assert (
                config.parameter.batch_size % torch.distributed.get_world_size() == 0
            ), f"Batch size must be divisible by world size."
            device = torch.distributed.get_rank() % torch.cuda.device_count()
            torch.cuda.set_device(device)
            log.info(
                f"Starting rank={torch.distributed.get_rank()}, world_size={torch.distributed.get_world_size()}."
            )

        diffusion_model = DiffusionModel(config=config.model.diffusion_model)
        diffusion_model = torch.compile(diffusion_model)

        vae = VariationalAutoencoder(config=config.vae)
        vae = torch.compile(vae)

        if config.parameter.train_mode == "ddp":
            diffusion_model = nn.parallel.DistributedDataParallel(
                diffusion_model.to(config.model.diffusion_model.device),
                device_ids=[torch.distributed.get_rank()],
            )
            vae = nn.parallel.DistributedDataParallel(
                vae.to(config.vae.device), device_ids=[torch.distributed.get_rank()]
            )
        else:
            diffusion_model = diffusion_model.to(config.model.diffusion_model.device)
            vae = vae.to(config.vae.device)

        transform = transforms.Compose(
            [
                transforms.Lambda(
                    lambda pil_image: center_crop_arr(pil_image, config.data.image_size)
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
                ),
            ]
        )

        dataset = MineRLImageDataset(config=config.data, transform=transform)

        if config.parameter.train_mode == "single_card":
            sampler = torch.utils.data.RandomSampler(dataset, replacement=False)
        else:
            sampler = torch.utils.data.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=True,
            )

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=(
                config.parameter.batch_size
                if config.parameter.train_mode == "single_card"
                else int(
                    config.parameter.batch_size // torch.distributed.get_world_size()
                )
            ),
            shuffle=False,
            sampler=sampler,
            num_workers=(
                config.parameter.num_workers
                if hasattr(config.parameter, "num_workers")
                else 2
            ),
            pin_memory=True,
            drop_last=True,
        )

        def get_train_data(dataloader):
            while True:
                yield from dataloader

        data_generator = get_train_data(data_loader)

        diffusion_model_optimizer = torch.optim.Adam(
            diffusion_model.parameters(),
            lr=config.parameter.learning_rate,
        )

        vae_optimizer = torch.optim.Adam(
            vae.parameters(),
            lr=config.parameter.learning_rate,
        )

        if config.parameter.checkpoint_path is not None:

            if not os.path.exists(config.parameter.checkpoint_path):
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
                diffusion_model.load_state_dict(checkpoint["diffusion_model"])
                vae.load_state_dict(checkpoint["vae"])
                last_iteration = checkpoint["iteration"]
        else:
            last_iteration = -1

        gradient_sum = 0
        loss_sum = 0
        counter = 0

        def save_checkpoint(diffusion_model, vae_model, iteration):
            if not os.path.exists(config.parameter.checkpoint_path):
                os.makedirs(config.parameter.checkpoint_path)
            torch.save(
                dict(
                    diffusion_model=diffusion_model.state_dict(),
                    vae_model=vae_model.state_dict(),
                    iteration=iteration,
                ),
                f=os.path.join(
                    config.parameter.checkpoint_path, f"checkpoint_{iteration}.pt"
                ),
            )

        history_iteration = [-1]

        def save_checkpoint_on_exit(diffusion_model, vae_model, iterations):
            def exit_handler(signal, frame):
                log.info("Saving checkpoint when exit...")
                save_checkpoint(diffusion_model, vae_model, iteration=iterations[-1])
                log.info("Done.")
                sys.exit(0)

            signal.signal(signal.SIGINT, exit_handler)

        save_checkpoint_on_exit(diffusion_model, vae, history_iteration)

        for iteration in track(
            range(config.parameter.iterations), description="Training"
        ):

            if iteration <= last_iteration:
                continue

            if False:  # iteration > 0 and iteration % config.parameter.eval_freq == 0:
                diffusion_model.eval()
                t_span = torch.linspace(0.0, 1.0, 1000)
                x_t = diffusion_model.sample_forward_process(
                    t_span=t_span, batch_size=1
                ).detach()
                x_t = [
                    x.squeeze(0)
                    for x in torch.split(x_t, split_size_or_sections=1, dim=0)
                ]

                image_x_1 = vae.decode(x_t[-1])
                if not os.path.exists(config.parameter.image_save_path):
                    os.makedirs(config.parameter.image_save_path)
                torchvision.utils.save_image(
                    image_x_1,
                    os.path.join(
                        config.parameter.image_save_path, f"iteration_{iteration}.png"
                    ),
                    nrow=4,
                    normalize=True,
                    value_range=(-1, 1),
                )
                image_x_1 = (
                    image_x_1[0]
                    .mul(255)
                    .add_(0.5)
                    .clamp_(0, 255)
                    .permute(1, 2, 0)
                    .to("cpu", torch.uint8)
                    .numpy()
                )
                image = wandb.Image(image_x_1, caption=f"iteration {iteration}")
                wandb_run.log(
                    data=dict(
                        image=image,
                    ),
                    commit=False,
                )

            batch_data = next(data_generator)
            batch_data = batch_data.to(config.device)

            batch_data = vae.encode(batch_data)

            diffusion_model.train()
            if config.parameter.training_loss_type == "flow_matching":
                loss = diffusion_model.flow_matching_loss(batch_data)
            elif config.parameter.training_loss_type == "score_matching":
                loss = diffusion_model.score_matching_loss(batch_data)
            else:
                raise NotImplementedError("Unknown loss type")
            diffusion_model_optimizer.zero_grad()
            loss.backward()
            gradien_norm = torch.nn.utils.clip_grad_norm_(
                diffusion_model.parameters(), config.parameter.clip_grad_norm
            )
            diffusion_model_optimizer.step()
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
                x_t = diffusion_model.sample_forward_process(
                    t_span=t_span, batch_size=1
                ).detach()
                x_t = [
                    x.squeeze(0)
                    for x in torch.split(x_t, split_size_or_sections=1, dim=0)
                ]
                # render_video(x_t, config.parameter.video_save_path, iteration, fps=100, dpi=100)
                # image_x_t = [vae.decode(z / 0.18215).sample for z in x_t]
                image_x_1 = vae.decode(x_t[-1] / 0.18215).sample
                if not os.path.exists(config.parameter.image_save_path):
                    os.makedirs(config.parameter.image_save_path)
                torchvision.utils.save_image(
                    image_x_1,
                    os.path.join(
                        config.parameter.image_save_path, f"iteration_{iteration}.png"
                    ),
                    nrow=4,
                    normalize=True,
                    value_range=(-1, 1),
                )
                image_x_1 = (
                    image_x_1[0]
                    .mul(255)
                    .add_(0.5)
                    .clamp_(0, 255)
                    .permute(1, 2, 0)
                    .to("cpu", torch.uint8)
                    .numpy()
                )
                image = wandb.Image(image_x_1, caption=f"iteration {iteration}")
                wandb_run.log(
                    data=dict(
                        image=image,
                    ),
                    commit=False,
                )

            wandb_run.log(
                data=dict(
                    iteration=iteration,
                    loss=loss.item(),
                    average_loss=loss_sum / counter,
                    average_gradient=gradient_sum / counter,
                ),
                commit=True,
            )

            if (iteration + 1) % config.parameter.checkpoint_freq == 0:
                save_checkpoint(diffusion_model, vae, iteration)

        wandb.finish()
