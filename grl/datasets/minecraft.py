import os

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from grl.utils.log import log


class MineRLVideoDataset(torch.utils.data.Dataset):
    """
    Overview:
        Dataset for MineRL video dataset.
    Interface:
        ``__init__``, ``__getitem__``, ``__len__``
    """

    def __init__(self, config, transform=None):
        """
        Overview:
            Initialize the dataset.
        Arguments:
            config (:obj:`EasyDict`): The configuration.
            transform (:obj:`torchvision.transforms.Compose`): The transformation.
        """
        self.config = config
        self.data_path = config.data_path
        self.video_length = config.video_length
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
            ])
        else:
            self.transform = transform

        self.videos = self._load()

    def _load(self):
        image_files = [f for f in os.listdir(self.data_path) if f.endswith('.png')]
        image_files.sort()

        videos = []
        for i in range(0, len(image_files), self.video_length):
            video_images = []
            for j in range(i, i + self.video_length):
                file = image_files[j]
                image = Image.open(os.path.join(self.data_path, file))
                image = self.transform(image)
                video_images.append(image)
                
            video = torch.stack(video_images)
            videos.append(video)

        return videos

    def __getitem__(self, index):
        video = self.videos[index]
        return video

    def __len__(self):
        return len(self.videos)

class MineRLImageDataset(torch.utils.data.Dataset):
    """
    Overview:
        Dataset for MineRL image dataset.
    Interface:
        ``__init__``, ``__getitem__``, ``__len__``
    """

    def __init__(self, config, transform=None):
        """
        Overview:
            Initialize the dataset.
        Arguments:
            config (:obj:`EasyDict`): The configuration.
            transform (:obj:`torchvision.transforms.Compose`): The transformation.
        """
        self.config = config
        self.data_path = config.data_path
        if transform is None or transform == "normalize":
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
            ])
        elif transform == "unnormalize":
            self.transform = transforms.Compose([
                transforms.Lambda(lambda x: transforms.functional.pil_to_tensor(x)),
            ])
        else:
            self.transform = transform
        self.images = self._load()

    def _load(self):
        image_files = [f for f in os.listdir(self.data_path) if f.endswith('.png')]
        image_files.sort()

        images = []
        for file in image_files:
            image = Image.open(os.path.join(self.data_path, file))
            image = self.transform(image)
            images.append(image)

        return images

    def __getitem__(self, index):
        image = self.images[index]
        return image

    def __len__(self):
        return len(self.images)
