from .d4rl import D4RLDataset
from .minecraft import MineRLImageDataset, MineRLVideoDataset
from .qgpo import QGPOCustomizedDataset, QGPOD4RLDataset, QGPODataset, QGPOOnlineDataset
from .gpo import GPOCustomizedDataset, GPOD4RLDataset, GPODataset, GPOOnlineDataset, GPOD4RLOnlineDataset
from .minari_dataset import MinariDataset

DATASETS = {
    "MineRLVideoDataset".lower(): MineRLVideoDataset,
    "MineRLImageDataset".lower(): MineRLImageDataset,
    "QGPOD4RLDataset".lower(): QGPOD4RLDataset,
    "QGPODataset".lower(): QGPODataset,
    "D4RLDataset".lower(): D4RLDataset,
    "QGPOOnlineDataset".lower(): QGPOOnlineDataset,
    "QGPOCustomizedDataset".lower(): QGPOCustomizedDataset,
    "MinariDataset".lower(): MinariDataset,
    "GPODataset".lower(): GPODataset,
    "GPOOnlineDataset".lower(): GPOOnlineDataset,
    "GPOD4RLDataset".lower(): GPOD4RLDataset,
    "GPOCustomizedDataset".lower(): GPOCustomizedDataset,
    "GPOD4RLOnlineDataset".lower(): GPOD4RLOnlineDataset,
}


def get_dataset(type: str):
    if type.lower() not in DATASETS:
        raise KeyError(f"Invalid dataset type: {type}")
    return DATASETS[type.lower()]


def create_dataset(config, **kwargs):
    return get_dataset(config.type)(**config.args, **kwargs)
