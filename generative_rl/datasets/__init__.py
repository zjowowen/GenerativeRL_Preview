from .minecraft import MineRLVideoDataset, MineRLImageDataset
from .qgpo import QGPOD4RLDataset, QGPODataset

DATASETS = {
    "MineRLVideoDataset".lower(): MineRLVideoDataset,
    "MineRLImageDataset".lower(): MineRLImageDataset,
    "QGPOD4RLDataset".lower(): QGPOD4RLDataset,
    "QGPODataset".lower(): QGPODataset,
}

def get_dataset(type: str):
    if type.lower() not in DATASETS:
        raise KeyError(f'Invalid dataset type: {type}')
    return DATASETS[type.lower()]
 
def create_dataset(config, **kwargs):
    return get_dataset(config.type)(**config.args, **kwargs)
