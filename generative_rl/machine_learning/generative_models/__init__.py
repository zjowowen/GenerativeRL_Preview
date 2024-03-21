from .diffusion_model.diffusion_model import DiffusionModel

GENERATIVE_MODELS = {
    "DiffusionModel".lower(): DiffusionModel,
}

def get_generative_model(name:str):
    if name.lower() not in GENERATIVE_MODELS:
        raise ValueError("Unknown activation function {}".format(name))
    return GENERATIVE_MODELS[name.lower()]
