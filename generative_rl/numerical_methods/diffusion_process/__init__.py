from .vpsde import VPSDE

DIFFUSION_PROCESS = {
    "VPSDE": VPSDE,
}

def get_diffusion_process(name:str):
    if name not in DIFFUSION_PROCESS:
        raise ValueError("Unknown activation function {}".format(name))
    return DIFFUSION_PROCESS[name]
