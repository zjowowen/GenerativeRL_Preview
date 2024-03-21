from .vpsde import VPSDE

DIFFUSION_PROCESS = {
    "VPSDE".lower(): VPSDE,
}

def get_diffusion_process(name:str):
    if name.lower() not in DIFFUSION_PROCESS:
        raise ValueError("Unknown activation function {}".format(name))
    return DIFFUSION_PROCESS[name.lower()]
