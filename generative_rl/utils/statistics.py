import torch


def calculate_tensor_memory_size(tensor):
    memory_usage_in_bytes = tensor.element_size() * tensor.nelement()
    return memory_usage_in_bytes

def memory_allocated(device=torch.device("cuda")):
    return torch.cuda.memory_allocated(device)/(1024*1024*1024)
