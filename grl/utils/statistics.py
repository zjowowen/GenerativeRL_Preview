import torch
import torch.nn as nn


def find_parameters(module):

    assert isinstance(module, nn.Module)

    # If called within DataParallel, parameters won't appear in module.parameters().
    if getattr(module, "_is_replica", False):

        def find_tensor_attributes(module):
            tuples = [
                (k, v)
                for k, v in module.__dict__.items()
                if torch.is_tensor(v) and v.requires_grad
            ]
            return tuples

        gen = module._named_members(get_members_fn=find_tensor_attributes)
        return [param for _, param in gen]
    else:
        return list(module.parameters())


def calculate_tensor_memory_size(tensor):
    memory_usage_in_bytes = tensor.element_size() * tensor.nelement()
    return memory_usage_in_bytes


def memory_allocated(device=torch.device("cuda")):
    return torch.cuda.memory_allocated(device) / (1024 * 1024 * 1024)
