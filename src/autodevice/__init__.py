import numpy as np
import torch
from . import core as __core

"""
Returns the device with the most free space (compute and memory)
TODO: multiple device configuration
"""
def device():
    if __core.cuda:
        device_matrix = __core.get_devices()

        free_compute = np.where(device_matrix[:, 3] == 0)

        if len(free_compute) > 0 and len(free_compute[0]) > 0:
            free_device_matrix = device_matrix[free_compute]
            best_device = free_compute[0][np.argmax(free_device_matrix[:, 2])]
            return torch.device(f"cuda:{best_device}")
        else:
            best_device = np.argmax(device_matrix[:, 2])
            return torch.device(f"cuda:{best_device}")
    else:
        return torch.device("cpu")
