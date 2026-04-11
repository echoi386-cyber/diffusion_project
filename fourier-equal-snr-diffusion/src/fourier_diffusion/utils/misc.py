import numpy as np
import torch

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def device_auto():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")