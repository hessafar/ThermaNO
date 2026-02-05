import torch
import torch.nn as nn
import numpy as np
from config import device

class Sin(nn.Module):
    def __init__(self, beta_init=1.0):
        super(Sin, self).__init__()
        # Initialize beta as a learnable parameter
        self.beta = nn.Parameter(torch.tensor(beta_init))

    def forward(self, x):
        return torch.sin(np.pi * self.beta * x)

class Swish(nn.Module):
    def __init__(self, beta_init=1.0):
        super(Swish, self).__init__()
        # Initialize beta as a learnable parameter
        self.beta = nn.Parameter(torch.tensor(beta_init))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)

def from_numpy_to_torch(np_array, grad):
    #from config import device
    return torch.tensor(np_array, requires_grad=grad).type(torch.get_default_dtype()).to(device)

def set_default_dtype1(dtype: torch.dtype) -> None:
    torch.set_default_dtype(dtype)
    print(f"Using {torch.get_default_dtype()} as default data type")