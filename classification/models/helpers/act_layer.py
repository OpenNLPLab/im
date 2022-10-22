import torch
import torch.nn as nn
import torch.nn.functional as F

from .helpers import get_activation_fn


class ActLayer(nn.Module):
    def __init__(self, act_fun):
        super().__init__()
        self.act = get_activation_fn(act_fun)
        
    def forward(self, x):
        return self.act(x)
