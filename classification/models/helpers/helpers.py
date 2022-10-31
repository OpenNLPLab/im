import logging

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from .normlization import SimpleRMSNorm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def get_activation_fn(activation):
    if activation == "gelu":
        return F.gelu
    elif activation == "relu":
        return F.relu
    elif activation == "elu":
        return F.elu
    elif activation == "sigmoid":
        return F.sigmoid
    elif activation == "exp":
        return torch.exp
    elif activation == "leak":
        return F.leaky_relu
    elif activation == "1+elu":
        def f(x):
            return 1 + F.elu(x)
        return f
    elif activation == "silu":
        return F.silu
    else:
        return lambda x: x
    
def get_norm_fn(norm_type):
    if norm_type == "simplermsnorm":
        return SimpleRMSNorm
    else:
        return nn.LayerNorm

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def print_params(**kwargs):
    if is_main_process():
        logging.info(f"start print config of {kwargs['__class__']}")
        for key in kwargs:
            if key in ["__class__", "self"]:
                continue
            logging.info(f"{key}: {kwargs[key]}")
        logging.info(f"end print config of {kwargs['__class__']}")
