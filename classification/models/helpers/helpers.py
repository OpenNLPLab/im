from torch import nn

import torch
import torch.nn.functional as F
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

from .normlization import SimpleRMSNorm

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
    
def get_norm(norm_type, embed_dim):
    if norm_type == "simplermsnorm":
        return SimpleRMSNorm(embed_dim)
    else:
        return nn.LayerNorm(embed_dim)

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def print_params(**kwargs):
    logging.info(f"start print config of {kwargs['__class__']}")
    for key in kwargs:
        if key in ["__class__", "self"]:
            continue
        logging.info(f"{key}: {kwargs[key]}")
    logging.info(f"end print config of {kwargs['__class__']}")