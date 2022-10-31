import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from models.helpers import SimpleRMSNorm, get_activation_fn, get_norm_fn
from torch import Tensor, nn
from torch.nn import Dropout, Parameter

from .tno import Tno


class Gtu(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        bias=True,
        act_fun="silu",
        causal=False,
        expand_ratio=3,
        resi_param=False,
        use_norm=False,
        norm_type="simplermsnorm",
        use_decay=False,
        use_multi_decay=False,
        rpe_layers=3,
        rpe_embedding=512,
        rpe_act="relu",
        normalize=False,
        par_type=1,
        residual=False,
        gamma=0.99,
        act_type="none",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.expand_ratio = expand_ratio
        self.resi_param = resi_param
        
        if self.resi_param:
            self.d = nn.Parameter(torch.randn(self.embed_dim))

        d1 = int(self.expand_ratio * embed_dim)
        d1 = (d1 // self.num_heads) * self.num_heads
        d2 = embed_dim
        self.head_dim = d1 // num_heads
        # d^2
        self.v_proj = nn.Linear(embed_dim, d1, bias=bias)
        # d^2
        self.u_proj = nn.Linear(embed_dim, d1, bias=bias)
        # d^2
        self.o = nn.Linear(d1, embed_dim, bias=bias)

        self.causal = causal
        self.act = get_activation_fn(act_fun)
        
        # toep
        self.use_decay = use_decay
        self.use_multi_decay = use_multi_decay
        self.rpe_embedding = rpe_embedding
        self.rpe_act = rpe_act
        self.normalize = normalize
        self.par_type = par_type
        self.residual = residual
        self.gamma = gamma
        self.bias = bias
        
        self.toep1 = Tno(
            h=num_heads, 
            dim=self.head_dim,
            rpe_dim=rpe_embedding, 
            causal=causal, 
            use_decay=use_decay, 
            use_multi_decay=use_multi_decay,
            residual=residual,
            act=rpe_act,
            par_type=par_type,
            gamma=gamma,
            bias=bias,
            act_type=act_type,
            layers=rpe_layers,
            norm_type=norm_type,
        )
        
        self.toep2 = Tno(
            h=num_heads, 
            dim=self.head_dim,
            rpe_dim=rpe_embedding, 
            causal=causal, 
            use_decay=use_decay, 
            use_multi_decay=use_multi_decay,
            residual=residual,
            act=rpe_act,
            par_type=par_type,
            gamma=gamma,
            bias=bias,
            act_type=act_type,
            layers=rpe_layers,
            norm_type=norm_type,
        )
        
        # norm
        self.norm_type = norm_type
        self.pre_norm = get_norm_fn(self.norm_type)(d2)
        
        self.use_norm = use_norm
        if self.use_norm:
            self.norm = get_norm_fn(self.norm_type)(d1)
    
    # col + row
    def forward(self, x, H, W):
        # x: b, h, w, d
        num_heads = self.num_heads

        shortcut, x = x, self.pre_norm(x)
        if self.resi_param:
            shortcut = shortcut * self.d
        u = self.act(self.u_proj(x))
        v = self.act(self.v_proj(x))
        # reshape
        v = rearrange(v, 'b H W (h d) -> b h H W d', h=num_heads)
        output = self.toep2(v, dim=-3, normalize=self.normalize) + self.toep1(v, dim=-2, normalize=self.normalize)
        output = rearrange(output, 'b h H W d -> b H W (h d)')
        output = u * output
        if self.use_norm:
            output = self.norm(output)
            
        output = self.o(output) + shortcut
        
        return output
