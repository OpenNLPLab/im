import math
import numpy as np
import torch
import torch.nn.functional as F

from typing import Dict, Optional, Tuple
from torch import Tensor, nn
from torch.nn import Parameter
from torch.nn import Dropout
from einops import rearrange

from models.helpers import SimpleRMSNorm
from models.helpers import get_activation_fn, get_norm_fun
from .tno import Tno

class Gtu(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        # add
        act_fun="silu",
        causal=False,
        expand_ratio=2,
        resi_param=False,
        # norm
        use_norm=False,
        norm_type="simplermsnorm",
        # Toeplizt
        use_decay=False,
        use_multi_decay=False,
        rpe_layers=3,
        rpe_embedding=512,
        rpe_act="relu",
        normalize=False,
        par_type=1,
        residual=False,
        transform_type=1,
        gamma=0.999,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.expand_ratio = expand_ratio
        self.resi_param = resi_param
        print(f"self.expand_ratio {self.expand_ratio}")
        print(f"self.resi_param {self.resi_param}")
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
        print(f"act_fun {act_fun}")
        print(f"causal {self.causal}")
        
        # toep
        self.use_decay = use_decay
        self.use_multi_decay = use_multi_decay
        self.rpe_embedding = rpe_embedding
        self.rpe_act = rpe_act
        self.normalize = normalize
        self.par_type = par_type
        self.residual = residual
        self.transform_type = transform_type
        self.gamma = gamma
        self.bias = bias
        
        self.toep1 = Tno(
            h=self.num_heads, 
            dim=self.head_dim,
            rpe_dim=self.rpe_embedding, 
            causal=self.causal, 
            use_decay=self.use_decay, 
            use_multi_decay=self.use_multi_decay,
            act=self.rpe_act,
            par_type=self.par_type,
            residual=self.residual,
            transform_type=self.transform_type,
            gamma=self.gamma,
            bias=self.bias,
            layers=rpe_layers,
        )
        self.toep2 = Tno(
            h=self.num_heads,  
            dim=self.head_dim,
            rpe_dim=self.rpe_embedding, 
            causal=self.causal, 
            use_decay=self.use_decay, 
            use_multi_decay=self.use_multi_decay,
            act=self.rpe_act,
            par_type=self.par_type,
            residual=self.residual,
            transform_type=self.transform_type,
            gamma=self.gamma,
            bias=self.bias,
            layers=rpe_layers,
        )

        print(f"self.num_heads {self.num_heads}")
        print(f"self.use_decay {self.use_decay}")
        print(f"self.use_multi_decay {self.use_multi_decay}")
        print(f"self.rpe_embedding {self.rpe_embedding}")
        print(f"self.rpe_act {self.rpe_act}")
        print(f"self.normalize {self.normalize}")
        print(f"self.par_type {self.par_type}")
        print(f"self.residual {self.residual}")
        print(f"self.transform_type {self.transform_type}")
        print(f"self.gamma {self.gamma}")
        print(f"bias {bias}")
        
        # norm
        self.norm_type = norm_type
        self.pre_norm = get_norm_fun(self.norm_type, d2)
        
        self.use_norm = use_norm
        if self.use_norm:
            self.norm = get_norm_fun(self.norm_type, d1)
        print(f"use_norm {self.use_norm}")
        print(f"norm_type {self.norm_type}")

        self.par_init()
        
    def par_init(self):
        nn.init.normal_(self.u_proj.weight, std=0.02)
        nn.init.normal_(self.u_proj.bias, std=0.02)
        nn.init.normal_(self.v_proj.weight, std=0.02)
        nn.init.normal_(self.v_proj.bias, std=0.02)
        nn.init.normal_(self.o.weight, std=0.02)
        nn.init.normal_(self.o.bias, std=0.02)
    
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