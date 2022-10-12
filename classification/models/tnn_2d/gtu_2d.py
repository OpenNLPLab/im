import math
import numpy as np
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Parameter
from torch.nn import Dropout
import sys
from .tno_2d import SimpleRMSNorm, Tno2D
from einops import rearrange

class Gtu2d(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        # add
        index=0,
        act_fun="silu",
        causal=False,
        expand_ratio=2,
        resi_param=False,
        # norm
        use_norm=False,
        norm_type="simplermsnorm",
        # Toeplitz
        use_decay=False,
        use_multi_decay=False,
        rpe_embedding=512,
        rpe_act="relu",
        normalize=False,
        par_type=1,
        rpe_layers=3,
        residual=False,
        l=1, 
        transform_type=1,
        gamma=0.999,
        n=14,
        m=14,
        # token shift
        token_shift_type=-1,
    ):
        # add
        self.index = index

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
        self.act = self.get_act_fun(act_fun)
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
        self.l = l
        self.transform_type = transform_type
        self.gamma = gamma
        self.bias = bias
        self.rpe_layers = rpe_layers
        self.n = n
        self.m = m
        self.toep = Tno2D(
            h=self.num_heads, 
            n=self.n, 
            m=self.m,
            dim=self.head_dim,
            rpe_dim=self.rpe_embedding, 
            causal=self.causal, 
            use_decay=self.use_decay, 
            use_multi_decay=self.use_multi_decay,
            act=self.rpe_act,
            par_type=self.par_type,
            residual=self.residual,
            layers=self.rpe_layers,
            l=self.l,
            transform_type=self.transform_type,
            gamma=self.gamma,
            bias=self.bias,
        )
        print(f"self.num_heads {self.num_heads}")
        print(f"self.use_decay {self.use_decay}")
        print(f"self.use_multi_decay {self.use_multi_decay}")
        print(f"self.rpe_embedding {self.rpe_embedding}")
        print(f"self.rpe_act {self.rpe_act}")
        print(f"self.normalize {self.normalize}")
        print(f"self.par_type {self.par_type}")
        print(f"self.residual {self.residual}")
        print(f"self.n {self.n}")
        print(f"self.m {self.m}")
        print(f"self.transform_type {self.transform_type}")
        print(f"self.gamma {self.gamma}")
        print(f"bias {bias}")
        print(f"rpe_layers {rpe_layers}")
        
        
        # norm
        self.norm_type = norm_type
        self.pre_norm = self.get_norm_fun(self.norm_type, d2)
        
        self.use_norm = use_norm
        if self.use_norm:
            self.norm = self.get_norm_fun(norm_type, d1)
        print(f"use_norm {self.use_norm}")
        print(f"norm_type {self.norm_type}")
        
        self.token_shift_type = token_shift_type
        print(f"self.token_shift_type {self.token_shift_type}")
        if self.token_shift_type == 1:
            self.token_shift = nn.ZeroPad2d((0, 0, 1, -1))
        elif self.token_shift_type == 2:
            self.token_shift = nn.ZeroPad2d((0, 0, 1, -1))
            self.coef = 0.5

        # self.par_init()
        
    def par_init(self):
        nn.init.normal_(self.u_proj.weight, std=0.02)
        nn.init.normal_(self.u_proj.bias, std=0.02)
        nn.init.normal_(self.v_proj.weight, std=0.02)
        nn.init.normal_(self.v_proj.bias, std=0.02)
        nn.init.normal_(self.o.weight, std=0.02)
        nn.init.normal_(self.o.bias, std=0.02)

    def get_norm_fun(self, norm_type, embed_dim):
        if norm_type == "rmsnorm":
            print("here! rmsnorm")
            return RMSNorm(embed_dim)
        elif norm_type == "gatedrmsnorm":
            print("here! gatedrmsnorm")
            return GatedRMSNorm(embed_dim)
        elif norm_type == "simplermsnorm":
            print("here! simple rmsnorm")
            return SimpleRMSNorm(embed_dim)
        elif norm_type == "scalenorm":
            print("here! scale norm")
            return ScaleNorm(embed_dim)
        else:
            print("here! layer norm")
            return nn.LayerNorm(embed_dim)

    def get_act_fun(self, act_fun):
        print(act_fun)
        if act_fun == "gelu":
            return F.gelu
        elif act_fun == "relu":
            return F.relu
        elif act_fun == "elu":
            return F.elu
        elif act_fun == "sigmoid":
            return torch.sigmoid
        elif act_fun == "exp":
            return torch.exp
        elif act_fun == "leak":
            def f(x):
                return F.leaky_relu(x, negative_slope=self.negative_slope)
            return f
        elif act_fun == "1+elu":
            def f(x):
                return 1 + F.elu(x)
            return f
        elif act_fun == "silu":
            return F.silu
        elif self.act_fun == "relu2":
            def f(x):
                return torch.square(torch.relu(x))
            return f
        else:
            return lambda x: x

    def forward(self, x, H, W):
        # x: b, h * w, d
        num_heads = self.num_heads
        
        if self.token_shift_type == 1:
            x = self.token_shift(x)
        elif self.token_shift_type == 2:
            q1 = self.token_shift(x)
            x = self.coef * q1 + (1 - self.coef) * x

        shortcut, x = x, self.pre_norm(x)
        if self.resi_param:
            shortcut = shortcut * self.d
        u = self.act(self.u_proj(x))
        v = self.act(self.v_proj(x))
        # reshape
        v = rearrange(v, 'b n m (h d) -> b h n m d', h=num_heads)
        output = self.toep(v)
        output = rearrange(output, 'b h n m d -> b n m (h d)')
        output = u * output
        if self.use_norm:
            output = self.norm(output)
            
        output = self.o(output) + shortcut
        
        return output