import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from models.helpers import SimpleRMSNorm, get_activation_fn, get_norm_fn, print_params
from torch import Tensor, nn
from torch.nn import Dropout, Parameter

from .tno import Tno


class Dtu(nn.Module):
    def __init__(
        self,
        embed_dim,
        causal=False,
        expand_ratio=3,
        act_fun="silu",
        bias=True,
        # rpe
        rpe_in_dim=32,
        rpe_out_dim=16,
        layers=3,
        # decay
        use_decay=True,
        index=0,
        lambda_=0.95,
    ):
        super().__init__()
        # get local varables
        params = locals()
        # print params
        print_params(**params)
        
        self.embed_dim = embed_dim
        self.expand_ratio = expand_ratio
        self.index = index
        self.causal = causal

        d1 = int(self.expand_ratio * embed_dim)
        # linear projection
        # p T q v
        self.p_proj = nn.Linear(embed_dim, d1, bias=bias)
        self.q_proj = nn.Linear(embed_dim, d1, bias=bias)
        self.v_proj = nn.Linear(embed_dim, d1, bias=bias)
        self.o_proj = nn.Linear(d1, embed_dim, bias=bias)
        self.act = get_activation_fn(act_fun)
        self.tno1 = Tno(
            dim=d1,
            rpe_in_dim=rpe_in_dim,
            rpe_out_dim=rpe_out_dim,
            layers=layers,
            causal=causal, 
        )
        self.tno2 = Tno(
            dim=d1,
            rpe_in_dim=rpe_in_dim,
            rpe_out_dim=rpe_out_dim,
            layers=layers,
            causal=causal, 
        )
        # decay
        self.use_decay = use_decay
        if self.use_decay:
            self.lambda_ = lambda_
            # 1 / (d ^ (1 / i))
            slope = 1 / (torch.arange(1, d1 + 1) ** (1 / (index + 1)))
            self.slope = nn.Parameter(slope.reshape(1, -1), requires_grad=True)
        # max_seq_len
        self.max_seq = 0
        # index
        self.index = torch.empty(0)
        # rpe input
        self.rpe_pos = torch.empty(0)
        self.rpe_neg = torch.empty(0)
        self.rpe_zero = torch.empty(0)
        
    def update_cache(self, x):
        n = x.shape[1]
        if self.max_seq < n:
            self.max_seq = n
            # index
            self.index = torch.arange(1, n).reshape(1, -1, 1).to(x)
            # rpe input
            self.rpe_zero = torch.zeros(1, 1).to(x)
            self.rpe_pos = torch.arange(1, n).reshape(-1, 1).to(x)
            self.rpe_neg = -torch.flip(self.rpe_pos, dims=[1])
    
    def forward(self, x):
        # cache
        self.update_cache(x)
        n = x.shape[1]
        rpe_input = torch.cat([self.rpe_zero, self.rpe_pos[:n - 1], self.rpe_zero, self.rpe_neg[-(n-1):]], dim=0)
        index = self.index[:n]
        
        # x: b, h, w, d; assume h=w
        h, w = x.shape[1], x.shape[2]
        # projection
        p = self.act(self.p_proj(x))
        q = self.act(self.q_proj(x))
        v = self.act(self.v_proj(x))
        if self.use_decay:
            # decay
            slope = self.lambda_ + (1 - self.lambda_) * torch.clamp(self.slope, min=0, max=1)
            gamma = slope ** index
            zero = torch.ones(1, 1, v.shape[-1]).to(x)
            pos = gamma
            if not self.causal:
                neg = torch.flip(gamma, dims=[1])
                # 1, n, 1
                decay = torch.cat([zero, pos, zero, neg], dim=1)
            else:
                # 1, n, 1
                decay = torch.cat([zero, pos], dim=1)
        else:
            decay = None
        # gate1
        output = q * v
        # reshape
        o1 = rearrange(output, 'b h w d -> (b h) w d')
        o2 = rearrange(output, 'b h w d -> (b w) h d')
        o1 = self.tno1(o1, rpe_input=rpe_input, decay=decay)
        o2 = self.tno2(o2, rpe_input=rpe_input, decay=decay)
        output = rearrange(o1, '(b h) w d -> b h w d', h=h) + rearrange(o2, '(b w) h d -> b h w d', w=w)
        # gate2
        output = p * output
        # outproj
        output = self.o_proj(output)

        return output
