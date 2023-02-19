# https://alinush.github.io/2020/03/19/multiplying-a-vector-by-a-toeplitz-matrix.html
# https://stackoverflow.com/questions/69809789/is-there-any-way-to-create-a-tensor-with-a-specific-pattern-in-pytorch
# https://github.com/cheerss/CrossFormer/blob/main/models/crossformer.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from models.helpers import SimpleRMSNorm, get_activation_fn, get_norm_fn, print_params


class Rpe(nn.Module):
    def __init__(
        self, 
        in_dim, 
        out_dim, 
        act="relu", 
        bias=True, 
        layers=3, 
        norm_type="layernorm",
    ):
        super().__init__()
        # get local varables
        params = locals()
        # print params
        print_params(**params)
        
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.act = act
        self.pos_proj = nn.Linear(1, self.in_dim, bias=bias)
        self.layers = nn.ModuleList([])
        for i in range(layers):
            self.layers.append(
                nn.Sequential(
                    get_norm_fn(norm_type)(self.in_dim),
                    self.get_act(),
                    nn.Linear(self.in_dim, self.in_dim, bias=bias),
                )
            )
        self.out = nn.Sequential(
            get_norm_fn(norm_type)(self.in_dim),
            self.get_act(),
            nn.Linear(self.in_dim, self.out_dim, bias=bias),
        )
        
    def get_act(self):
        if self.act == "silu":
            return nn.SiLU(inplace=True)
        elif self.act == "relu":
            return nn.ReLU(inplace=True)
        else:
            return ActLayer(self.act)

    def forward(self, biases):
        x = self.pos_proj(biases)
        for m in self.layers:
            x = m(x) + x
        x = self.out(x)

        return x

class Tno(nn.Module):
    def __init__(
        self, 
        dim,
        rpe_in_dim=32,
        rpe_out_dim=16, 
        layers=3,
        causal=False, 
    ):
        super().__init__()
        # get local varables
        params = locals()
        # print params
        print_params(**params)
        
        self.dim = dim
        self.causal = causal
        self.rpe = Rpe(
            in_dim=rpe_in_dim, 
            out_dim=rpe_out_dim, 
            layers=layers,
        )
        self.coef = nn.Parameter(torch.randn(1, rpe_out_dim, dim), requires_grad=True)

    def forward(self, x, rpe_input, n=None, decay=None):
        # x: b, n, d
        # causal:
        # decay: 1, n, 1; lambda ^ (0, 1, ..., n - 1, 0, -(n-1), ... , -1)
        # non causal:
        # decay: 1, 2n - 1, 1; lambda ^ (0, 1, ..., n - 1, 0, -(n-1), ... , -1)
        if n == None:
            n = x.shape[-2]
        # (n, 1) -> (n, e) -> (n, e, 1)
        base = self.rpe(rpe_input).unsqueeze(-1)
        # a0, a1, ... , a(n-1), a0, a(-(n-1)), ... , a(-1)
        # (n, e, 1), (1, e, d) -> (n, d) -> (1, n, d)
        a = torch.sum(base * self.coef, dim=-2).unsqueeze(0)
        # (1, n, 1), (1, n, d) -> (1, n, d)
        if decay != None:
            a = decay * a
        # x: b, n, d; a: 1, n, d
        output = self.compute(x, a, n)

        return output
        
    def compute(self, x, a, n):
        # input: x: b, m, d; a: 1, max(n, m), d
        # output: b, n, d
        m = x.shape[-2]
        l = max(n, m)
        y = torch.fft.rfft(x, 2 * l, dim=-2)
        v = torch.fft.rfft(a, 2 * l, dim=-2)
        u = v * y
        output = torch.fft.irfft(u, 2 * l, dim=-2)[:, :n]

        return output

    def mm(self, x, decay, cos):
        # shape of x: ..., h, n, d
        n = x.shape[-2]
        # (h, 1, k, d), (1, n, k, 1) -> (h, n, k, d) -> (h, n, d)
        # a0, a1, ... , a(n-1), a0, a(-(n-1)), ... , a(-1)
        a = decay * torch.sum(self.coef * cos, dim=-2)
        zero = a[:, 0, None]
        pos = a[:, 1: n]
        if self.causal:
            neg = torch.zeros_like(pos)
        else:
            neg = a[:, n + 1:]
        c = torch.cat([zero, pos], dim=-2)
        r = torch.cat([zero, neg.flip(1)], dim=-2)
        vals = torch.cat([r, c[:, 1:].flip(1)], dim=-2)
        n = c.shape[-2]
        shape = self.h, n, n
        i, j = torch.ones(n, n).nonzero().T
        T = vals[:, j - i].reshape(self.h, n, n, -1)

        res = torch.einsum('h n m d, b h m d -> b h n d', T, x)
        
        return res, T

