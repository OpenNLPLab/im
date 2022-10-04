# https://alinush.github.io/2020/03/19/multiplying-a-vector-by-a-toeplitz-matrix.html
# https://stackoverflow.com/questions/69809789/is-there-any-way-to-create-a-tensor-with-a-specific-pattern-in-pytorch
# https://github.com/cheerss/CrossFormer/blob/main/models/crossformer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
from models.helpers import SimpleRMSNorm

class Rpe2d(nn.Module):
    def __init__(self, dim, outdim, residual, act="relu", bias=True, layers=3):
        super().__init__()
        self.residual = residual
        self.outdim = outdim
        self.pos_dim = dim
        self.act = act
        self.pos_proj = nn.Linear(2, self.pos_dim, bias=bias)
        self.layers = nn.ModuleList([])
        for i in range(layers):
            self.layers.append(
                nn.Sequential(
                    SimpleRMSNorm(self.pos_dim),
                    self.get_act(),
                    nn.Linear(self.pos_dim, self.pos_dim, bias=bias),
                )
            )
        self.out = nn.Sequential(
            SimpleRMSNorm(self.pos_dim),
            self.get_act(),
            nn.Linear(self.pos_dim, self.outdim, bias=bias)
        )
        
    def get_act(self):
        if self.act == "silu":
            return nn.SiLU(inplace=True)
        else:
            return nn.ReLU(inplace=True)

    def forward(self, biases):
        x = self.pos_proj(biases)
        if self.residual:
            for m in self.layers:
                x = m(x) + x
        else:
            for m in self.layers:
                x = m(x)
        x = self.out(x)

        return x

class Tno2D(nn.Module):
    def __init__(
        self, 
        h, 
        n,
        m,
        dim, 
        rpe_dim, 
        causal=False, 
        use_decay=False, 
        use_multi_decay=False, 
        residual=False, 
        act="relu", 
        par_type=1, 
        l=10,
        transform_type=1,
        gamma=0.999,
        bias=True,
        act_type="none",
        layers=3,
    ):
        super().__init__()
        self.h = h
        self.n = n
        self.m = m
        self.dim = dim
        self.causal = causal
        self.par_type = par_type
        
        # bttb
        index = self.get_index(n, m)
        # 2n - 1
        self.register_buffer("index", index)
        # 1d
        index_row, index_col = self.get_1d_index(n, m)
        self.register_buffer("index_row", index_row)
        self.register_buffer("index_col", index_col)

        self.use_decay = use_decay
        if self.use_decay:
            self.gamma = nn.Parameter(torch.ones(1) * gamma, requires_grad=False)
        self.use_multi_decay = use_multi_decay
        if self.use_multi_decay:
            self.lambda_ = gamma
            self.gamma = nn.Parameter(torch.randn(self.h, 1, 1, self.dim))

        self.rpe = Rpe2d(dim=rpe_dim, outdim=self.h * self.dim, residual=residual, bias=bias, layers=layers)  
        self.act_type = act_type
        self.act_fun = self.get_act_fun(self.act_type)

    def get_act_fun(self, act_fun):
        # print(act_fun)
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
                return F.leaky_relu(x)
            return f
        elif act_fun == "1+elu":
            def f(x):
                return 1 + F.elu(x)
            return f
        elif act_fun == "silu":
            return F.silu
        elif act_fun == "relu2":
            def f(x):
                return torch.square(torch.relu(x))
            return f
        elif act_fun == "cos":
            return torch.cos
        elif act_fun == "sin":
            return torch.sin
        else:
            return lambda x: x

    def get_arr(self, n):
        # 0, 1, ..., (n - 1), -(n - 1), ..., -1
        array = torch.cat([torch.zeros(1), torch.arange(1, n), -torch.arange(n - 1, 0, -1)]) * 1.0
        
        return array
    
    def get_1d_index(self, n, m):
        index_row = torch.cat([torch.zeros(1), torch.arange(1, n), torch.arange(n - 1, 0, -1)]).reshape(-1, 1, 1)
        index_col = torch.cat([torch.zeros(1), torch.arange(1, m), torch.arange(m - 1, 0, -1)]).reshape(1, -1, 1)
    
        return index_row, index_col
    
    def get_index(self, n, m):
        # 2n - 1: toeplitz, each size: 2m - 1
        row = self.get_arr(n)
        col = self.get_arr(m)
        # 2, (2n - 1), (2m - 1)
        grid = torch.stack(torch.meshgrid([row, col]))
        grid = rearrange(grid, 'd n m -> n m d')

        return grid
    
    def rpe_transform(self):
        # (2n - 1), (2m - 1), 2 -> (2n - 1), (2m - 1), d
        res = self.rpe(self.index)
        # (2n - 1), (2m - 1), (h * d) -> h, (2n - 1), (2m - 1), d
        res = rearrange(res, 'n m (h d) -> h n m d', h=self.h)

        return res
     
    def forward(self, x):
        # shape of x: b, h, n, m, d
        n = x.shape[-3]
        m = x.shape[-2]
        # update index
        if n > self.n or m > self.m:
            self.n = n
            self.m = m
            # 2D
            index = self.get_index(n, m).type_as(self.index)
            self.index = index
            # 1D
            index_row, index_col = self.get_1d_index(n, m)
            self.index_row = index_row.type_as(self.index_row)
            self.index_col = index_col.type_as(self.index_col)
        # coef: 1, h, (2n - 1), (2m - 1), d
        coef = self.rpe_transform().unsqueeze(0)

        if self.use_decay or self.use_multi_decay:
            if self.use_decay:
                gamma = self.gamma
            else:
                gamma = torch.sigmoid(self.gamma)
                gamma = self.lambda_ + (1 - self.lambda_) * gamma
            tmp = self.index_row + self.index_col
            print(tmp.shape)
            print(tmp[:, :, 0])
            gamma = gamma ** (self.index_row + self.index_col)
            coef = coef * gamma
            # if self.use_multi_decay:
            #     print(gamma[0, :, :, 0])
            # else:
            #     print(gamma[:, :, 0])
        # tmp = self.index_row + self.index_col
        # print(tmp.shape)

        # coef: 1, h, (2n - 1), (2m - 1), d
        # x: ..., h, n, m, d
        # print(coef[0, 0, :, :, 0])
        # print(coef.shape, x.shape)
        # # print(2 * n - 1, 2 * m - 1)
        # n1 = next_fast_len(2 * n - 1)
        # m1 = next_fast_len(2 * m - 1)
        n1 = 2 * n - 1
        m1 = 2 * m - 1
        # print(n1, m1)
        coef_fft = torch.fft.rfft2(coef, s=(n1, m1), dim=(-3, -2))
        x_fft = torch.fft.rfft2(x, s=(n1, m1), dim=(-3, -2))
        # print(coef_fft.shape, x_fft.shape)
        output_fft = coef_fft * x_fft
        # print(torch.fft.irfft2(output_fft, s=(n1, m1), dim=(-3, -2)).shape)
        output = torch.fft.irfft2(output_fft, s=(n1, m1), dim=(-3, -2))[:, :, :n, :m]

        return output
        ##### for test
        # matrix = self.toeplizt_matrix(n)
        # res = torch.einsum('...nme,...me->...ne', matrix, x)
        # # print(torch.norm(res - output))
        ##### for test

    def block_toeplizt_matrix(self, x):
        # shape of x: b, h, n, m, d
        h = x.shape[1]
        n = x.shape[-3]
        m = x.shape[-2]
        d = x.shape[-1]
        # update index
        if n > self.n or m > self.m:
            self.n = n
            self.m = m
            # 2D
            index = self.get_index(n, m).type_as(self.index)
            self.index = index
            # 1D
            index_row, index_col = self.get_1d_index(n, m)
            self.index_row = index_row.type_as(self.index_row)
            self.index_col = index_col.type_as(self.index_col)
        # coef: h, (2n - 1), (2m - 1), d
        coef = self.rpe_transform()
        
        l = n * m
        T0 = torch.zeros(h, l, l, d)
        for i in range(n):
            for j in range(n):
                s1 = i * m
                s2 = j * m
                # h, 2m - 1, d
                # # print("coef")
                # # print(T.shape)
                # # print(coef.shape)
                # col, row
                vals = coef[:, i - j, :, :]
                # # print(vals.shape)
                s, t = torch.ones(m, m).nonzero().T
                # # print(vals[:, t - s].reshape(self.h, m, m, -1).shape)
                T0[:, s1: s1 + m, s2: s2 + m, :] = vals[:, s - t].reshape(self.h, m, m, -1)
        # print(T0[0, :, :, 0])
        
        if self.use_decay or self.use_multi_decay:
            if self.use_decay:
                gamma = self.gamma
            else:
                gamma = torch.sigmoid(self.gamma)
                gamma = self.lambda_ + (1 - self.lambda_) * gamma
            gamma = gamma ** (self.index_row + self.index_col)
            coef = coef * gamma
        
        l = n * m
        T = torch.zeros(h, l, l, d)
        for i in range(n):
            for j in range(n):
                s1 = i * m
                s2 = j * m
                # h, 2m - 1, d
                # # print("coef")
                # # print(T.shape)
                # # print(coef.shape)
                # col, row
                vals = coef[:, i - j, :, :]
                # # print(vals.shape)
                s, t = torch.ones(m, m).nonzero().T
                # # print(vals[:, t - s].reshape(self.h, m, m, -1).shape)
                T[:, s1: s1 + m, s2: s2 + m, :] = vals[:, s - t].reshape(self.h, m, m, -1)
        # print(T.shape)
        # print(T[0, :, :, 0])
        # print(T[0, :, :, 0] / T0[0, :, :, 0])

        x = rearrange(x, 'b h n m d -> b h (n m) d')
        res = torch.einsum('h n m d, b h m d -> b h n d', T, x)
        res = rearrange(res, 'b h (n m) d -> b h n m d', n=n)

        return res