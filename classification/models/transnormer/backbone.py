# https://github.com/lucidrains/vit-pytorch/blob/main/norm_vit_pytorch/vit.py
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from models.helpers import (FFN, GLU, SimpleRMSNorm, Lrpe, get_activation_fn,
                            get_norm_fn)
from torch import nn

from ..helpers import pair


##### no cls
class DiagBlockAttention(nn.Module):
    def __init__(
        self, 
        dim, 
        num_heads=8, 
        dim_head=64, 
        dropout=0., 
        num_row_patches=7, # number of patches in a row
        use_lrpe=False,
        use_softmax=True,
        norm_type="layernorm",
        block_size=4,
        act_fun="relu",
    ):
        super().__init__()
        inner_dim = dim_head *  num_heads

        self.num_heads = num_heads
        self.scale = dim_head ** -0.5

        self.atten = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        self.num_row_patches = num_row_patches

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

        self.use_lrpe = use_lrpe
        if self.use_lrpe:
            self.lrpe = Lrpe(core_matrix=1, p_matrix=3, embedding_dim=dim_head, theta_learned=True, dims=[2, 3])
        self.use_softmax = use_softmax
        if not self.use_softmax:
            self.norm = get_norm_fn(norm_type)(inner_dim)
        self.block_size = block_size
        self.act_fun = get_activation_fn(act_fun)

    def transform(self, x, r, c):
        # reshape
        x = rearrange(x, 'b h (r c) d -> b h r c d', r=self.num_row_patches)
        # pad
        r_pad = (self.block_size - r % self.block_size) % self.block_size
        c_pad = (self.block_size - c % self.block_size) % self.block_size
        x = F.pad(x, (0, 0, 0, r_pad, 0, c_pad, 0, 0, 0, 0))
        # reshape
        x = rearrange(x, 'b h (n g) (m e) d -> b h n g m e d', g=self.block_size, e=self.block_size)
        x = rearrange(x, 'b h n g m e d -> b h n m (g e) d')

        return x

    def reverse_transform(self, x, r, c):
        x = rearrange(x, 'b h n m (g e) d -> b h n g m e d', g=self.block_size, e=self.block_size)
        x = rearrange(x, 'b h n g m e d -> b h (n g) (m e) d')
        # remove pad
        x = x[:, :, :r, :c, :]
        x = rearrange(x, 'b h r c d -> b h (r c) d')

        return x

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.num_heads), qkv)
        if self.use_lrpe:
            q = rearrange(q, 'b h (r c) d -> b h r c d', r=self.num_row_patches)
            k = rearrange(k, 'b h (r c) d -> b h r c d', r=self.num_row_patches)
            q = self.lrpe(q)
            k = self.lrpe(k)
            q = rearrange(q, 'b h r c d -> b h (r c) d')
            k = rearrange(k, 'b h r c d -> b h (r c) d')

        r = self.num_row_patches
        c = q.shape[-2] // r
        # chunk
        # b h (n g) (m e) d -> b h n m (g e) d
        q = self.transform(q, r, c)
        k = self.transform(k, r, c)
        v = self.transform(v, r, c)

        dots = torch.einsum('...nd,...md->...nm', q, k) * self.scale
        if self.use_softmax:
            attn = self.atten(dots)
        else:
            attn = self.act_fun(dots)
        attn = self.dropout(attn)

        out = torch.einsum("...ls,...sd->...ld", attn, v)
        out = self.reverse_transform(out, r, c)
        out = rearrange(out, 'b h n d -> b n (h d)')
        if not self.use_softmax:
            out = self.norm(out)
        return self.to_out(out)

##### no cls
class NormLinearAttention(nn.Module):
    def __init__(
        self, 
        dim, 
        num_heads=8, 
        dim_head=64, 
        dropout=0., 
        num_row_patches=7, # number of patches in a row
        use_lrpe=False,
        norm_type="layernorm",
        act_fun="relu",
    ):
        super().__init__()
        inner_dim = dim_head *  num_heads

        self.num_heads = num_heads
        self.scale = dim_head ** -0.5

        self.dropout = nn.Dropout(dropout)
        self.num_row_patches = num_row_patches

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

        self.use_lrpe = use_lrpe
        if self.use_lrpe:
            self.lrpe = Lrpe(core_matrix=1, p_matrix=3, embedding_dim=dim_head, theta_learned=True, dims=[2, 3])
        self.norm = get_norm_fn(norm_type)(inner_dim)
        self.act_fun = get_activation_fn(act_fun)

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.num_heads), qkv)
        q = self.act_fun(q)
        k = self.act_fun(k)
        if self.use_lrpe:
            q = rearrange(q, 'b h (r c) d -> b h r c d', r=self.num_row_patches)
            k = rearrange(k, 'b h (r c) d -> b h r c d', r=self.num_row_patches)
            q = self.lrpe(q)
            k = self.lrpe(k)
            q = rearrange(q, 'b h r c d -> b h (r c) d')
            k = rearrange(k, 'b h r c d -> b h (r c) d')

        kv = torch.einsum('...nm,...nd->...md', k, v)
        qkv = torch.einsum('...nm,...md->...nd', q, kv)
        out = rearrange(qkv, 'b h n d -> b n (h d)')
        out = self.norm(out)

        return self.to_out(out)

class Block(nn.Module):
    def __init__(
        self, 
        dim, 
        depth, 
        num_heads, 
        dim_head, 
        mlp_dim, 
        dropout=0., 
        drop_path=0., 
        num_row_patches=7, # number of patches in a row
        use_lrpe=False, 
        # add
        type_index=-1,
        norm_type="simplermsnorm",
        use_softmax=True,
        block_act="relu",
        block_size=4,
        linear_act="1+elu",
        # glu
        use_glu=False,
        glu_act="silu",
        glu_dim=-1,
    ):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.token_mixer = self.get_attention(
            type_index=type_index, 
            norm_type=norm_type,
            use_softmax=use_softmax, 
            block_size=block_size,
            block_act=block_act, 
            linear_act=linear_act,
            dim=dim,
            num_heads=num_heads,
            dim_head=dim_head,
            dropout=dropout,
            num_row_patches=num_row_patches,
            use_lrpe=use_lrpe,
        )
        if use_glu:
            self.feature_mixer = GLU(dim, glu_dim, glu_act)
        else:
            self.feature_mixer = FFN(dim, mlp_dim, dropout=dropout)
        self.token_norm = get_norm_fn(norm_type)(dim)
        self.feature_norm = get_norm_fn(norm_type)(dim)
    
    def get_attention(
        self, 
        type_index, 
        norm_type, 
        use_softmax, 
        block_size,
        block_act, 
        linear_act,
        dim,
        num_heads,
        dim_head,
        dropout,
        num_row_patches,
        use_lrpe,
    ):
        if type_index == 1:
            return DiagBlockAttention(
                dim=dim,
                num_heads=num_heads,
                dim_head=dim_head,
                dropout=dropout,
                num_row_patches=num_row_patches,
                use_lrpe=use_lrpe,
                use_softmax=use_softmax,
                norm_type=norm_type,
                block_size=block_size,
                act_fun=block_act,
            )
        else:
            return NormLinearAttention(
                dim=dim,
                num_heads=num_heads,
                dim_head=dim_head,
                dropout=dropout,
                num_row_patches=num_row_patches,
                use_lrpe=use_lrpe,
                norm_type=norm_type,
                act_fun=linear_act,
            )
    
    def forward(self, x):
        x = x + self.drop_path(self.token_mixer(self.token_norm(x)))
        x = x + self.drop_path(self.feature_mixer(self.feature_norm(x)))

        return x
