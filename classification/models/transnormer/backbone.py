# https://github.com/lucidrains/vit-pytorch/blob/main/norm_vit_pytorch/vit.py
import torch
import torch.nn.functional as F

from torch import nn
from einops import rearrange, repeat

from models.helpers import SimpleRMSNorm
from models.helpers import Urpe
from models.helpers import GLU
from models.helpers import FFN
from models.helpers import get_activation_fn, get_norm

##### no cls
class DiagBlockAttention(nn.Module):
    def __init__(
        self, 
        dim, 
        heads=8, 
        dim_head=64, 
        dropout=0., 
        r=7, # 每行的patch数量
        use_urpe=False,
        use_softmax=True,
        norm_type="layernorm",
        block_size=4,
        act_fun="relu",
    ):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.atten = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        self.r = r

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.use_urpe = use_urpe
        print(f"self.use_urpe {self.use_urpe}")
        if self.use_urpe:
            self.urpe = Urpe(core_matrix=1, p_matrix=3, embedding_dim=dim_head, theta_learned=True, dims=[2, 3])
        self.use_softmax = use_softmax
        print(f"self.use_softmax {self.use_softmax}")
        if not self.use_softmax:
            self.norm = get_norm(norm_type, inner_dim)
        self.block_size = block_size
        print(f"self.block_size {self.block_size}")
        print(f"act_fun {act_fun}")
        self.act_fun = get_activation_fn(act_fun)

    def transform(self, x, r, c):
        # reshape
        x = rearrange(x, 'b h (r c) d -> b h r c d', r=self.r)
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
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        if self.use_urpe:
            q = rearrange(q, 'b h (r c) d -> b h r c d', r=self.r)
            k = rearrange(k, 'b h (r c) d -> b h r c d', r=self.r)
            q = self.urpe(q)
            k = self.urpe(k)
            q = rearrange(q, 'b h r c d -> b h (r c) d')
            k = rearrange(k, 'b h r c d -> b h (r c) d')

        r = self.r
        c = q.shape[-2] // r
        # chunk
        # b h n m (g g) d
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
        heads=8, 
        dim_head=64, 
        dropout=0., 
        r=7, # 每行的patch数量
        use_urpe=False,
        norm_type="layernorm",
        act_fun="relu",
    ):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.dropout = nn.Dropout(dropout)
        self.r = r

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.use_urpe = use_urpe
        print(f"self.use_urpe {self.use_urpe}")
        if self.use_urpe:
            self.urpe = Urpe(core_matrix=1, p_matrix=3, embedding_dim=dim_head, theta_learned=True, dims=[2, 3])
        self.norm = get_norm(norm_type, inner_dim)
        print(f"act_fun {act_fun}")
        self.act_fun = get_activation_fn(act_fun)

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        q = self.act_fun(q)
        k = self.act_fun(k)
        if self.use_urpe:
            q = rearrange(q, 'b h (r c) d -> b h r c d', r=self.r)
            k = rearrange(k, 'b h (r c) d -> b h r c d', r=self.r)
            q = self.urpe(q)
            k = self.urpe(k)
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
        heads, 
        dim_head, 
        mlp_dim, 
        dropout=0., 
        drop_path=0., 
        r=7, 
        use_urpe=False, 
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
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            r=r,
            use_urpe=use_urpe,
        )
        if use_glu:
            self.feature_mixer = GLU(dim, glu_dim, glu_act)
        else:
            self.feature_mixer = FFN(dim, mlp_dim, dropout=dropout)
        self.token_norm = get_norm(norm_type, dim)
        self.feature_norm = get_norm(norm_type, dim)
    
    def get_attention(
        self, 
        type_index, 
        norm_type, 
        use_softmax, 
        block_size,
        block_act, 
        linear_act,
        dim,
        heads,
        dim_head,
        dropout,
        r,
        use_urpe,
    ):
        if type_index == 1:
            print("diag block")
            return DiagBlockAttention(
                dim=dim,
                heads=heads,
                dim_head=dim_head,
                dropout=dropout,
                r=r,
                use_urpe=use_urpe,
                use_softmax=use_softmax,
                norm_type=norm_type,
                block_size=block_size,
                act_fun=block_act,
            )
        else:
            print("norm linear")
            return NormLinearAttention(
                dim=dim,
                heads=heads,
                dim_head=dim_head,
                dropout=dropout,
                r=r,
                use_urpe=use_urpe,
                norm_type=norm_type,
                act_fun=linear_act,
            )
    
    def forward(self, x):
        x = x + self.drop_path(self.token_mixer(self.token_norm(x)))
        x = x + self.drop_path(self.feature_mixer(self.feature_norm(x)))

        return x
