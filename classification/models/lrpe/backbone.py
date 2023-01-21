# https://github.com/lucidrains/vit-pytorch/blob/main/norm_vit_pytorch/vit.py
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from models.helpers import (FFN, GLU, SimpleRMSNorm, Lrpe, get_activation_fn,
                            get_norm_fn)
from torch import nn
from timm.models.layers import DropPath

from ..helpers import pair, Lrpe, rope, SineSPE, SPEFilter, print_params

##### no cls
class LinearAttention(nn.Module):
    def __init__(
        self, 
        dim, 
        num_heads=8, 
        dim_head=64, 
        dropout=0., 
        num_row_patches=7, # number of patches in a row
        norm_type="layernorm",
        act_fun="1+elu",
        # lrpe
        use_lrpe=False,
        core_matrix=1, 
        p_matrix=1, 
        theta_type="a",
        theta_learned=False, 
        householder_learned=False,
        # rope
        use_rope=False,
        # spe
        use_spe=False,
        # permutate
        use_permutate=False,
    ):
        super().__init__()
        # get local varables
        params = locals()
        # print params
        print_params(**params)
        
        inner_dim = dim_head *  num_heads

        self.num_heads = num_heads
        self.head_dim = dim_head
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
            self.lrpe = Lrpe(
                core_matrix=core_matrix, 
                p_matrix=p_matrix, 
                max_positions=num_row_patches,
                embedding_dim=self.head_dim, 
                theta_type=theta_type, 
                theta_learned=theta_learned, 
                householder_learned=householder_learned,
                dims=[-2, -3],
            )
            self.layer_norm = nn.LayerNorm(inner_dim)
        self.act = get_activation_fn(act_fun)
        self.use_spe = use_spe
        if self.use_spe:
            self.spe_encoder = SineSPE(num_heads=self.num_heads,          # Number of attention heads
                                       in_features=self.head_dim,       # Dimension of keys and queries
                                       num_realizations=self.head_dim,  # New dimension of keys and queries
                                       num_sines=1)          # Number of sinusoidal components
            self.spe_filter = SPEFilter(gated=True, code_shape=self.spe_encoder.code_shape)
        self.use_permutate = use_permutate
        if self.use_permutate:
            raw_permutation = self.generate_random_permutation(self.num_heads, self.kdim // self.num_heads, 0xdeadbeefdeadbeef)
            permutation = self.expand_permutation(max_seq_len, raw_permutation)
            self.register_buffer("permutation", permutation.unsqueeze(0))
            self.register_buffer("ratio", torch.sigmoid(torch.arange(self.num_heads) / self.num_heads * 3 + 2))
        
        self.act_fun = get_activation_fn(act_fun)

    # https://github.com/cpcp1998/PermuteFormer/blob/master/language_model/permute/__init__.py
    def generate_random_permutation(self, num_head, head_size, seed):
        rng = torch.Generator()
        rng.manual_seed(seed)
        permutate = torch.randperm(head_size, generator=rng)
        permutation = [permutate for _ in range(num_head)]
        # change to the same setting in urpe
        permutation = torch.stack(permutation, dim=0)
        return permutation

    # https://github.com/cpcp1998/PermuteFormer/blob/master/language_model/permute/__init__.py
    def expand_permutation(self, max_seq_length, permutation):
        num_head, head_size = permutation.shape
        expanded = [torch.arange(head_size).unsqueeze(0).expand(num_head, head_size)]
        for _ in range(max_seq_length - 1):
            previous = expanded[-1]
            current = previous.gather(-1, permutation)
            expanded.append(current)
        expanded = torch.stack(expanded, dim=1)
        return expanded

    def forward(self, x, eps=1e-4):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.num_heads), qkv)
        q = self.act_fun(q)
        k = self.act_fun(k)
        if self.use_lrpe:
            q = rearrange(q, 'b h (r c) d -> b h r c d', r=self.num_row_patches)
            k = rearrange(k, 'b h (r c) d -> b h r c d', r=self.num_row_patches)
            q = self.lrpe(q)
            k = self.lrpe(k)
            if self.lrpe.core_matrix == 4:
                q = torch.cat([q.real, q.imag], dim=-1)
                k = torch.cat([k.real, k.imag], dim=-1)
            q = rearrange(q, 'b h r c d -> b h (r c) d')
            k = rearrange(k, 'b h r c d -> b h (r c) d')

        if self.use_lrpe:
            kv = torch.einsum('...nm,...nd->...md', k, v)
            qkv = torch.einsum('...nm,...md->...nd', q, kv)
            out = rearrange(qkv, 'b h n d -> b n (h d)')
            out = self.layer_norm(out)
        else:
            kv = torch.einsum('...nm,...nd->...md', k, v)
            z = 1 / torch.clamp_min(torch.einsum('...ld,...d->...l', q, torch.sum(k, axis=-2)), eps)
            out = torch.einsum('...ld,...dm,...l->...lm', q, kv, z)
            out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)

class Block(nn.Module):
    def __init__(
        self, 
        dim, 
        num_heads, 
        dim_head, 
        mlp_dim, 
        dropout=0., 
        drop_path=0., 
        num_row_patches=7, # number of patches in a row
        norm_type="layernorm",
        act_fun="1+elu",
        # lrpe
        use_lrpe=False,
        core_matrix=1, 
        p_matrix=1, 
        theta_type="a",
        theta_learned=False, 
        householder_learned=False,
        # rope
        use_rope=False,
        # spe
        use_spe=False,
        # permutate
        use_permutate=False,
    ):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.token_mixer = LinearAttention(
            dim=dim, 
            num_heads=num_heads, 
            dim_head=dim_head, 
            dropout=dropout, 
            num_row_patches=num_row_patches, # number of patches in a row
            norm_type=norm_type,
            act_fun=act_fun,
            # lrpe
            use_lrpe=use_lrpe,
            core_matrix=core_matrix, 
            p_matrix=p_matrix, 
            theta_type=theta_type,
            theta_learned=theta_learned, 
            householder_learned=householder_learned,
            # rope
            use_rope=use_rope,
            # spe
            use_spe=use_spe,
            # permutate
            use_permutate=use_permutate,
        )
        self.feature_mixer = FFN(dim, mlp_dim, dropout=dropout)
        self.token_norm = get_norm_fn(norm_type)(dim)
        self.feature_norm = get_norm_fn(norm_type)(dim)
    
    def forward(self, x):
        x = x + self.drop_path(self.token_mixer(self.token_norm(x)))
        x = x + self.drop_path(self.feature_mixer(self.feature_norm(x)))

        return x
