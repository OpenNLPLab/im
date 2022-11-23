import torch
from einops import rearrange
from models.helpers import GLU, SimpleRMSNorm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch import nn

from ..helpers import get_norm_fn
from .gtu_2d import Gtu2d


class Block(nn.Module):
    def __init__(
        self, 
        dim, 
        num_heads,
        rpe_embedding,
        rpe_act,
        glu_dim,
        glu_act,
        drop_path=0., 
        expand_ratio=3,
        # rpe
        use_decay=False,
        use_multi_decay=False,
        gamma=0.999,
        rpe_layers=3,
        # n, m
        n=14,
        m=14,
        prenorm=False,
        norm_type="simplermsnorm",
    ):
        super().__init__()
        self.token_mixer = Gtu2d(
            embed_dim=dim,
            num_heads=num_heads,
            rpe_embedding=rpe_embedding,
            rpe_act="relu",
            expand_ratio=expand_ratio,
            # rpe
            rpe_layers=rpe_layers,
            use_decay=use_decay,
            use_multi_decay=use_multi_decay,
            gamma=gamma,
            # n, m
            n=n,
            m=m,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # v2 add
        self.token_norm = get_norm_fn(norm_type)(dim)
        self.feature_norm = get_norm_fn(norm_type)(dim)
        
        self.feature_mixer = GLU(
            d1=dim, 
            d2=glu_dim,
            act_fun=glu_act,
        )
        
        if prenorm:
            self.forward = self.forward_prenorm
        else:
            self.forward = self.forward_postnorm

    # def forward_postnorm(self, x, H, W):
    #     x = x + self.drop_path(self.token_mixer(x, H, W))
    #     x = x + self.drop_path(self.norm(self.feature_mixer(x)))

    #     return x
    
    # def forward_prenorm(self, x, H, W):
    #     x = x + self.drop_path(self.token_mixer(x, H, W))
    #     x = x + self.drop_path(self.feature_mixer(self.norm(x)))

    #     return x
    def forward_postnorm(self, x, H, W):
        x = x + self.drop_path(self.token_norm(self.token_mixer(x, H, W)))
        x = x + self.drop_path(self.feature_norm(self.feature_mixer(x)))

        return x
    
    def forward_prenorm(self, x, H, W):
        x = x + self.drop_path(self.token_mixer(self.token_norm(x), H, W))
        x = x + self.drop_path(self.feature_mixer(self.feature_norm(x)))

        return x
