import torch
from einops import rearrange
from models.helpers import GLU, SimpleRMSNorm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch import nn

from ..helpers import get_norm_fn
from hgru import Hgru1d, BiHgru1d


class Block(nn.Module):
    def __init__(
        self, 
        dim, 
        glu_dim,
        glu_act,
        drop_path=0., 
        prenorm=False,
        norm_type="layernorm",
        param_share=True,
    ):
        super().__init__()
        if param_share:
            self.token_mixer = Hgru1d(
                embed_dim=dim,
                causal=False
            )
        else:
            self.token_mixer = BiHgru1d(
                embed_dim=dim,
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

    def forward_postnorm(self, x, lower_bound=0):
        x = x + self.drop_path(self.token_norm(self.token_mixer(x, lower_bound)))
        x = x + self.drop_path(self.feature_norm(self.feature_mixer(x)))

        return x
    
    def forward_prenorm(self, x, lower_bound=0):
        x = x + self.drop_path(self.token_mixer(self.token_norm(x), lower_bound))
        x = x + self.drop_path(self.feature_mixer(self.feature_norm(x)))

        return x
