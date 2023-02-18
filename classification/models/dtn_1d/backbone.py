import torch.nn as nn
from models.helpers import GLU, SimpleRMSNorm
from timm.models.layers import DropPath

from ..helpers import get_norm_fn
from .dtu import Dtu


class TokenFeatureBlock(nn.Module):
    def __init__(
        self, 
        embed_dim,
        drop_path=0., 
        causal=False,
        expand_ratio=3,
        dtu_act="silu",
        prenorm=True,
        norm_type="layernorm",
        # glu
        glu_dim=576,
        glu_act="silu",
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
        self.token_mixer = Dtu(
            embed_dim=embed_dim,
            causal=causal,
            expand_ratio=expand_ratio,
            act_fun=dtu_act,
            # rpe
            rpe_in_dim=rpe_in_dim,
            rpe_out_dim=rpe_out_dim,
            layers=layers,
            # decay
            use_decay=use_decay,
            index=index,
            lambda_=lambda_,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.token_norm = get_norm_fn(norm_type)(embed_dim)
        self.feature_norm = get_norm_fn(norm_type)(embed_dim)
        
        self.feature_mixer = GLU(
            d1=embed_dim, 
            d2=glu_dim,
            act_fun=glu_act,
        )
        
        if prenorm:
            self.forward = self.forward_prenorm
        else:
            self.forward = self.forward_postnorm

    def forward_postnorm(self, x):
        x = x + self.drop_path(self.token_norm(self.token_mixer(x)))
        x = x + self.drop_path(self.feature_norm(self.feature_mixer(x)))

        return x
    
    def forward_prenorm(self, x):
        x = x + self.drop_path(self.token_mixer(self.token_norm(x)))
        x = x + self.drop_path(self.feature_mixer(self.feature_norm(x)))

        return x

class Block(nn.Module):
    def __init__(
        self, 
        embed_dim,
        drop_path=0., 
        causal=False,
        expand_ratio=3,
        dtu_act="silu",
        prenorm=True,
        norm_type="layernorm",
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
        self.mixer = Dtu(
            embed_dim=embed_dim,
            causal=causal,
            expand_ratio=expand_ratio,
            act_fun=dtu_act,
            # rpe
            rpe_in_dim=rpe_in_dim,
            rpe_out_dim=rpe_out_dim,
            layers=layers,
            # decay
            use_decay=use_decay,
            index=index,
            lambda_=lambda_,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm = get_norm_fn(norm_type)(embed_dim)
        
        if prenorm:
            self.forward = self.forward_prenorm
        else:
            self.forward = self.forward_postnorm

    def forward_postnorm(self, x):
        x = x + self.drop_path(self.norm(self.mixer(x)))

        return x
    
    def forward_prenorm(self, x):
        x = x + self.drop_path(self.mixer(self.norm(x)))

        return x
