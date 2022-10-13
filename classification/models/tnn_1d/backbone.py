import torch.nn as nn

from models.helpers import SimpleRMSNorm
from models.helpers import GLU
from .gtu import Gtu

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
        prenorm=False,
    ):
        super().__init__()
        self.token_mixer = Gtu(
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
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # v2 add
        self.norm = SimpleRMSNorm(dim)
        self.feature_mixer = GLU(
            d1=dim, 
            d2=glu_dim,
            act_fun=glu_act,
        )
        
        if prenorm:
            self.forward = self.forward_prenorm
        else:
            self.forward = self.forward_postnorm

    def forward_postnorm(self, x, H, W):
        x = x + self.drop_path(self.token_mixer(x, H, W))
        x = x + self.drop_path(self.norm(self.feature_mixer(x)))

        return x
    
    def forward_prenorm(self, x, H, W):
        x = x + self.drop_path(self.token_mixer(x, H, W))
        x = x + self.drop_path(self.feature_mixer(self.norm(x)))

        return x