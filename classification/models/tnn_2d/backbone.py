import torch
from torch import nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange

from models.helpers import GLU, SimpleRMSNorm
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
        shrink_ratio=1,
        # rpe
        use_decay=False,
        use_multi_decay=False,
        gamma=0.999,
        rpe_layers=3,
        # n, m
        n=14,
        m=14,
        prenorm=False,
    ):
        super().__init__()
        self.token_mixer = Gtu2d(
            embed_dim=dim,
            num_heads=num_heads,
            rpe_embedding=rpe_embedding,
            rpe_act="silu",
            expand_ratio=expand_ratio,
            shrink_ratio=shrink_ratio,
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
        self.norm = SimpleRMSNorm(dim)
        self.mlp = GLU(
            d1=dim, 
            d2=glu_dim,
            act_fun=glu_act,
        )

        self.apply(self._init_weights)
        print(f"prenorm {prenorm}")
        
        if prenorm:
            self.forward = self.forward_prenorm
        else:
            self.forward = self.forward_postnorm

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward_postnorm(self, x, H, W):
        x = x + self.drop_path(self.token_mixer(x, H, W))
        x = x + self.drop_path(self.norm(self.mlp(x)))

        return x
    
    def forward_prenorm(self, x, H, W):
        x = x + self.drop_path(self.token_mixer(x, H, W))
        x = x + self.drop_path(self.mlp(self.norm(x)))

        return x

# ref: https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
class DownSample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = SimpleRMSNorm(4 * dim)

    def forward(self, x):
        """
        x: B, H, W, C
        """
        B, H, W, C = x.shape
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
        H1 = H // 2
        W1 = W // 2
        
        x = rearrange(x, 'b (k H) W d -> b H W (k d)', k=2, H=H1)
        x = rearrange(x, 'b H (k W) d -> b H W (k d)', k=2, W=W1)
        x = self.norm(x)
        x = self.reduction(x)

        return x