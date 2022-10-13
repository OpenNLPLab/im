import torch
import math

from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg

from models.helpers import GLU, SimpleRMSNorm, pair, print_params

from .tno import Tno
from .backbone import Block

class TNNVit(nn.Module):
    def __init__(
        self, 
        img_size=224, 
        patch_size=16, 
        in_chans=3, 
        num_classes=1000, 
        embed_dim=384,
        rpe_embdding=96,
        num_heads=6,
        rpe_act="relu",
        glu_act="silu",
        glu_dim=576,
        depth=12, 
        expand_ratio=3,
        channels=3,
        emb_dropout=0,
        use_pos=True,
        # rpe
        use_decay=False,
        use_multi_decay=False,
        gamma=0.999,
        rpe_layers=3,
        # pad
        drop_rate=0.,
        drop_path_rate=0.,
        prenorm=False,
    ):
        super().__init__()
        # get local varables
        params = locals()
        # print params
        print_params(**params)
        
        self.num_classes = num_classes

        image_height, image_width = pair(img_size)
        patch_height, patch_width = pair(patch_size)
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        
        self.H = image_height // patch_height
        self.W = image_width // patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b h w (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, embed_dim),
        )
        self.use_pos = use_pos
        if self.use_pos:
            self.pos_embedding = nn.Parameter(torch.randn(1, self.H, self.W, embed_dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                Block(
                    dim=embed_dim, 
                    num_heads=num_heads, 
                    rpe_embedding=rpe_embdding, 
                    rpe_act=rpe_act,
                    glu_dim=glu_dim,
                    glu_act=glu_act,
                    expand_ratio=expand_ratio,
                    rpe_layers=rpe_layers,
                    use_decay=use_decay,
                    use_multi_decay=use_multi_decay,
                    gamma=gamma,
                    prenorm=prenorm,
                )
            )

        # classification head
        self.head = nn.Sequential(
            SimpleRMSNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.to_patch_embedding(x)
        
        if self.use_pos:
            x += self.pos_embedding
            
        x = self.dropout(x)

        for i, block in enumerate(self.layers):
            x = block(x, self.H, self.W)
            
        x = rearrange(x, 'b n m d -> b (n m) d')
        
        return x.mean(dim=1)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        return x