# https://github.com/lucidrains/vit-pytorch/blob/main/norm_vit_pytorch/vit.py
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from models.helpers import (FFN, GLU, SimpleRMSNorm, Urpe, get_activation_fn,
                            get_norm_fn, pair, print_params)
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from torch import nn

from .backbone import Block, OverlapPatchEmbed


##### no cls
class Vin(nn.Module):
    def __init__(
        self, 
        *, 
        image_size=224, 
        patch_size, 
        num_classes, 
        dim, 
        depth, 
        num_heads, 
        mlp_dim, 
        pool='cls', 
        channels=3, 
        drop_rate=0., 
        emb_dropout=0., 
        drop_path_rate=0.,
        # add
        use_urpe=False, 
        # add
        type_list=[],
        norm_type="simplermsnorm",
        use_softmax=True,
        block_act="relu",
        block_size=4,
        linear_act="1+elu",
        use_pos=True,
        # glu
        use_glu=False,
        glu_act="silu",
        glu_dim=-1,
        # num_heads_list
        num_heads_list=[],
        # overlap patch
        use_over_lap=False,
        stride=-1
    ):
        super().__init__()
        if num_heads_list == []:
            num_heads_list = [num_heads] * depth
        # get local varables
        params = locals()
        # print params
        print_params(**params)
        
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        if use_over_lap and stride > 0:
            self.to_patch_embedding = OverlapPatchEmbed(
                patch_size=patch_size,
                stride=stride,
                in_chans=channels,
                embed_dim=dim,
            )
            num_patches = self.to_patch_embedding.num_patches
            r = self.to_patch_embedding.H
        else:
            self.to_patch_embedding = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
                nn.Linear(patch_dim, dim),
            )
            num_patches = (image_height // patch_height) * (image_width // patch_width)
            r = image_height // patch_height
 
        self.use_pos = use_pos
        if self.use_pos:
            self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.layers = nn.ModuleList([])
        assert len(type_list) == depth
        for i in range(depth):
            num_heads = num_heads_list[i]
            dim_head = dim // num_heads
            self.layers.append(
                Block(
                    dim=dim, 
                    depth=depth, 
                    num_heads=num_heads, 
                    dim_head=dim_head, 
                    mlp_dim=mlp_dim, 
                    dropout=drop_rate, 
                    r=r, 
                    use_urpe=use_urpe, 
                    type_index=type_list[i],
                    norm_type=norm_type,
                    use_softmax=use_softmax,
                    block_act=block_act,
                    block_size=block_size,
                    linear_act=linear_act,
                    use_glu=use_glu,
                    glu_act=glu_act,
                    glu_dim=glu_dim,
                )
            )

        # classification head
        self.head = nn.Sequential(
            get_norm_fn(norm_type)(dim),
            nn.Linear(dim, num_classes)
        )
        
    def forward_features(self, x):
        B = x.shape[0]
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape
        
        if self.use_pos:
            x += self.pos_embedding[:, :n]
        x = self.dropout(x)

        for i, block in enumerate(self.layers):
            x = block(x)
        
        return x.mean(dim=1)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        return x
