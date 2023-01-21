# https://github.com/lucidrains/vit-pytorch/blob/main/norm_vit_pytorch/vit.py
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from models.helpers import (FFN, GLU, SimpleRMSNorm, Lrpe, get_activation_fn,
                            get_norm_fn, pair, print_params)
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from torch import nn

from ..helpers import get_patch_embedding
from .backbone import Block


##### no cls
class Vit(nn.Module):
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
        channels=3, 
        drop_rate=0., 
        emb_dropout=0., 
        drop_path_rate=0.,
        use_pos=True,
        # add
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
        
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        
        patch_dim = channels * patch_height * patch_width
        
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b h w (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),
        )
 
        self.use_pos = use_pos
        num_row_patches = image_height // patch_height
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        if self.use_pos:
            self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.layers = nn.ModuleList([])
        dim_head = dim // num_heads
        for i in range(depth):
            self.layers.append(
                Block(
                    dim=dim, 
                    num_heads=num_heads, 
                    dim_head=dim_head,
                    mlp_dim=mlp_dim, 
                    dropout=drop_rate, 
                    drop_path=drop_path_rate,
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
            )

        # classification head
        self.head = nn.Sequential(
            get_norm_fn(norm_type)(dim),
            nn.Linear(dim, num_classes)
        )
        
    def forward_features(self, x):
        B = x.shape[0]
        x = self.to_patch_embedding(x)
        x = rearrange(x, 'b h w d -> b (h w) d')
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
