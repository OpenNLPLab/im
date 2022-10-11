# https://github.com/lucidrains/vit-pytorch/blob/main/norm_vit_pytorch/vit.py
import torch
import torch.nn.functional as F
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg

from models.helpers import SimpleRMSNorm
from models.helpers import Urpe
from models.helpers import GLU
from models.helpers import FFN
from models.helpers import get_activation_fn, get_norm, pair

class PreNorm(nn.Module):
    def __init__(self, dim, fn, norm_type):
        super().__init__()
        self.norm = get_norm(norm_type, dim)
        self.fn = fn
        
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

##### no cls
class Transnormer(nn.Module):
    def __init__(
        self, 
        dim, 
        depth, 
        heads, 
        dim_head, 
        mlp_dim, 
        dropout=0., 
        r=7, 
        use_urpe=False, 
        # add
        type_list=[],
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
        assert len(type_list) == depth
        self.layers = nn.ModuleList([])
        for i in range(depth):
            Attention = self.get_attention(
                type_index=type_list[i], 
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
                FFN = GLU(dim, glu_dim, glu_act)
            else:
                FFN = FeedForward(dim, mlp_dim, dropout=dropout)
            self.layers.append(nn.ModuleList([
                    PreNorm(dim, Attention, norm_type),
                    PreNorm(dim, FFN, norm_type)
            ]))
    
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
            print("block")
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
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

##### no cls
class NormViT(nn.Module):
    def __init__(
        self, *, 
        image_size=224, 
        patch_size, 
        num_classes, 
        dim, depth, 
        heads, 
        mlp_dim, 
        pool='cls', 
        channels=3, 
        drop_rate=0., 
        emb_dropout = 0., 
        drop_path_rate=0.,
        # add
        r=7, 
        use_urpe=False, 
        # add
        type_list=[],
        norm_type="simplermsnorm",
        use_softmax=True,
        block_act="relu",
        block_size=4,
        linear_act="1+elu",
        use_abs=True,
        # glu
        use_glu=False,
        glu_act="silu",
        glu_dim=-1,
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.use_abs = use_abs
        print(f"use_abs {use_abs}")
        if self.use_abs:
            self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        dim_head = dim // heads
        self.transnormer = Transnormer(
            dim=dim, 
            depth=depth, 
            heads=heads, 
            dim_head=dim_head, 
            mlp_dim=mlp_dim, 
            dropout=drop_rate, 
            r=image_height // patch_height, 
            use_urpe=use_urpe, 
            type_list=type_list,
            norm_type=norm_type,
            use_softmax=use_softmax,
            block_act=block_act,
            block_size=block_size,
            linear_act=linear_act,
            use_glu=use_glu,
            glu_act=glu_act,
            glu_dim=glu_dim,
        )

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            get_norm(norm_type, dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        if self.use_abs:
            x += self.pos_embedding[:, :n]
        x = self.dropout(x)
 
        x = self.transnormer(x)

        x = x.mean(dim = 1)

        x = self.to_latent(x)
        return self.mlp_head(x)

########## Deit tiny
##### base
@register_model
def norm_vit_tiny_patch16_224_mix_softmax_1_elu_rmsnorm(pretrained=False, **kwargs):
    patch_size = 16
    dim = 192
    depth = 12
    heads = 6
    mlp_dim = 4 * dim
    dropout = 0.0
    r = 224 // patch_size
    use_urpe = True
    type_list = [1 for i in range(depth // 2)] + [2 for i in range(depth // 2)]
    norm_type = "simplermsnorm"
    use_softmax = True
    block_act = "relu"
    block_size = 4
    linear_act = "1+elu"
    model = NormViT(
        patch_size=patch_size, 
        dim=dim, 
        depth=depth, 
        heads=heads, 
        mlp_dim=mlp_dim,
        r=r,
        use_urpe=use_urpe,
        type_list=type_list,
        use_softmax=use_softmax,
        block_act=block_act,
        block_size=block_size,
        linear_act=linear_act,
        norm_type=norm_type,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model

@register_model
def norm_vit_tiny_patch16_224_mix_relu_elu_rmsnorm(pretrained=False, **kwargs):
    patch_size = 16
    dim = 192
    depth = 12
    heads = 6
    mlp_dim = 4 * dim
    dropout = 0.0
    r = 224 // patch_size
    use_urpe = True
    type_list = [1 for i in range(depth // 2)] + [2 for i in range(depth // 2)]
    norm_type = "simplermsnorm"
    use_softmax = False
    block_act = "relu"
    block_size = 4
    linear_act = "elu"
    model = NormViT(
        patch_size=patch_size, 
        dim=dim, 
        depth=depth, 
        heads=heads, 
        mlp_dim=mlp_dim,
        r=r,
        use_urpe=use_urpe,
        type_list=type_list,
        use_softmax=use_softmax,
        block_act=block_act,
        block_size=block_size,
        linear_act=linear_act,
        norm_type=norm_type,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model
##### base

##### no urpe
@register_model
def norm_vit_tiny_patch16_224_mix_softmax_1_elu_rmsnorm_no_urpe(pretrained=False, **kwargs):
    patch_size = 16
    dim = 192
    depth = 12
    heads = 6
    mlp_dim = 4 * dim
    dropout = 0.0
    r = 224 // patch_size
    use_urpe = False
    type_list = [1 for i in range(depth // 2)] + [2 for i in range(depth // 2)]
    norm_type = "simplermsnorm"
    use_softmax = True
    block_act = "relu"
    block_size = 4
    linear_act = "1+elu"
    model = NormViT(
        patch_size=patch_size, 
        dim=dim, 
        depth=depth, 
        heads=heads, 
        mlp_dim=mlp_dim,
        r=r,
        use_urpe=use_urpe,
        type_list=type_list,
        use_softmax=use_softmax,
        block_act=block_act,
        block_size=block_size,
        linear_act=linear_act,
        norm_type=norm_type,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model

@register_model
def norm_vit_tiny_patch16_224_mix_relu_elu_rmsnorm_no_urpe(pretrained=False, **kwargs):
    patch_size = 16
    dim = 192
    depth = 12
    heads = 6
    mlp_dim = 4 * dim
    dropout = 0.0
    r = 224 // patch_size
    use_urpe = False
    type_list = [1 for i in range(depth // 2)] + [2 for i in range(depth // 2)]
    norm_type = "simplermsnorm"
    use_softmax = False
    block_act = "relu"
    block_size = 4
    linear_act = "elu"
    model = NormViT(
        patch_size=patch_size, 
        dim=dim, 
        depth=depth, 
        heads=heads, 
        mlp_dim=mlp_dim,
        r=r,
        use_urpe=use_urpe,
        type_list=type_list,
        use_softmax=use_softmax,
        block_act=block_act,
        block_size=block_size,
        linear_act=linear_act,
        norm_type=norm_type,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model
##### no urpe

##### glu
@register_model
def norm_vit_tiny_patch16_224_mix_softmax_1_elu_rmsnorm_glu(pretrained=False, **kwargs):
    patch_size = 16
    dim = 192
    depth = 12
    heads = 6
    mlp_dim = 4 * dim
    dropout = 0.0
    r = 224 // patch_size
    use_urpe = True
    type_list = [1 for i in range(depth // 2)] + [2 for i in range(depth // 2)]
    norm_type = "simplermsnorm"
    use_softmax = True
    block_act = "relu"
    block_size = 4
    linear_act = "1+elu"
    # glu
    use_glu = True
    glu_act = "swish"
    glu_dim = (8 * dim // 3)
    model = NormViT(
        patch_size=patch_size, 
        dim=dim, 
        depth=depth, 
        heads=heads, 
        mlp_dim=mlp_dim,
        r=r,
        use_urpe=use_urpe,
        type_list=type_list,
        use_softmax=use_softmax,
        block_act=block_act,
        block_size=block_size,
        linear_act=linear_act,
        norm_type=norm_type,
        use_glu=use_glu,
        glu_act=glu_act,
        glu_dim=glu_dim,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model

@register_model
def norm_vit_tiny_patch16_224_mix_relu_elu_rmsnorm_glu(pretrained=False, **kwargs):
    patch_size = 16
    dim = 192
    depth = 12
    heads = 6
    mlp_dim = 4 * dim
    dropout = 0.0
    r = 224 // patch_size
    use_urpe = True
    type_list = [1 for i in range(depth // 2)] + [2 for i in range(depth // 2)]
    norm_type = "simplermsnorm"
    use_softmax = False
    block_act = "relu"
    block_size = 4
    linear_act = "elu"
    # glu
    use_glu = True
    glu_act = "swish"
    glu_dim = (8 * dim // 3)
    model = NormViT(
        patch_size=patch_size, 
        dim=dim, 
        depth=depth, 
        heads=heads, 
        mlp_dim=mlp_dim,
        r=r,
        use_urpe=use_urpe,
        type_list=type_list,
        use_softmax=use_softmax,
        block_act=block_act,
        block_size=block_size,
        linear_act=linear_act,
        norm_type=norm_type,
        use_glu=use_glu,
        glu_act=glu_act,
        glu_dim=glu_dim,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model
##### glu
########## Deit tiny

# to do: mask padding部分