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
from models.helpers import get_activation_fn, get_norm

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class PreNorm(nn.Module):
    def __init__(self, dim, fn, norm_type):
        super().__init__()
        self.norm = get_norm(norm_type, dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

#### no cls
class BlockAttention(nn.Module):
    def __init__(
        self, 
        dim, 
        heads=8, 
        dim_head=64, 
        dropout=0., 
        r=7, # 每行的patch数量
        use_urpe=False,
        use_softmax=True,
        norm_type="layernorm",
        block_size=4,
        act_fun="relu",
    ):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.atten = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        self.r = r

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.use_urpe = use_urpe
        print(f"self.use_urpe {self.use_urpe}")
        if self.use_urpe:
            self.urpe = Urpe(core_matrix=1, p_matrix=3, embedding_dim=dim_head, theta_learned=True, dims=[2, 3])
        self.use_softmax = use_softmax
        print(f"self.use_softmax {self.use_softmax}")
        if not self.use_softmax:
            self.norm = get_norm(norm_type, inner_dim)
        self.block_size = block_size
        print(f"self.block_size {self.block_size}")
        print(f"act_fun {act_fun}")
        self.act_fun = get_activation_fn(act_fun)

    def transform(self, x, r, c):
        # reshape
        x = rearrange(x, 'b h (r c) d -> b h r c d', r=self.r)
        # pad
        r_pad = (self.block_size - r % self.block_size) % self.block_size
        c_pad = (self.block_size - c % self.block_size) % self.block_size
        x = F.pad(x, (0, 0, 0, r_pad, 0, c_pad, 0, 0, 0, 0))
        # reshape
        x = rearrange(x, 'b h (n g) (m e) d -> b h n g m e d', g=self.block_size, e=self.block_size)
        x = rearrange(x, 'b h n g m e d -> b h n m (g e) d')

        return x

    def reverse_transform(self, x, r, c):
        x = rearrange(x, 'b h n m (g e) d -> b h n g m e d', g=self.block_size, e=self.block_size)
        x = rearrange(x, 'b h n g m e d -> b h (n g) (m e) d')
        # remove pad
        x = x[:, :, :r, :c, :]
        x = rearrange(x, 'b h r c d -> b h (r c) d')

        return x

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        if self.use_urpe:
            q = rearrange(q, 'b h (r c) d -> b h r c d', r=self.r)
            k = rearrange(k, 'b h (r c) d -> b h r c d', r=self.r)
            q = self.urpe(q)
            k = self.urpe(k)
            q = rearrange(q, 'b h r c d -> b h (r c) d')
            k = rearrange(k, 'b h r c d -> b h (r c) d')

        r = self.r
        c = q.shape[-2] // r
        # chunk
        # b h n m (g g) d
        q = self.transform(q, r, c)
        k = self.transform(k, r, c)
        v = self.transform(v, r, c)
        
        dots = torch.einsum('...nd,...md->...nm', q, k) * self.scale
        if self.use_softmax:
            attn = self.atten(dots)
        else:
            attn = self.act_fun(dots)
        attn = self.dropout(attn)

        out = torch.einsum("...ls,...sd->...ld", attn, v)
        out = self.reverse_transform(out, r, c)
        out = rearrange(out, 'b h n d -> b n (h d)')
        if not self.use_softmax:
            out = self.norm(out)
        return self.to_out(out)

#### no cls
class NormLinearAttention(nn.Module):
    def __init__(
        self, 
        dim, 
        heads=8, 
        dim_head=64, 
        dropout=0., 
        r=7, # 每行的patch数量
        use_urpe=False,
        norm_type="layernorm",
        act_fun="relu",
    ):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.dropout = nn.Dropout(dropout)
        self.r = r

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.use_urpe = use_urpe
        print(f"self.use_urpe {self.use_urpe}")
        if self.use_urpe:
            self.urpe = Urpe(core_matrix=1, p_matrix=3, embedding_dim=dim_head, theta_learned=True, dims=[2, 3])
        self.norm = get_norm(norm_type, inner_dim)
        print(f"act_fun {act_fun}")
        self.act_fun = get_activation_fn(act_fun)

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        q = self.act_fun(q)
        k = self.act_fun(k)
        if self.use_urpe:
            q = rearrange(q, 'b h (r c) d -> b h r c d', r=self.r)
            k = rearrange(k, 'b h (r c) d -> b h r c d', r=self.r)
            q = self.urpe(q)
            k = self.urpe(k)
            q = rearrange(q, 'b h r c d -> b h (r c) d')
            k = rearrange(k, 'b h r c d -> b h (r c) d')

        kv = torch.einsum('...nm,...nd->...md', k, v)
        qkv = torch.einsum('...nm,...md->...nd', q, kv)
        out = rearrange(qkv, 'b h n d -> b n (h d)')
        out = self.norm(out)

        return self.to_out(out)

class Transnormer(nn.Module):
    def __init__(
        self, 
        dim, 
        depth, 
        heads, 
        dim_head, 
        mlp_dim, 
        dropout = 0., 
        r=7, 
        use_urpe=False, 
        # add
        type_list=[],
        norm_type="simplermsnorm",
        use_softmax=True,
        block_act="relu",
        block_size=4,
        linear_act="1+elu",
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
            self.layers.append(nn.ModuleList([
                    PreNorm(dim, Attention, norm_type),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout), norm_type)
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
            return BlockAttention(
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

#### no cls
class NormViT(nn.Module):
    def __init__(
        self, *, image_size=224, patch_size, num_classes, dim, depth, heads, mlp_dim, 
        pool='cls', channels=3, drop_rate=0., 
        emb_dropout = 0., drop_path_rate=0.,
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
########## Deit tiny

# to do: mask padding部分