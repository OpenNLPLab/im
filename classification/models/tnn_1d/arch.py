# Network architecture under test
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from .model_vit import *

############### model_vit
########## Deit tiny
##### Base test
@register_model
def tnn_vit_e3g1_tiny_rpe_l1_95(pretrained=False, **kwargs):
    dim = 192
    glu_dim = dim
    rpe_dim = 48
    num_heads = 1
    depth = 12
    max_l = 224 // 16
    model = TNNVit(
        patch_size=16, 
        embed_dim=dim, 
        num_heads=num_heads, 
        rpe_embdding=rpe_dim,
        rpe_act="relu",
        glu_act="silu",
        glu_dim=glu_dim,
        expand_ratio=3,
        depth=depth, 
        use_pos=False,
        rpe_layers=1,
        use_decay=True,
        gamma=0.95,
        **kwargs
    )
    model.default_cfg = _cfg()

    return model

@register_model
def tnn_vit_e3g1_tiny_rpe_l1_95_prenorm(pretrained=False, **kwargs):
    dim = 192
    glu_dim = dim
    rpe_dim = 48
    num_heads = 1
    depth = 12
    max_l = 224 // 16
    model = TNNVit(
        patch_size=16, 
        embed_dim=dim, 
        num_heads=num_heads, 
        rpe_embdding=rpe_dim,
        rpe_act="relu",
        glu_act="silu",
        glu_dim=glu_dim,
        expand_ratio=3,
        depth=depth, 
        use_pos=False,
        rpe_layers=1,
        use_decay=True,
        gamma=0.95,
        prenorm=True,
        **kwargs
    )
    model.default_cfg = _cfg()

    return model

@register_model
def tnn_vit_e3g1_tiny_rpe_l1_90_prenorm(pretrained=False, **kwargs):
    dim = 192
    glu_dim = dim
    rpe_dim = 48
    num_heads = 1
    depth = 12
    max_l = 224 // 16
    model = TNNVit(
        patch_size=16, 
        embed_dim=dim, 
        num_heads=num_heads, 
        rpe_embdding=rpe_dim,
        rpe_act="relu",
        glu_act="silu",
        glu_dim=glu_dim,
        expand_ratio=3,
        depth=depth, 
        use_pos=False,
        rpe_layers=1,
        use_decay=True,
        gamma=0.9,
        prenorm=True,
        **kwargs
    )
    model.default_cfg = _cfg()

    return model
##### Base test
########## Deit tiny

########## Deit small
@register_model
def tno_vit_e3g1_small_rpe_l1_90_prenorm(pretrained=False, **kwargs):
    dim = 384
    glu_dim = dim
    rpe_dim = 48
    num_heads = 1
    depth = 12
    max_l = 224 // 16
    model = TNNVit(
        patch_size=16, 
        embed_dim=dim, 
        num_heads=num_heads, 
        rpe_embdding=rpe_dim,
        rpe_act="relu",
        glu_act="silu",
        glu_dim=glu_dim,
        expand_ratio=3,
        depth=depth, 
        use_pos=False,
        rpe_layers=1,
        use_decay=True,
        gamma=0.9,
        prenorm=True,
        **kwargs
    )
    model.default_cfg = _cfg()

    return model

@register_model
def tno_vit_e3g1_small_rpe_l1_95_prenorm(pretrained=False, **kwargs):
    dim = 384
    glu_dim = dim
    rpe_dim = 48
    num_heads = 1
    depth = 12
    max_l = 224 // 16
    model = TNNVit(
        patch_size=16, 
        embed_dim=dim, 
        num_heads=num_heads, 
        rpe_embdding=rpe_dim,
        rpe_act="relu",
        glu_act="silu",
        glu_dim=glu_dim,
        expand_ratio=3,
        depth=depth, 
        use_pos=False,
        rpe_layers=1,
        use_decay=True,
        gamma=0.95,
        prenorm=True,
        **kwargs
    )
    model.default_cfg = _cfg()

    return model
########## Deit small
############### model_vit