# Network architecture under test
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg

from .model_pyr import *
from .model_vit import *


############### model_vit
########## Deit tiny
##### rpe layer test
@register_model
def tnn_2d_vit_tiny_rpe_v8_l1(pretrained=False, **kwargs):
    dim = 192
    glu_dim = dim
    rpe_dim = 32
    num_heads = 1
    depth = 12
    model = TNN2DVit(
        patch_size=16, 
        embed_dim=dim, 
        num_heads=num_heads, 
        rpe_embedding=rpe_dim,
        rpe_act="relu",
        glu_act="silu",
        glu_dim=glu_dim,
        expand_ratio=3,
        depth=depth, 
        use_pos=False,
        rpe_layers=1,
        **kwargs
    )
    model.default_cfg = _cfg()

    return model

@register_model
def tnn_2d_vit_tiny_rpe_v8_l2(pretrained=False, **kwargs):
    dim = 192
    glu_dim = dim
    rpe_dim = 32
    num_heads = 1
    depth = 12
    model = TNN2DVit(
        patch_size=16, 
        embed_dim=dim, 
        num_heads=num_heads, 
        rpe_embedding=rpe_dim,
        rpe_act="relu",
        glu_act="silu",
        glu_dim=glu_dim,
        expand_ratio=3,
        depth=depth, 
        use_pos=False,
        rpe_layers=2,
        **kwargs
    )
    model.default_cfg = _cfg()

    return model

@register_model
def tnn_2d_vit_tiny_rpe_v8_l3(pretrained=False, **kwargs):
    dim = 192
    glu_dim = dim
    rpe_dim = 32
    num_heads = 1
    depth = 12
    model = TNN2DVit(
        patch_size=16, 
        embed_dim=dim, 
        num_heads=num_heads, 
        rpe_embedding=rpe_dim,
        rpe_act="relu",
        glu_act="silu",
        glu_dim=glu_dim,
        expand_ratio=3,
        depth=depth, 
        use_pos=False,
        rpe_layers=3,
        **kwargs
    )
    model.default_cfg = _cfg()

    return model

@register_model
def tnn_2d_vit_tiny_rpe_v8_l4(pretrained=False, **kwargs):
    dim = 192
    glu_dim = dim
    rpe_dim = 32
    num_heads = 1
    depth = 12
    model = TNN2DVit(
        patch_size=16, 
        embed_dim=dim, 
        num_heads=num_heads, 
        rpe_embedding=rpe_dim,
        rpe_act="relu",
        glu_act="silu",
        glu_dim=glu_dim,
        expand_ratio=3,
        depth=depth, 
        use_pos=False,
        rpe_layers=4,
        **kwargs
    )
    model.default_cfg = _cfg()

    return model

@register_model
def tnn_2d_vit_tiny_rpe_v8_l5(pretrained=False, **kwargs):
    dim = 192
    glu_dim = dim
    rpe_dim = 32
    num_heads = 1
    depth = 12
    model = TNN2DVit(
        patch_size=16, 
        embed_dim=dim, 
        num_heads=num_heads, 
        rpe_embedding=rpe_dim,
        rpe_act="relu",
        glu_act="silu",
        glu_dim=glu_dim,
        expand_ratio=3,
        depth=depth, 
        use_pos=False,
        rpe_layers=5,
        **kwargs
    )
    model.default_cfg = _cfg()

    return model

@register_model
def tnn_2d_vit_tiny_rpe_v8_l6(pretrained=False, **kwargs):
    dim = 192
    glu_dim = dim
    rpe_dim = 32
    num_heads = 1
    depth = 12
    model = TNN2DVit(
        patch_size=16, 
        embed_dim=dim, 
        num_heads=num_heads, 
        rpe_embedding=rpe_dim,
        rpe_act="relu",
        glu_act="silu",
        glu_dim=glu_dim,
        expand_ratio=3,
        depth=depth, 
        use_pos=False,
        rpe_layers=6,
        **kwargs
    )
    model.default_cfg = _cfg()

    return model
##### rpe layer test

##### prenorm
@register_model
def tnn_2d_vit_tiny_rpe_v8_l1_prenorm(pretrained=False, **kwargs):
    dim = 192
    glu_dim = dim
    rpe_dim = 32
    num_heads = 1
    depth = 12
    prenorm = True
    model = TNN2DVit(
        patch_size=16, 
        embed_dim=dim, 
        num_heads=num_heads, 
        rpe_embedding=rpe_dim,
        rpe_act="relu",
        glu_act="silu",
        glu_dim=glu_dim,
        expand_ratio=3,
        depth=depth, 
        use_pos=False,
        rpe_layers=1,
        prenorm=prenorm,
        **kwargs
    )
    model.default_cfg = _cfg()

    return model
##### prenorm

##### patch embedding test
@register_model
def tnn_2d_vit_tiny_rpe_v8_l1_prenorm_tno_patch(pretrained=False, **kwargs):
    dim = 192
    glu_dim = dim
    rpe_dim = 32
    num_heads = 1
    depth = 12
    prenorm = True
    model = TNN2DVit(
        patch_size=16, 
        embed_dim=dim, 
        num_heads=num_heads, 
        rpe_embedding=rpe_dim,
        rpe_act="relu",
        glu_act="silu",
        glu_dim=glu_dim,
        expand_ratio=3,
        depth=depth, 
        use_pos=False,
        rpe_layers=1,
        prenorm=prenorm,
        use_tno_patch=True,
        **kwargs
    )
    model.default_cfg = _cfg()

    return model
##### patch embedding test

##### decay test
@register_model
def tnn_2d_vit_tiny_rpe_v8_l1_prenorm_99(pretrained=False, **kwargs):
    dim = 192
    glu_dim = dim
    rpe_dim = 32
    num_heads = 1
    depth = 12
    prenorm = True
    use_decay = True
    gamma = 0.99
    model = TNN2DVit(
        patch_size=16, 
        embed_dim=dim, 
        num_heads=num_heads, 
        rpe_embedding=rpe_dim,
        rpe_act="relu",
        glu_act="silu",
        glu_dim=glu_dim,
        expand_ratio=3,
        depth=depth, 
        use_pos=False,
        rpe_layers=1,
        prenorm=prenorm,
        use_decay=use_decay,
        gamma=gamma,
        **kwargs
    )
    model.default_cfg = _cfg()

    return model

@register_model
def tnn_2d_vit_tiny_rpe_v8_l1_prenorm_95(pretrained=False, **kwargs):
    dim = 192
    glu_dim = dim
    rpe_dim = 32
    num_heads = 1
    depth = 12
    prenorm = True
    use_decay = True
    gamma = 0.95
    model = TNN2DVit(
        patch_size=16, 
        embed_dim=dim, 
        num_heads=num_heads, 
        rpe_embedding=rpe_dim,
        rpe_act="relu",
        glu_act="silu",
        glu_dim=glu_dim,
        expand_ratio=3,
        depth=depth, 
        use_pos=False,
        rpe_layers=1,
        prenorm=prenorm,
        use_decay=use_decay,
        gamma=gamma,
        **kwargs
    )
    model.default_cfg = _cfg()

    return model

@register_model
def tnn_2d_vit_tiny_rpe_v8_l1_prenorm_90(pretrained=False, **kwargs):
    dim = 192
    glu_dim = dim
    rpe_dim = 32
    num_heads = 1
    depth = 12
    prenorm = True
    use_decay = True
    gamma = 0.90
    model = TNN2DVit(
        patch_size=16, 
        embed_dim=dim, 
        num_heads=num_heads, 
        rpe_embedding=rpe_dim,
        rpe_act="relu",
        glu_act="silu",
        glu_dim=glu_dim,
        expand_ratio=3,
        depth=depth, 
        use_pos=False,
        rpe_layers=1,
        prenorm=prenorm,
        use_decay=use_decay,
        gamma=gamma,
        **kwargs
    )
    model.default_cfg = _cfg()

    return model
##### decay test
########## Deit tiny

########## Deit small
@register_model
def tnn_2d_vit_small_rpe_v8_l1_prenorm(pretrained=False, **kwargs):
    dim = 384
    glu_dim = dim
    rpe_dim = 32
    num_heads = 1
    depth = 12
    prenorm = True
    model = TNN2DVit(
        patch_size=16, 
        embed_dim=dim, 
        num_heads=num_heads, 
        rpe_embedding=rpe_dim,
        rpe_act="relu",
        glu_act="silu",
        glu_dim=glu_dim,
        expand_ratio=3,
        depth=depth, 
        use_pos=False,
        rpe_layers=1,
        prenorm=prenorm,
        **kwargs
    )
    model.default_cfg = _cfg()

    return model

##### decay test
@register_model
def tnn_2d_vit_small_rpe_v8_l1_prenorm_99(pretrained=False, **kwargs):
    dim = 384
    glu_dim = dim
    rpe_dim = 32
    num_heads = 1
    depth = 12
    prenorm = True
    use_decay = True
    gamma = 0.99
    model = TNN2DVit(
        patch_size=16, 
        embed_dim=dim, 
        num_heads=num_heads, 
        rpe_embedding=rpe_dim,
        rpe_act="relu",
        glu_act="silu",
        glu_dim=glu_dim,
        expand_ratio=3,
        depth=depth, 
        use_pos=False,
        rpe_layers=1,
        prenorm=prenorm,
        use_decay=use_decay,
        gamma=gamma,
        **kwargs
    )
    model.default_cfg = _cfg()

    return model

@register_model
def tnn_2d_vit_small_rpe_v8_l1_prenorm_95(pretrained=False, **kwargs):
    dim = 384
    glu_dim = dim
    rpe_dim = 32
    num_heads = 1
    depth = 12
    prenorm = True
    use_decay = True
    gamma = 0.95
    model = TNN2DVit(
        patch_size=16, 
        embed_dim=dim, 
        num_heads=num_heads, 
        rpe_embedding=rpe_dim,
        rpe_act="relu",
        glu_act="silu",
        glu_dim=glu_dim,
        expand_ratio=3,
        depth=depth, 
        use_pos=False,
        rpe_layers=1,
        prenorm=prenorm,
        use_decay=use_decay,
        gamma=gamma,
        **kwargs
    )
    model.default_cfg = _cfg()

    return model

@register_model
def tnn_2d_vit_small_rpe_v8_l1_prenorm_90(pretrained=False, **kwargs):
    dim = 384
    glu_dim = dim
    rpe_dim = 32
    num_heads = 1
    depth = 12
    prenorm = True
    use_decay = True
    gamma = 0.90
    model = TNN2DVit(
        patch_size=16, 
        embed_dim=dim, 
        num_heads=num_heads, 
        rpe_embedding=rpe_dim,
        rpe_act="relu",
        glu_act="silu",
        glu_dim=glu_dim,
        expand_ratio=3,
        depth=depth, 
        use_pos=False,
        rpe_layers=1,
        prenorm=prenorm,
        use_decay=use_decay,
        gamma=gamma,
        **kwargs
    )
    model.default_cfg = _cfg()

    return model
##### decay test
########## Deit small
############### model_vit

############### model_pyr
########## Pyramid tiny
@register_model
def tnn_2d_pyr_tiny_rpe_v8_l1(pretrained=False, **kwargs):
    patch_size = 4
    dim = 48
    glu_dim = dim
    rpe_dim = 32
    num_heads = 1
    depth = 12
    model = TNN2DPyr(
        patch_size=patch_size, 
        embed_dim=dim, 
        num_heads=num_heads, 
        rpe_embedding=rpe_dim,
        rpe_act="relu",
        glu_act="silu",
        glu_dim=glu_dim,
        expand_ratio=3,
        depths=[2, 2, 4, 2], 
        use_pos=False,
        rpe_layers=1,
        **kwargs
    )
    model.default_cfg = _cfg()

    return model

@register_model
def tnn_2d_pyr_tiny_rpe_v8_l1_prenorm(pretrained=False, **kwargs):
    patch_size = 4
    dim = 48
    glu_dim = dim
    rpe_dim = 32
    num_heads = 1
    depth = 12
    prenorm = True
    model = TNN2DPyr(
        patch_size=patch_size, 
        embed_dim=dim, 
        num_heads=num_heads, 
        rpe_embedding=rpe_dim,
        rpe_act="relu",
        glu_act="silu",
        glu_dim=glu_dim,
        expand_ratio=3,
        depths=[2, 2, 4, 2], 
        use_pos=False,
        rpe_layers=1,
        prenorm=prenorm,
        **kwargs
    )
    model.default_cfg = _cfg()

    return model
########## Pyramid tiny
############### model_pyr
