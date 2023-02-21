# Network architecture under test
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg

from .model_vit import *

############### model_vit
########## Deit tiny
##### Base test
@register_model
def dnn_vit_tiny_rpe_l1_90_glu_2_4_3(pretrained=False, **kwargs):
    dim = 192
    expand_ratio = 2
    use_glu = True
    glu_dim = 4 * dim // 3
    depth = 12
    model = DTNVit(
        patch_size=16, 
        embed_dim=dim, 
        depth=depth,
        # dtu
        expand_ratio=expand_ratio,
        dtu_act="silu",
        prenorm=True,
        norm_type="layernorm",
        # glu
        use_glu=use_glu,
        glu_act="silu",
        glu_dim=glu_dim,
        # rpe
        rpe_in_dim=32,
        rpe_out_dim=16,
        rpe_layers=1,
        # decay
        use_decay=True,
        lambda_=0.90,
        **kwargs
    )
    model.default_cfg = _cfg()

    return model

@register_model
def dnn_vit_tiny_rpe_l1_90_e3(pretrained=False, **kwargs):
    dim = 192
    expand_ratio = 3
    depth = 12
    model = DTNVit(
        patch_size=16, 
        embed_dim=dim, 
        depth=depth,
        # dtu
        expand_ratio=expand_ratio,
        dtu_act="silu",
        prenorm=True,
        norm_type="layernorm",
        # rpe
        rpe_in_dim=32,
        rpe_out_dim=16,
        rpe_layers=1,
        # decay
        use_decay=True,
        lambda_=0.90,
        **kwargs
    )
    model.default_cfg = _cfg()

    return model

@register_model
def dnn_vit_tiny_rpe_l1_90_e2(pretrained=False, **kwargs):
    dim = 192
    expand_ratio = 2
    depth = 12 * 3 // 2
    model = DTNVit(
        patch_size=16, 
        embed_dim=dim, 
        depth=depth,
        # dtu
        expand_ratio=expand_ratio,
        dtu_act="silu",
        prenorm=True,
        norm_type="layernorm",
        # rpe
        rpe_in_dim=32,
        rpe_out_dim=16,
        rpe_layers=1,
        # decay
        use_decay=True,
        lambda_=0.90,
        **kwargs
    )
    model.default_cfg = _cfg()

    return model

@register_model
def dnn_vit_tiny_rpe_l1_90_e1(pretrained=False, **kwargs):
    dim = 192
    expand_ratio = 1
    depth = 12 * 3
    model = DTNVit(
        patch_size=16, 
        embed_dim=dim, 
        depth=depth,
        # dtu
        expand_ratio=expand_ratio,
        dtu_act="silu",
        prenorm=True,
        norm_type="layernorm",
        # rpe
        rpe_in_dim=32,
        rpe_out_dim=16,
        rpe_layers=1,
        # decay
        use_decay=True,
        lambda_=0.90,
        **kwargs
    )
    model.default_cfg = _cfg()

    return model
##### Base test

##### No decay
@register_model
def dnn_vit_tiny_rpe_l1_90_glu_2_4_3_no_decay(pretrained=False, **kwargs):
    dim = 192
    expand_ratio = 2
    use_glu = True
    glu_dim = 4 * dim // 3
    depth = 12
    model = DTNVit(
        patch_size=16, 
        embed_dim=dim, 
        depth=depth,
        # dtu
        expand_ratio=expand_ratio,
        dtu_act="silu",
        prenorm=True,
        norm_type="layernorm",
        # glu
        use_glu=use_glu,
        glu_act="silu",
        glu_dim=glu_dim,
        # rpe
        rpe_in_dim=32,
        rpe_out_dim=16,
        rpe_layers=1,
        # decay
        use_decay=False,
        lambda_=0.90,
        **kwargs
    )
    model.default_cfg = _cfg()

    return model

@register_model
def dnn_vit_tiny_rpe_l1_90_glu_2_4_3_no_decay(pretrained=False, **kwargs):
    dim = 192
    expand_ratio = 2
    use_glu = True
    glu_dim = 4 * dim // 3
    depth = 12
    model = DTNVit(
        patch_size=16, 
        embed_dim=dim, 
        depth=depth,
        # dtu
        expand_ratio=expand_ratio,
        dtu_act="silu",
        prenorm=True,
        norm_type="layernorm",
        # glu
        use_glu=use_glu,
        glu_act="silu",
        glu_dim=glu_dim,
        # rpe
        rpe_in_dim=32,
        rpe_out_dim=16,
        rpe_layers=1,
        # decay
        use_decay=False,
        lambda_=0.90,
        **kwargs
    )
    model.default_cfg = _cfg()

    return model

@register_model
def dnn_vit_tiny_rpe_l1_90_e3_no_decay(pretrained=False, **kwargs):
    dim = 192
    expand_ratio = 3
    depth = 12
    model = DTNVit(
        patch_size=16, 
        embed_dim=dim, 
        depth=depth,
        # dtu
        expand_ratio=expand_ratio,
        dtu_act="silu",
        prenorm=True,
        norm_type="layernorm",
        # rpe
        rpe_in_dim=32,
        rpe_out_dim=16,
        rpe_layers=1,
        # decay
        use_decay=False,
        lambda_=0.90,
        **kwargs
    )
    model.default_cfg = _cfg()

    return model

@register_model
def dnn_vit_tiny_rpe_l1_90_e2_no_decay(pretrained=False, **kwargs):
    dim = 192
    expand_ratio = 2
    depth = 12 * 3 // 2
    model = DTNVit(
        patch_size=16, 
        embed_dim=dim, 
        depth=depth,
        # dtu
        expand_ratio=expand_ratio,
        dtu_act="silu",
        prenorm=True,
        norm_type="layernorm",
        # rpe
        rpe_in_dim=32,
        rpe_out_dim=16,
        rpe_layers=1,
        # decay
        use_decay=False,
        lambda_=0.90,
        **kwargs
    )
    model.default_cfg = _cfg()

    return model

@register_model
def dnn_vit_tiny_rpe_l1_90_e1_no_decay(pretrained=False, **kwargs):
    dim = 192
    expand_ratio = 1
    depth = 12 * 3
    model = DTNVit(
        patch_size=16, 
        embed_dim=dim, 
        depth=depth,
        # dtu
        expand_ratio=expand_ratio,
        dtu_act="silu",
        prenorm=True,
        norm_type="layernorm",
        # rpe
        rpe_in_dim=32,
        rpe_out_dim=16,
        rpe_layers=1,
        # decay
        use_decay=False,
        lambda_=0.90,
        **kwargs
    )
    model.default_cfg = _cfg()

    return model
##### No decay
########## Deit tiny
############### model_vit