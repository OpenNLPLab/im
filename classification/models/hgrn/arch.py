# Network architecture under test
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg

from .model_vit import *

@register_model
def hgrn1d_vit_tiny(pretrained=False, **kwargs):
    dim = 192
    glu_dim = 2 * dim 
    depth = 12
    model = Hgrn1dVit(
        patch_size=14, 
        embed_dim=dim, 
        depth=depth,
        prenorm=True,
        # glu
        glu_act="silu",
        glu_dim=glu_dim,
        param_share=False,
        **kwargs
    )
    model.default_cfg = _cfg()

    return model

@register_model
def hgrn1d_vit_small(pretrained=False, **kwargs):
    dim = 384
    glu_dim = 2 * dim 
    depth = 12
    model = Hgrn1dVit(
        patch_size=14, 
        embed_dim=dim, 
        depth=depth,
        prenorm=True,
        # glu
        glu_act="silu",
        glu_dim=glu_dim,
        param_share=False,
        **kwargs
    )
    model.default_cfg = _cfg()

    return model