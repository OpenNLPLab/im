# Network architecture under test
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg

from .model_vit import Vin

# to do: mask padding部分

############### model_vit
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
    model = Vin(
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
    model = Vin(
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
    model = Vin(
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
    model = Vin(
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
    model = Vin(
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
    model = Vin(
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

##### no pe
@register_model
def norm_vit_tiny_patch16_224_mix_softmax_1_elu_rmsnorm_glu_nopos(pretrained=False, **kwargs):
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
    model = Vin(
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
        use_pos=False,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model

@register_model
def norm_vit_tiny_patch16_224_mix_relu_elu_rmsnorm_glu_nopos(pretrained=False, **kwargs):
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
    model = Vin(
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
        use_pos=False,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model
##### no pe

##### head
@register_model
def norm_vit_tiny_patch16_224_mix_softmax_1_elu_rmsnorm_glu_maxhead(pretrained=False, **kwargs):
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
    # head
    headslist = [6 for i in range(depth // 2)] + [dim for i in range(depth // 2)]
    model = Vin(
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
        use_pos=False,
        headslist=headslist,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model

@register_model
def norm_vit_tiny_patch16_224_mix_relu_elu_rmsnorm_glu_maxhead(pretrained=False, **kwargs):
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
    # head
    headslist = [6 for i in range(depth // 2)] + [dim for i in range(depth // 2)]
    model = Vin(
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
        use_pos=False,
        headslist=headslist,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model
##### head

##### glu standard head
@register_model
def norm_vit_tiny_patch16_224_mix_softmax_1_elu_rmsnorm_glu_standard(pretrained=False, **kwargs):
    patch_size = 16
    dim = 192
    depth = 12
    heads = 3
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
    model = Vin(
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
def norm_vit_tiny_patch16_224_mix_relu_elu_rmsnorm_glu_standard(pretrained=False, **kwargs):
    patch_size = 16
    dim = 192
    depth = 12
    heads = 3
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
    model = Vin(
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
##### glu glu standard head
########## Deit tiny

########## Deit small
@register_model
def norm_vit_small_patch16_224_mix_softmax_1_elu_rmsnorm_glu(pretrained=False, **kwargs):
    patch_size = 16
    dim = 384
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
    model = Vin(
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
def norm_vit_small_patch16_224_mix_relu_elu_rmsnorm_glu(pretrained=False, **kwargs):
    patch_size = 16
    dim = 384
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
    model = Vin(
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
########## Deit small
############### model_vit