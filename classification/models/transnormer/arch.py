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
    num_heads = 6
    mlp_dim = 4 * dim
    dropout = 0.0
    use_lrpe = True
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
        num_heads=num_heads, 
        mlp_dim=mlp_dim,
        use_lrpe=use_lrpe,
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
    num_heads = 6
    mlp_dim = 4 * dim
    dropout = 0.0
    use_lrpe = True
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
        num_heads=num_heads, 
        mlp_dim=mlp_dim,
        use_lrpe=use_lrpe,
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

##### no lrpe
@register_model
def norm_vit_tiny_patch16_224_mix_softmax_1_elu_rmsnorm_no_lrpe(pretrained=False, **kwargs):
    patch_size = 16
    dim = 192
    depth = 12
    num_heads = 6
    mlp_dim = 4 * dim
    dropout = 0.0
    use_lrpe = False
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
        num_heads=num_heads, 
        mlp_dim=mlp_dim,
        use_lrpe=use_lrpe,
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
def norm_vit_tiny_patch16_224_mix_relu_elu_rmsnorm_no_lrpe(pretrained=False, **kwargs):
    patch_size = 16
    dim = 192
    depth = 12
    num_heads = 6
    mlp_dim = 4 * dim
    dropout = 0.0
    use_lrpe = False
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
        num_heads=num_heads, 
        mlp_dim=mlp_dim,
        use_lrpe=use_lrpe,
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
##### no lrpe

##### glu
@register_model
def norm_vit_tiny_patch16_224_mix_softmax_1_elu_rmsnorm_glu(pretrained=False, **kwargs):
    patch_size = 16
    dim = 192
    depth = 12
    num_heads = 6
    mlp_dim = 4 * dim
    dropout = 0.0
    use_lrpe = True
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
        num_heads=num_heads, 
        mlp_dim=mlp_dim,
        use_lrpe=use_lrpe,
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
    num_heads = 6
    mlp_dim = 4 * dim
    dropout = 0.0
    use_lrpe = True
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
        num_heads=num_heads, 
        mlp_dim=mlp_dim,
        use_lrpe=use_lrpe,
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
    num_heads = 6
    mlp_dim = 4 * dim
    dropout = 0.0
    use_lrpe = True
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
        num_heads=num_heads, 
        mlp_dim=mlp_dim,
        use_lrpe=use_lrpe,
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
    num_heads = 6
    mlp_dim = 4 * dim
    dropout = 0.0
    use_lrpe = True
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
        num_heads=num_heads, 
        mlp_dim=mlp_dim,
        use_lrpe=use_lrpe,
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
    num_heads = 6
    mlp_dim = 4 * dim
    dropout = 0.0
    use_lrpe = True
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
    num_heads_list = [6 for i in range(depth // 2)] + [dim for i in range(depth // 2)]
    model = Vin(
        patch_size=patch_size, 
        dim=dim, 
        depth=depth, 
        num_heads=num_heads, 
        mlp_dim=mlp_dim,
        use_lrpe=use_lrpe,
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
        num_heads_list=num_heads_list,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model

@register_model
def norm_vit_tiny_patch16_224_mix_relu_elu_rmsnorm_glu_maxhead(pretrained=False, **kwargs):
    patch_size = 16
    dim = 192
    depth = 12
    num_heads = 6
    mlp_dim = 4 * dim
    dropout = 0.0
    use_lrpe = True
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
    num_heads_list = [6 for i in range(depth // 2)] + [dim for i in range(depth // 2)]
    model = Vin(
        patch_size=patch_size, 
        dim=dim, 
        depth=depth, 
        num_heads=num_heads, 
        mlp_dim=mlp_dim,
        use_lrpe=use_lrpe,
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
        num_heads_list=num_heads_list,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model
##### head

##### glu head test
@register_model
def norm_vit_tiny_patch16_224_mix_softmax_1_elu_rmsnorm_glu_h3(pretrained=False, **kwargs):
    patch_size = 16
    dim = 192
    depth = 12
    num_heads = 3
    mlp_dim = 4 * dim
    dropout = 0.0
    use_lrpe = True
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
        num_heads=num_heads, 
        mlp_dim=mlp_dim,
        use_lrpe=use_lrpe,
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
def norm_vit_tiny_patch16_224_mix_relu_elu_rmsnorm_glu_h3(pretrained=False, **kwargs):
    patch_size = 16
    dim = 192
    depth = 12
    num_heads = 3
    mlp_dim = 4 * dim
    dropout = 0.0
    use_lrpe = True
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
        num_heads=num_heads, 
        mlp_dim=mlp_dim,
        use_lrpe=use_lrpe,
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
def norm_vit_tiny_patch16_224_mix_softmax_1_elu_rmsnorm_glu_h1(pretrained=False, **kwargs):
    patch_size = 16
    dim = 192
    depth = 12
    num_heads = 1
    mlp_dim = 4 * dim
    dropout = 0.0
    use_lrpe = True
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
        num_heads=num_heads, 
        mlp_dim=mlp_dim,
        use_lrpe=use_lrpe,
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
def norm_vit_tiny_patch16_224_mix_relu_elu_rmsnorm_glu_h1(pretrained=False, **kwargs):
    patch_size = 16
    dim = 192
    depth = 12
    num_heads = 1
    mlp_dim = 4 * dim
    dropout = 0.0
    use_lrpe = True
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
        num_heads=num_heads, 
        mlp_dim=mlp_dim,
        use_lrpe=use_lrpe,
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
##### glu head test

##### patch 14
@register_model
def norm_vit_tiny_patch14_224_mix_softmax_1_elu_rmsnorm_glu(pretrained=False, **kwargs):
    patch_size = 14
    dim = 192
    depth = 12
    num_heads = 6
    mlp_dim = 4 * dim
    dropout = 0.0
    use_lrpe = True
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
        num_heads=num_heads, 
        mlp_dim=mlp_dim,
        use_lrpe=use_lrpe,
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
def norm_vit_tiny_patch14_224_mix_relu_elu_rmsnorm_glu(pretrained=False, **kwargs):
    patch_size = 14
    dim = 192
    depth = 12
    num_heads = 6
    mlp_dim = 4 * dim
    dropout = 0.0
    use_lrpe = True
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
        num_heads=num_heads, 
        mlp_dim=mlp_dim,
        use_lrpe=use_lrpe,
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
##### patch 14

##### patch 7
@register_model
def norm_vit_tiny_patch7_224_mix_softmax_1_elu_rmsnorm_glu(pretrained=False, **kwargs):
    patch_size = 7
    dim = 192
    depth = 12
    num_heads = 6
    mlp_dim = 4 * dim
    dropout = 0.0
    use_lrpe = True
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
        num_heads=num_heads, 
        mlp_dim=mlp_dim,
        use_lrpe=use_lrpe,
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
def norm_vit_tiny_patch7_224_mix_relu_elu_rmsnorm_glu(pretrained=False, **kwargs):
    patch_size = 7
    dim = 192
    depth = 12
    num_heads = 6
    mlp_dim = 4 * dim
    dropout = 0.0
    use_lrpe = True
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
        num_heads=num_heads, 
        mlp_dim=mlp_dim,
        use_lrpe=use_lrpe,
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
##### patch 7

##### no block
@register_model
def norm_vit_tiny_patch16_224_mix_softmax_1_elu_rmsnorm_glu_no_block(pretrained=False, **kwargs):
    patch_size = 16
    dim = 192
    depth = 12
    num_heads = 6
    mlp_dim = 4 * dim
    dropout = 0.0
    use_lrpe = True
    type_list = [1 for i in range(depth // 2)] + [2 for i in range(depth // 2)]
    norm_type = "simplermsnorm"
    use_softmax = True
    block_act = "relu"
    block_size = 224 // patch_size
    linear_act = "1+elu"
    # glu
    use_glu = True
    glu_act = "swish"
    glu_dim = (8 * dim // 3)
    model = Vin(
        patch_size=patch_size, 
        dim=dim, 
        depth=depth, 
        num_heads=num_heads, 
        mlp_dim=mlp_dim,
        use_lrpe=use_lrpe,
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
def norm_vit_tiny_patch16_224_mix_relu_elu_rmsnorm_glu_no_block(pretrained=False, **kwargs):
    patch_size = 16
    dim = 192
    depth = 12
    num_heads = 6
    mlp_dim = 4 * dim
    dropout = 0.0
    use_lrpe = True
    type_list = [1 for i in range(depth // 2)] + [2 for i in range(depth // 2)]
    norm_type = "simplermsnorm"
    use_softmax = False
    block_act = "relu"
    block_size = 224 // patch_size
    linear_act = "elu"
    # glu
    use_glu = True
    glu_act = "swish"
    glu_dim = (8 * dim // 3)
    model = Vin(
        patch_size=patch_size, 
        dim=dim, 
        depth=depth, 
        num_heads=num_heads, 
        mlp_dim=mlp_dim,
        use_lrpe=use_lrpe,
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
##### no block

##### overlap patch
@register_model
def norm_vit_tiny_overlap_patch16_224_mix_softmax_1_elu_rmsnorm_glu(pretrained=False, **kwargs):
    patch_size = 16
    stride = 14
    dim = 192
    depth = 12
    num_heads = 6
    mlp_dim = 4 * dim
    dropout = 0.0
    use_lrpe = True
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
    # patch type
    patch_type = "overlap"
    model = Vin(
        patch_size=patch_size, 
        dim=dim, 
        depth=depth, 
        num_heads=num_heads, 
        mlp_dim=mlp_dim,
        use_lrpe=use_lrpe,
        type_list=type_list,
        use_softmax=use_softmax,
        block_act=block_act,
        block_size=block_size,
        linear_act=linear_act,
        norm_type=norm_type,
        use_glu=use_glu,
        glu_act=glu_act,
        glu_dim=glu_dim,
        stride=stride,
        patch_type=patch_type,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model

@register_model
def norm_vit_tiny_overlap_patch16_224_mix_relu_elu_rmsnorm_glu(pretrained=False, **kwargs):
    patch_size = 16
    stride = 14
    dim = 192
    depth = 12
    num_heads = 6
    mlp_dim = 4 * dim
    dropout = 0.0
    use_lrpe = True
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
    # patch type
    patch_type = "overlap"
    model = Vin(
        patch_size=patch_size, 
        dim=dim, 
        depth=depth, 
        num_heads=num_heads, 
        mlp_dim=mlp_dim,
        use_lrpe=use_lrpe,
        type_list=type_list,
        use_softmax=use_softmax,
        block_act=block_act,
        block_size=block_size,
        linear_act=linear_act,
        norm_type=norm_type,
        use_glu=use_glu,
        glu_act=glu_act,
        glu_dim=glu_dim,
        stride=stride,
        patch_type=patch_type,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model
##### overlap patch
########## Deit tiny

########## Deit small
@register_model
def norm_vit_small_patch16_224_mix_softmax_1_elu_rmsnorm_glu_h12(pretrained=False, **kwargs):
    patch_size = 16
    dim = 384
    depth = 12
    num_heads = 12
    mlp_dim = 4 * dim
    dropout = 0.0
    use_lrpe = True
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
        num_heads=num_heads, 
        mlp_dim=mlp_dim,
        use_lrpe=use_lrpe,
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
def norm_vit_small_patch16_224_mix_relu_elu_rmsnorm_glu_h12(pretrained=False, **kwargs):
    patch_size = 16
    dim = 384
    depth = 12
    num_heads = 12
    mlp_dim = 4 * dim
    dropout = 0.0
    use_lrpe = True
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
        num_heads=num_heads, 
        mlp_dim=mlp_dim,
        use_lrpe=use_lrpe,
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
def norm_vit_small_patch16_224_mix_softmax_1_elu_rmsnorm_glu_h6(pretrained=False, **kwargs):
    patch_size = 16
    dim = 384
    depth = 12
    num_heads = 6
    mlp_dim = 4 * dim
    dropout = 0.0
    use_lrpe = True
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
        num_heads=num_heads, 
        mlp_dim=mlp_dim,
        use_lrpe=use_lrpe,
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
def norm_vit_small_patch16_224_mix_relu_elu_rmsnorm_glu_h6(pretrained=False, **kwargs):
    patch_size = 16
    dim = 384
    depth = 12
    num_heads = 6
    mlp_dim = 4 * dim
    dropout = 0.0
    use_lrpe = True
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
        num_heads=num_heads, 
        mlp_dim=mlp_dim,
        use_lrpe=use_lrpe,
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

##### patch 14
@register_model
def norm_vit_small_patch14_224_mix_softmax_1_elu_rmsnorm_glu_h6(pretrained=False, **kwargs):
    patch_size = 14
    dim = 384
    depth = 12
    num_heads = 6
    mlp_dim = 4 * dim
    dropout = 0.0
    use_lrpe = True
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
        num_heads=num_heads, 
        mlp_dim=mlp_dim,
        use_lrpe=use_lrpe,
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
def norm_vit_small_patch14_224_mix_relu_elu_rmsnorm_glu_h6(pretrained=False, **kwargs):
    patch_size = 14
    dim = 384
    depth = 12
    num_heads = 6
    mlp_dim = 4 * dim
    dropout = 0.0
    use_lrpe = True
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
        num_heads=num_heads, 
        mlp_dim=mlp_dim,
        use_lrpe=use_lrpe,
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
##### patch 14

##### no block
@register_model
def norm_vit_small_patch16_224_mix_softmax_1_elu_rmsnorm_glu_h12_no_block(pretrained=False, **kwargs):
    patch_size = 16
    dim = 384
    depth = 12
    num_heads = 12
    mlp_dim = 4 * dim
    dropout = 0.0
    use_lrpe = True
    type_list = [1 for i in range(depth // 2)] + [2 for i in range(depth // 2)]
    norm_type = "simplermsnorm"
    use_softmax = True
    block_act = "relu"
    block_size = 224 // patch_size
    linear_act = "1+elu"
    # glu
    use_glu = True
    glu_act = "swish"
    glu_dim = (8 * dim // 3)
    model = Vin(
        patch_size=patch_size, 
        dim=dim, 
        depth=depth, 
        num_heads=num_heads, 
        mlp_dim=mlp_dim,
        use_lrpe=use_lrpe,
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
def norm_vit_small_patch16_224_mix_relu_elu_rmsnorm_glu_h12_no_block(pretrained=False, **kwargs):
    patch_size = 16
    dim = 384
    depth = 12
    num_heads = 12
    mlp_dim = 4 * dim
    dropout = 0.0
    use_lrpe = True
    type_list = [1 for i in range(depth // 2)] + [2 for i in range(depth // 2)]
    norm_type = "simplermsnorm"
    use_softmax = False
    block_act = "relu"
    block_size = 224 // patch_size
    linear_act = "elu"
    # glu
    use_glu = True
    glu_act = "swish"
    glu_dim = (8 * dim // 3)
    model = Vin(
        patch_size=patch_size, 
        dim=dim, 
        depth=depth, 
        num_heads=num_heads, 
        mlp_dim=mlp_dim,
        use_lrpe=use_lrpe,
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
##### no block

##### no block conv patch
@register_model
def norm_vit_small_conv_patch16_224_mix_softmax_1_elu_rmsnorm_glu_h12_no_block(pretrained=False, **kwargs):
    patch_size = 16
    dim = 384
    depth = 12
    num_heads = 12
    mlp_dim = 4 * dim
    dropout = 0.0
    use_lrpe = True
    type_list = [1 for i in range(depth // 2)] + [2 for i in range(depth // 2)]
    norm_type = "simplermsnorm"
    use_softmax = True
    block_act = "relu"
    block_size = 224 // patch_size
    linear_act = "1+elu"
    # glu
    use_glu = True
    glu_act = "swish"
    glu_dim = (8 * dim // 3)
    # patch type
    patch_type = "conv"
    model = Vin(
        patch_size=patch_size, 
        dim=dim, 
        depth=depth, 
        num_heads=num_heads, 
        mlp_dim=mlp_dim,
        use_lrpe=use_lrpe,
        type_list=type_list,
        use_softmax=use_softmax,
        block_act=block_act,
        block_size=block_size,
        linear_act=linear_act,
        norm_type=norm_type,
        use_glu=use_glu,
        glu_act=glu_act,
        glu_dim=glu_dim,
        patch_type=patch_type,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model

@register_model
def norm_vit_small_conv_patch16_224_mix_relu_elu_rmsnorm_glu_h12_no_block(pretrained=False, **kwargs):
    patch_size = 16
    dim = 384
    depth = 12
    num_heads = 12
    mlp_dim = 4 * dim
    dropout = 0.0
    use_lrpe = True
    type_list = [1 for i in range(depth // 2)] + [2 for i in range(depth // 2)]
    norm_type = "simplermsnorm"
    use_softmax = False
    block_act = "relu"
    block_size = 224 // patch_size
    linear_act = "elu"
    # glu
    use_glu = True
    glu_act = "swish"
    glu_dim = (8 * dim // 3)
    # patch type
    patch_type = "conv"
    model = Vin(
        patch_size=patch_size, 
        dim=dim, 
        depth=depth, 
        num_heads=num_heads, 
        mlp_dim=mlp_dim,
        use_lrpe=use_lrpe,
        type_list=type_list,
        use_softmax=use_softmax,
        block_act=block_act,
        block_size=block_size,
        linear_act=linear_act,
        norm_type=norm_type,
        use_glu=use_glu,
        glu_act=glu_act,
        glu_dim=glu_dim,
        patch_type=patch_type,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model
##### no block conv patch
########## Deit small
############### model_vit
