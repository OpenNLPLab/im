# Network architecture under test
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg

from .model_vit import Vit

##### Deit small
@register_model
def linear_vit_small(pretrained=False, **kwargs):
    patch_size = 16
    dim = 384
    depth = 12
    num_heads = 6
    mlp_dim = 4 * dim
    dropout = 0.0
    act_fun = "1+elu"
    model = Vit(
        patch_size=patch_size, 
        dim=dim, 
        depth=depth, 
        num_heads=num_heads, 
        mlp_dim=mlp_dim,
        act_fun=act_fun,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model

##### lrpe
# rotate learnable
@register_model
def linear_vit_small_l_ro(pretrained=False, **kwargs):
    patch_size = 16
    dim = 384
    depth = 12
    num_heads = 6
    mlp_dim = 4 * dim
    dropout = 0.0
    act_fun = "1+elu"
    # lrpe
    use_lrpe = True
    core_matrix = 1
    p_matrix = 3
    theta_learned = True
    model = Vit(
        patch_size=patch_size, 
        dim=dim, 
        depth=depth, 
        num_heads=num_heads, 
        mlp_dim=mlp_dim,
        act_fun=act_fun,
        # lrpe
        use_lrpe=use_lrpe,
        core_matrix=core_matrix,
        p_matrix=p_matrix,
        theta_learned=theta_learned,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model

# permutate
@register_model
def linear_vit_small_l_per(pretrained=False, **kwargs):
    patch_size = 16
    dim = 384
    depth = 12
    num_heads = 6
    mlp_dim = 4 * dim
    dropout = 0.0
    act_fun = "1+elu"
    # lrpe
    use_lrpe = True
    core_matrix = 3
    p_matrix = 3
    model = Vit(
        patch_size=patch_size, 
        dim=dim, 
        depth=depth, 
        num_heads=num_heads, 
        mlp_dim=mlp_dim,
        act_fun=act_fun,
        # lrpe
        use_lrpe=use_lrpe,
        core_matrix=core_matrix,
        p_matrix=p_matrix,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model

# unitary
@register_model
def linear_vit_small_l_un(pretrained=False, **kwargs):
    patch_size = 16
    dim = 384
    depth = 12
    num_heads = 6
    mlp_dim = 4 * dim
    dropout = 0.0
    act_fun = "1+elu"
    # lrpe
    use_lrpe = True
    core_matrix = 4
    p_matrix = 3
    model = Vit(
        patch_size=patch_size, 
        dim=dim, 
        depth=depth, 
        num_heads=num_heads, 
        mlp_dim=mlp_dim,
        act_fun=act_fun,
        # lrpe
        use_lrpe=use_lrpe,
        core_matrix=core_matrix,
        p_matrix=p_matrix,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model

# unitary learned
@register_model
def linear_vit_small_l_unl(pretrained=False, **kwargs):
    patch_size = 16
    dim = 384
    depth = 12
    num_heads = 6
    mlp_dim = 4 * dim
    dropout = 0.0
    act_fun = "1+elu"
    # lrpe
    use_lrpe = True
    core_matrix = 4
    p_matrix = 3
    theta_learned = True
    model = Vit(
        patch_size=patch_size, 
        dim=dim, 
        depth=depth, 
        num_heads=num_heads, 
        mlp_dim=mlp_dim,
        act_fun=act_fun,
        # lrpe
        use_lrpe=use_lrpe,
        core_matrix=core_matrix,
        p_matrix=p_matrix,
        theta_learned=theta_learned,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model
##### lrpe

##### rope
@register_model
def linear_vit_small_rope(pretrained=False, **kwargs):
    patch_size = 16
    dim = 384
    depth = 12
    num_heads = 6
    mlp_dim = 4 * dim
    dropout = 0.0
    act_fun = "1+elu"
    # rope
    use_rope = True
    model = Vit(
        patch_size=patch_size, 
        dim=dim, 
        depth=depth, 
        num_heads=num_heads, 
        mlp_dim=mlp_dim,
        act_fun=act_fun,
        # rope
        use_rope=use_rope,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model
##### rope

##### permutate
@register_model
def linear_vit_small_per(pretrained=False, **kwargs):
    patch_size = 16
    dim = 384
    depth = 12
    num_heads = 6
    mlp_dim = 4 * dim
    dropout = 0.0
    act_fun = "1+elu"
    # per
    use_permutate = True
    model = Vit(
        patch_size=patch_size, 
        dim=dim, 
        depth=depth, 
        num_heads=num_heads, 
        mlp_dim=mlp_dim,
        act_fun=act_fun,
        # permutate
        use_permutate=use_permutate,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model
##### permutate

##### spe
@register_model
def linear_vit_small_per(pretrained=False, **kwargs):
    patch_size = 16
    dim = 384
    depth = 12
    num_heads = 6
    mlp_dim = 4 * dim
    dropout = 0.0
    act_fun = "1+elu"
    # spe
    use_spe = True
    model = Vit(
        patch_size=patch_size, 
        dim=dim, 
        depth=depth, 
        num_heads=num_heads, 
        mlp_dim=mlp_dim,
        act_fun=act_fun,
        # spe
        use_spe=use_spe,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model
##### spe