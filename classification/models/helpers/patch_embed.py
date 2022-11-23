import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange

from .helpers import get_world_size
from .normlization import SimpleRMSNorm


def compute_resulotion(size, padding, kernel_size, stride, dilation=1):
    return int((size + (padding) * 2 - dilation * (kernel_size - 1) - 1) / stride) + 1

def get_batch_norm(dim):
    if get_world_size() == 1:
        return nn.BatchNorm2d(dim)
    else:
        return nn.SyncBatchNorm(dim)

class OverlapPatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, stride=14, in_chans=3, dim=192, use_2d=False):
        super().__init__()
        self.use_2d = use_2d
        # compute num_patches
        padding = patch_size // 2
        self.num_row_patches = compute_resulotion(
            size=img_size, 
            padding=padding, 
            kernel_size=patch_size, 
            stride=stride,
        )
        self.proj = nn.Conv2d(in_chans, dim, kernel_size=patch_size, stride=stride,
                              padding=(padding, padding))
        
    def forward(self, x):
        x = self.proj(x)
        if self.use_2d:
            x = rearrange(x, 'b e h w -> b h w e')
        else:
            x = rearrange(x, 'b e h w -> b (h w) e')

        return x

def get_patch_embedding(
    patch_type, 
    dim,
    # mlp
    patch_dim,
    # overlap
    patch_size,
    stride,
    # conv
    conv_kernel_size=3, 
    conv_stride=2, 
    conv_padding=1,
    # basic params
    channels=3,
    img_size=224,
    # 2d: if true, use (h w c), else use ((h w) c)
    use_2d=False,
):
    if patch_type == "overlap":
        to_patch_embedding = OverlapPatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            stride=stride,
            in_chans=channels,
            dim=dim,
            use_2d=use_2d,
        )
        num_row_patches = to_patch_embedding.num_row_patches
    elif patch_type == "conv":
        to_patch_embedding = nn.Sequential(
            nn.Conv2d(channels, dim // 8, conv_kernel_size, conv_stride, conv_padding),
            get_batch_norm(dim // 8),
            nn.GELU(),
            nn.Conv2d(dim // 8, dim // 4, conv_kernel_size, conv_stride, conv_padding),
            get_batch_norm(dim // 4),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim // 2, conv_kernel_size, conv_stride, conv_padding),
            get_batch_norm(dim // 2),
            nn.GELU(),
            nn.Conv2d(dim // 2, dim, conv_kernel_size, conv_stride, conv_padding),
            get_batch_norm(dim),
            Rearrange('b c h w -> b h w c') if use_2d else Rearrange('b c h w -> b (h w) c'),
        )
        num_row_patches = img_size
        for i in range(4):
            num_row_patches = compute_resulotion(
                num_row_patches, 
                padding=conv_padding, 
                kernel_size=conv_kernel_size, 
                stride=conv_stride,
            )
    elif patch_type == "stem":
        to_patch_embedding = nn.Sequential(
            nn.Conv2d(channels, dim // 2, conv_kernel_size, conv_stride, conv_padding),
            SimpleRMSNorm(dim // 2),
            nn.GELU(),
            nn.Conv2d(dim // 2, dim, conv_kernel_size, conv_stride, conv_padding),
            SimpleRMSNorm(dim),
            Rearrange('b c h w -> b h w c') if use_2d else Rearrange('b c h w -> b (h w) c'),
        )
        num_row_patches = img_size
        for i in range(2):
            num_row_patches = compute_resulotion(
                num_row_patches, 
                padding=conv_padding, 
                kernel_size=conv_kernel_size, 
                stride=conv_stride,
            )
    else:
        to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b h w (p1 p2 c)', p1=patch_size, p2=patch_size) if use_2d else Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, dim),
        )
        num_row_patches = img_size // patch_size
    num_patches = num_row_patches * num_row_patches
        
    return to_patch_embedding, num_patches, num_row_patches
