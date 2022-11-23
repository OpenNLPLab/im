import torch
from einops import rearrange
from torch import nn

from .normlization import SimpleRMSNorm


# ref: https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
class DownSample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = SimpleRMSNorm(4 * dim)

    def forward(self, x):
        """
        x: B, H, W, C
        """
        B, H, W, C = x.shape
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
        H1 = H // 2
        W1 = W // 2
        
        x = rearrange(x, 'b (k H) W d -> b H W (k d)', k=2, H=H1)
        x = rearrange(x, 'b H (k W) d -> b H W (k d)', k=2, W=W1)
        x = self.norm(x)
        x = self.reduction(x)

        return x

# use idea from https://github.com/OpenGVLab/InternImage
class DownSampleInternImage(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Conv2d(dim, 2 * dim, kernel_size=3, stride=2, padding=1)
        self.norm = SimpleRMSNorm(2 * dim)

    def forward(self, x):
        """
        x: B, H, W, C
        """
        x = rearrange(x, 'b h w c -> b c h w')
        x = self.reduction(x)
        x = self.norm(x)
        x = rearrange(x, 'b c h w -> b h w c')

        return x
