import torch
import torch.nn as nn

from models.tnn_1d import Tno
from models.tnn_2d import Tno2D

b = 2
h = 8
n = 4
m = 5
e = 4
d = 16

print("======Start Test Tno=====")
x = torch.rand(b, h, n, m, e)
models = [
    Tno(h, e, d, use_decay=True),
    Tno(h, e, d, use_multi_decay=True),
    Tno(h, e, d, causal=True),
]

for dim in [-2, -3]:
    for model in models:
        y1 = model.forward(x, dim=dim)
        y2 = model.toeplizt_matrix(x, dim=dim)
        print(torch.norm(y1 - y2))
print("======End Test Tno=====")

print("======Start Test Tno2D=====")
x = torch.rand(b, h, n, m, e)
models = [
    Tno2D(h, n, m, e, d, use_decay=True),
    Tno2D(h, n, m, e, d, use_multi_decay=True),
    Tno2D(h, n, m, e, d, causal=True),
]

for dim in [-2, -3]:
    for model in models:
        y1 = model.forward(x)
        y2 = model.block_toeplizt_matrix(x)
        print(torch.norm(y1 - y2))
print("======End Test Tno2D=====")

print("======Start Test conv patch embedding=====")
image = torch.rand(1, 3, 224, 224)
dim = 192
to_patch_embedding = nn.Sequential(
    nn.Conv2d(3, dim // 8, 3, 2, 1),
    nn.BatchNorm2d(dim // 8),
    nn.GELU(),
    nn.Conv2d(dim // 8, dim // 4, 3, 2, 1),
    nn.BatchNorm2d(dim // 4),
    nn.GELU(),
    nn.Conv2d(dim // 4, dim // 2, 3, 2, 1),
    nn.BatchNorm2d(dim // 2),
    nn.GELU(),
    nn.Conv2d(dim // 2, dim, 3, 2, 1),
    nn.BatchNorm2d(dim),
)
res = to_patch_embedding(image)
print(res.shape)
print("======Start Test conv patch embedding=====")
