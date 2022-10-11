import torch

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