import torch
from tno_2d import Tno2D

# multi head
h = 8
b = 1
# n = 16
# m = 16
n = 3
m = 3
n = 14
m = 14
n = 3
m = 3
# n = 20
# m = 20
e = 4
d = 16
# x = torch.rand(b, h, 2 * n, 2 * m, e)
x = torch.rand(b, h, n, m, e)
##### no exp
# model = Tno2D(h, n, m, e, d)
# y1 = model.forward(x)
# y2 = model.block_toeplizt_matrix(x)
# print(y1.shape, y2.shape)
# print(torch.norm(y1 - y2))
model = Tno2D(h, n, m, e, d, use_decay=True)
y1 = model.forward(x)
y2 = model.block_toeplizt_matrix(x)
print(torch.norm(y1 - y2))

# model = Tno2D(h, n, m, e, d, use_multi_decay=True)
# y1 = model.forward(x)
# y2 = model.block_toeplizt_matrix(x)
# print(torch.norm(y1 - y2))

# model = Tno2D(h, 100, e, d, causal=True, use_pad=True)
# y = model.forward(x, dim=-2)
# model = Tno2D(h, 100, e, d, use_multi_decay=True)
# y = model.forward(x, dim=-2)
# model = Tno2D(h, 100, e, d, use_pad=True)
# y = model.forward(x, dim=-2)
# ##### exp
# model = Tno2D(h, 100, e, d, causal=True, use_exp=True, act_type="relu2")
# y = model.forward(x, dim=-2)
# model = Tno2D(h, 100, e, d, causal=True, use_exp=True, use_pad=True)
# y = model.forward(x, dim=-2)
# model = Tno2D(h, 100, e, d, causal=True, use_exp=True, use_pad=True, use_neg_exp=True)
# y = model.forward(x, dim=-2)
# model = Tno2D(h, 100, e, d, use_exp=True)
# y = model.forward(x, dim=-2)
# model = Tno2D(h, 100, e, d, use_exp=True, use_pad=True)
# y = model.forward(x, dim=-2)
# model = Tno2D(h, 100, e, d, use_exp=True, use_pad=True, use_neg_exp=True)
# y = model.forward(x, dim=-2)
# model = Tno2D(h, 100, e, d, use_exp=True, use_pad=True, use_neg_exp=True, use_multi_decay=True)
# y = model.forward(x, dim=-2)
# model = Tno2D(h, 100, e, d, use_exp=True, use_pad=True, use_neg_exp=True, use_decay=True)
# y = model.forward(x, dim=-2)