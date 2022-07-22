import torch
from utils import getvaluedim2
from torch.nn.functional import one_hot

a = torch.tensor([[1, 0], [1, 0], [0, 1], [0, 1], [1, 0]])
b = torch.randn(5, 2)
c = torch.randn(3, 3)
d = torch.tensor([0, 0, 1, 1, 0])
idx = torch.nonzero(a)
# print(b)
# print(idx)
# idx = torch.nonzero(a)
# print(b)
# print(b[a == 1])
# print(b @ c)
# print(torch.matmul(b,c))
# d = torch.cat([b,c])
# # print(d.shape)
# print(b)
# for i,idx in enumerate(a):
#     print(b[i,idx])
# print(b[a==1])
print(one_hot(d))