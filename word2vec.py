from torch import nn
from net import SkipNet
from torch.optim import Adam
import torch
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as axes3d
from data import *
from utils import endswapstart

net = SkipNet()
opt = Adam(SkipNet().parameters())

net.load_state_dict(torch.load("module/skip.pth"))
opt.load_state_dict(torch.load("module/opt.pth"))
weight = net.sentences_emb_weight.requires_grad_(False)
weight[0] = 0.
# embedding = nn.Embedding.from_pretrained(weight)
# sentences = list_id
# list_embedding = embedding(torch.tensor(list_id))

x = weight[:, 0].detach().numpy()
y = weight[:, 1].detach().numpy()
z = weight[:, 2].detach().numpy()
d = np.sqrt(x ** 2 + y ** 2 + z ** 2)

plt.figure("3D")
ax3d = plt.gca(projection="3d")
ax3d.set_xlabel("x", fontsize=14)
ax3d.set_ylabel("y", fontsize=14)
ax3d.set_zlabel("z", fontsize=14)
ax3d.scatter(x, y, z, s=60, alpha=0.6, c=d, cmap="jet")

print(maps)
word_list = list(maps.keys())
word_list = endswapstart(word_list)
print(word_list)
for i in range(len(word_list)):
    ax3d.text(x[i], y[i], z[i], word_list[i])
plt.rcParams['font.sans-serif'] = 'KaiTi'
plt.rcParams['axes.unicode_minus'] = False
plt.show()
