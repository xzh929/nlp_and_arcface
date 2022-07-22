from torch import nn
import torch


class SkipNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.sentences_emb_weight = nn.Parameter(torch.randn(125, 3))
        self.layer = nn.Sequential(
            nn.Linear(3, 6),
            nn.Hardswish(),
            nn.Linear(6, 4 * 3),
            nn.Softmax(dim=1)
        )

    def forward(self, x, y):
        x = self.sentences_emb_weight[x]
        label = self.sentences_emb_weight[y]
        out = self.layer(x)
        out = out.reshape(-1, 4, 3)
        return out, label




if __name__ == '__main__':
    a = torch.randint(10, (2,))
    print(a)
    b = torch.randint(10, (2, 4))
    net = SkipNet()
    out, y = net(a, b)
    print(out, y)
