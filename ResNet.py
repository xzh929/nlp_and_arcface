from torch import nn
import torch
from torch.nn import functional as F
import math

# 下采样层
class downsample(nn.Module):
    def __init__(self, c_in, c_out):
        super(downsample, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(c_in, c_out, (1, 1), (2, 2), bias=False),
            nn.BatchNorm2d(c_out)
        )

    def forward(self, x):
        return self.down(x)


# 残差块
class BasicBlock(nn.Module):
    def __init__(self, c_in, c_out, is_first=True):
        super(BasicBlock, self).__init__()
        self.stride = (2, 2) if is_first else (1, 1)
        self.is_first = is_first
        if is_first:
            self.basic = nn.Sequential(
                nn.Conv2d(c_in, c_out, (3, 3), self.stride, padding=(1, 1), bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(),
                nn.Conv2d(c_out, c_out, (3, 3), (1, 1), padding=(1, 1), bias=False),
                nn.BatchNorm2d(c_out),
            )
            self.down = downsample(c_in, c_out)
        else:
            self.basic = nn.Sequential(
                nn.Conv2d(c_in, c_out, (3, 3), self.stride, padding=(1, 1), bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(),
                nn.Conv2d(c_out, c_out, (3, 3), (1, 1), padding=(1, 1), bias=False),
                nn.BatchNorm2d(c_out)
            )

    # 第一次输入和输出形状不一，用下采样做一次转换
    def forward(self, x):
        if self.is_first:
            return self.basic(x) + self.down(x)
        else:
            return self.basic(x) + x


class Arcloss(nn.Module):
    def __init__(self, input_dim, output_dim, m):
        super().__init__()
        self._w = nn.Parameter(torch.randn(input_dim, output_dim))
        self._m = m
        self.cls_num = output_dim

    def forward(self, f):
        # t = F.one_hot(t, self.cls_num)
        f = F.normalize(f, dim=1)
        w = F.normalize(self._w, dim=0)
        s = torch.sqrt(torch.sum(torch.pow(f, 2))) * torch.sqrt(torch.sum(torch.pow(w, 2)))
        cosa = torch.matmul(f, self._w) / s
        angle = torch.acos(cosa)

        arcsoftmax = torch.exp(
            s * torch.cos(angle + self._m)) / (torch.sum(torch.exp(s * cosa), dim=1, keepdim=True) - torch.exp(
            s * cosa) + torch.exp(s * torch.cos(angle + self._m)))

        return arcsoftmax

class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin

            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output

class FocalLoss(nn.Module):

    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(1, 64, (7, 7), (2, 2), padding=(3, 3), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, padding=1),
            BasicBlock(64, 64, False),
            BasicBlock(64, 64, False),
            BasicBlock(64, 128, True),
            BasicBlock(128, 128, False),
            BasicBlock(128, 256, True),
            BasicBlock(256, 256, False),
            BasicBlock(256, 512, True),
            BasicBlock(512, 512, False),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.out_layer = nn.Sequential(
            nn.Linear(512, 512)
        )

    def forward(self, x):
        cnn_out = self.layer(x)
        cnn_out = cnn_out.reshape(-1, 512)
        feature = self.out_layer(cnn_out)
        return feature

class ResNet34(nn.Module):
    def __init__(self):
        super(ResNet34, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(1, 64, (7, 7), (2, 2), padding=(3, 3), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, padding=1),
            BasicBlock(64, 64, False),
            BasicBlock(64, 64, False),
            BasicBlock(64, 64, False),
            BasicBlock(64, 128, True),
            BasicBlock(128, 128, False),
            BasicBlock(128, 128, False),
            BasicBlock(128, 128, False),
            BasicBlock(128, 256, True),
            BasicBlock(256, 256, False),
            BasicBlock(256, 256, False),
            BasicBlock(256, 256, False),
            BasicBlock(256, 256, False),
            BasicBlock(256, 256, False),
            BasicBlock(256, 512, True),
            BasicBlock(512, 512, False),
            BasicBlock(512, 512, False),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.feature = nn.Linear(512, 2)
        self.arcloss = Arcloss(2, 10, 0.5)
        self.out_layer = nn.Sequential(
            nn.Linear(512, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        cnn_out = self.layer(x)
        cnn_out = cnn_out.reshape(-1, 512)
        feature = self.feature(cnn_out)
        arc_loss = torch.log(self.arcloss(feature))
        cls_out = self.out_layer(cnn_out)
        return arc_loss, feature


if __name__ == '__main__':
    a = torch.randn(5, 1, 28, 28)
    tag = torch.tensor([1, 1, 0, 1, 0])
    # net = BasicBlock(64,64,True).forward(a)
    net2 = ResNet18()
    f = net2(a)
    # print(net2)
    # print(net1(a).shape)
    print(f.shape)
