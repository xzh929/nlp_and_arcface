from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from ResNet import ResNet18,ArcMarginProduct,FocalLoss
from torch import optim
from torch import nn
import torch
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

module_path = r"module/mnist.pth"


def datavision(feature, label, epoch):
    plt.ion()
    feature_data = feature.numpy()
    tag_data = label.numpy()
    color = ['#006400', '#2F4F4F', '#808080', '#191970', '#473C8B', '#D2691E', '#8B3626',
             '#000000', '#008B45', '#0000FF', ]
    x = feature_data[:, 0]
    y = feature_data[:, 1]
    plt.title("epoch={}".format(epoch))
    for i in range(10):
        plt.plot(x[tag_data == i], y[tag_data == i], ".", c=color[i])
    plt.savefig(r"pltimages/epoch{}.jpg".format(epoch))
    plt.draw()
    plt.pause(0.001)
    plt.clf()


class Trainer:
    def __init__(self):
        self.dataset = datasets.MNIST(root=r"D:\data", transform=transforms.ToTensor())
        self.train_loader = DataLoader(self.dataset, batch_size=512, shuffle=True)
        self.net = ResNet18().cuda()
        self.loss_fun = FocalLoss(gamma=2)
        self.arc_margin = ArcMarginProduct(512, 10, s=30, m=0.5, easy_margin=False).cuda()
        self.opt = optim.Adam(self.net.parameters())

    def __call__(self):
        init_acc = 0.
        for epoch in range(100000):
            sum_train_loss = 0.
            sum_train_acc = 0.
            feature_data = []
            tag_data = []
            for i, (data, tag) in enumerate(self.train_loader):
                data, tag = data.cuda(), tag.cuda()
                feature = self.net(data)
                arc = self.arc_margin(feature, tag)
                loss = self.loss_fun(arc, tag)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                train_acc = torch.mean(torch.eq(torch.argmax(arc, dim=1), tag).float())
                sum_train_loss += loss.item()
                sum_train_acc += train_acc

                feature_data.append(feature.detach().cpu())
                tag_data.append(tag.detach().cpu())

            feature_data = torch.cat(feature_data)
            tag_data = torch.cat(tag_data)
            avg_train_loss = sum_train_loss / len(self.train_loader)
            avg_train_acc = sum_train_acc / len(self.train_loader)
            print("epoch:{} train_loss:{} train_acc:{}".format(epoch, avg_train_loss, avg_train_acc))
            if avg_train_acc > init_acc:
                init_acc = avg_train_acc
                torch.save(self.net.state_dict(), module_path)
                print("save")
            datavision(feature_data, tag_data, epoch)


class Tester:
    def __init__(self):
        self.test_loader = DataLoader(datasets.MNIST(root=r"D:\data", train=False, transform=transforms.ToTensor()),
                                      batch_size=128, shuffle=True)
        self.net = ResNet34().cuda()
        self.loss_fun = nn.CrossEntropyLoss()
        self.opt = optim.Adam(self.net.parameters())
        if os.path.exists(module_path):
            self.net.load_state_dict(torch.load(module_path))
            print("load")

    def __call__(self):
        sum_test_loss = 0.
        sum_test_acc = 0.
        self.net.eval()
        feature_data = []
        tag_data = []
        for i, (data, tag) in enumerate(self.test_loader):
            data, tag = data.cuda(), tag.cuda()
            out, feature = self.net(data)
            loss = self.loss_fun(out, tag)
            test_acc = torch.mean(torch.eq(torch.argmax(out, dim=1), tag).float())
            sum_test_loss += loss.item()
            sum_test_acc += test_acc
            feature_data.append(feature.cpu().detach())
            tag_data.append(tag.cpu().detach())

        feature_data = torch.cat(feature_data)
        tag_data = torch.cat(tag_data)
        avg_test_loss = sum_test_loss / len(self.test_loader)
        avg_test_acc = sum_test_acc / len(self.test_loader)
        print("test_loss:{} test_acc:{}".format(avg_test_loss, avg_test_acc))


if __name__ == '__main__':
    trainer = Trainer()
    trainer()
    # tester = Tester()
    # tester()
