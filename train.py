import torch
from data import Sendataset
from net import SkipNet
from torch.utils.data import DataLoader
from torch import optim
from torch import nn
from torch.utils.tensorboard import SummaryWriter

module_path = "module/skip.pth"
opt_path = "module/opt.pth"

train_loader = DataLoader(Sendataset(), batch_size=128, shuffle=True)
net = SkipNet().cuda()
loss_fun = nn.MSELoss()
opt = optim.Adam(net.parameters())
summary = SummaryWriter("logs")

for epoch in range(1000):
    sum_loss = 0.
    for i, (data, label) in enumerate(train_loader):
        data, label = data.cuda(), label.cuda()
        out, label = net(data, label)
        loss = loss_fun(out, label)

        opt.zero_grad()
        loss.backward()
        opt.step()

        sum_loss += loss.item()

    avg_loss = sum_loss / len(train_loader)
    summary.add_scalar("loss", avg_loss, epoch)
    print("epoch:{} loss:{}".format(epoch, avg_loss))
    torch.save(net.state_dict(), module_path)
    torch.save(opt.state_dict(), opt_path)
