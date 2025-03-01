import os
import numpy as np
import torch
from torch import nn
from torch import optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST,CIFAR10,CIFAR100
import torchvision.transforms as transforms

from LeNetwork.Net_model import Model_Net

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # 卷积层网络构建
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # 全连接层网络构建
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120,84),
            nn.ReLU(),
            nn.Linear(84,10)
        )

    def forward(self, x):
        # print(x.size())
        x = self.features(x)
        x = torch.flatten(x,1)
        res = self.classifier(x)
        return res

def train():
    # 加载数据集，直接从pytorch中拉数据
    train_dataset = MNIST(root='../data/', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = MNIST(root='../data/', train=False, transform=transforms.ToTensor())
    # model = LeNet5()
    model = LeNet5()
    # 将模型放入GPU中
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    batch_size,epoch = 32,100
    train_loader = DataLoader(train_dataset,batch_size,shuffle=True)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=model.parameters(),lr=0.01)
    acc_list = []
    for i in range(epoch):
        correct = 0  # 每一个epoch最后一组batch用来求acc，correct记录正确数
        loss_temp = 0  # 记录每一个epoch的总损失值
        for j,(data_x,data_y) in enumerate(train_loader):
            data_x, data_y = data_x.cuda(), data_y.cuda()
            optimizer.zero_grad()
            predict = model(data_x)
            loss = loss_function(predict, data_y)
            loss_temp += loss.item()
            loss.backward()
            optimizer.step()
            if j==batch_size-1:
                predicted = torch.max(predict.data, 1)[1]
                # 获取准确个数
                correct += (predicted == data_y).sum()
                acc = (100 * correct / batch_size).item()
                acc_list.append((100 * correct / batch_size).item())
        print('[%d] loss: %.4f' % (i + 1, loss_temp / len(train_loader)),'acc: ',acc)
    print(acc_list)
    plot_acc(acc_list)

def plot_acc(acc_list):
    plt.title("LeNet-acc")
    plt.ylim(0, 110)
    x = np.arange(len(acc_list))
    plt.plot(x, acc_list)
    plt.show()

if __name__ == '__main__':
    train()