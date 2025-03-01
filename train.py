import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import copy

from LeNetwork.LeNet5 import LeNet5
from LeNetwork.Net_model import Model_Net

# 超参数设置
BATCH_SIZE = 128
EPOCHS = 200
LEARNING_RATE = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据增强与预处理
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],  # CIFAR-100专用统计值
                         std=[0.2675, 0.2565, 0.2761]),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                         std=[0.2675, 0.2565, 0.2761]),
])

# 加载数据集
train_dataset = torchvision.datasets.CIFAR100(
    root='../data/',
    train=True,
    download=True,
    transform=train_transform
)

test_dataset = torchvision.datasets.CIFAR100(
    root='../data/',
    train=False,
    download=True,
    transform=test_transform
)



train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# 初始化模型
model = Model_Net(3,100).to(DEVICE)

# 定义优化器和学习率调度
optimizer = optim.SGD(model.parameters(),
                      lr=LEARNING_RATE,
                      momentum=MOMENTUM,
                      weight_decay=WEIGHT_DECAY)
# optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)

scheduler = MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
criterion = nn.CrossEntropyLoss()

# 训练准备
best_acc = 0.0
best_model_wts = copy.deepcopy(model.state_dict())


# 训练函数
def train(epoch):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    scheduler.step()
    print(f'Epoch {epoch + 1} Loss: {running_loss / len(train_loader):.4f}')


# 测试函数（返回准确率）
def test():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100.0 * correct / total
    print(f'Test Accuracy: {acc:.2f}%')
    return acc


# 主训练循环
for epoch in range(EPOCHS):
    train(epoch)
    current_acc = test()

    # 保存最佳模型
    if current_acc > best_acc:
        best_acc = current_acc
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(best_model_wts, f'best_model.pth')
        print(f'New best model saved! Accuracy: {best_acc:.2f}%')

# 加载最佳模型进行最终测试
print('-' * 50)
print('Loading best model for final evaluation...')
model.load_state_dict(torch.load('best_model.pth'))
final_acc = test()
print(f'Final Test Accuracy: {final_acc:.2f}%')