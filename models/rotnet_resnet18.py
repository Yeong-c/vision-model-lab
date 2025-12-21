import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

transform = transforms.Compose(
    [transforms.ToTensor(), #이미지를 텐서로
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) #정규화, 입력값 - 0.5 / 0.5

"""
lr 수정
"""
batch_size = 128
"""
lr 수정
"""

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('0','90','180','270')


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, filters, kernel_size, stride=1):
        super(ResidualBlock, self).__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv2d(in_channels, filters, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(filters)

        self.conv2 = nn.Conv2d(filters, filters, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(filters)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != filters:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, filters, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(filters)
            )

    def forward(self, x):
        # Conv1
        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.functional.relu(out)

        # Conv2
        out = self.conv2(out)
        out = self.bn2(out)

        # Shortcut Connection
        out += self.shortcut(x)

        # ReLU
        out = nn.functional.relu(out)
        return out

class ResNet18(nn.Module):
    # CIFAR-10 용으로
    def __init__(self, input_shape=(3, 32, 32), num_classes=10):
        super(ResNet18, self).__init__()

        in_channels = input_shape[0]

        # ImageNet이 아니라 CIFAR-10이라 7x7 Stride 2, Max Pooling 대신에
        # 3x3 Stride 1, No Pooling으로

        # Conv1: 3x3 Stride 1 (원래는 7x7 Stride 2)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # Conv2_1: Max Pooling
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual Blocks
        # Conv2_x
        self.block1 = self.block(in_channels=64, filters=64, stride=1)

        # Conv3_x
        self.block2 = self.block(in_channels=64, filters=128, stride=2)

        # Conv4_x
        self.block3 = self.block(in_channels=128, filters=256, stride=2)

        # Conv5_x
        self.block4 = self.block(in_channels=256, filters=512, stride=2)

        # Last Layer
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, num_classes)

    def block(self, in_channels, filters, stride):
        return nn.Sequential(
            ResidualBlock(in_channels, filters, kernel_size=3, stride=stride),
            ResidualBlock(filters, filters, kernel_size=3, stride=1)
        )

    def forward(self, x):
        # Conv1
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.functional.relu(x)
        # x = self.maxpool(x) CIFAR-10

        # Residual Blocks
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        # Global Average Pooling -> Dense
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
    
def rotate_batch(images, device):

    batch_size = images.size(0)
    #dims = 배치, 채널, 높 , 넙
    x0 = images
    y0 = torch.zeros(batch_size, dtype=torch.long, device=device)

    x90 = torch.rot90(images, k=1, dims=[2,3])
    y90 = torch.ones(batch_size, dtype=torch.long, device=device)

    x180 = torch.rot90(images, k=2, dims=[2,3])
    y180 = torch.full((batch_size,), 2, dtype=torch.long, device=device)

    x270 = torch.rot90(images, k=3, dims=[2,3])
    y270 = torch.full((batch_size,), 3, dtype=torch.long, device=device)

    imgs = torch.cat([x0,x90,x180,x270], dim=0)
    labs = torch.cat([y0,y90,y180,y270], dim=0)

    return imgs, labs

class RotNet(nn.Module):
    def __init__(self, input_shape=(3,32,32)):
        super(RotNet,self).__init__()
        self.backbone = ResNet18(input_shape = input_shape, num_classes=4)

    def forward(self,x):
        return self.backbone(x)


#net = Net().to(device) # 글카 사용
net = RotNet().to(device)
criterion = nn.CrossEntropyLoss() #nn에서 loss함수 선택
"""
lr 수정
"""
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True) #논문 설정 #경사하강법으로 옵티마이저 설정, 모멘텀은 관성
"""
lr 수정
"""
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,60,80], gamma=0.2) # 300 에폭 기준

for epoch in range(100): 

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs = data[0].to(device)
        inputs, labels = rotate_batch(inputs, device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs) # 입력에 대한 통과
        loss = criterion(outputs, labels) # 로스 
        loss.backward() # 역전파
        optimizer.step() # 개선 (이경우 sgd)

        # print statistics
        running_loss += loss.item()

        """
        lr 수정
        """

        if i % 10 == 9:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.3f}')
            running_loss = 0.0

        """
        lr 수정
        """

    scheduler.step()    

print('Finished Training')


# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in testloader:
        inputs = data[0].to(device)
        images, labels = rotate_batch(inputs, device)
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

