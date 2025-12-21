import torch
import torch.nn as nn
import torch.optim as optim

class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, filters, stride=1):
        super(BottleneckBlock, self).__init__()

        # 1x1 Conv
        self.conv1 = nn.Conv2d(in_channels, filters, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filters)

        # 3x3 Conv
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(filters)

        # 1x1 Conv
        self.conv3 = nn.Conv2d(filters, filters * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(filters * 4)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != filters * 4:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, filters * 4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(filters * 4)
            )

    def forward(self, x):
        # Conv1
        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.functional.relu(out)

        # Conv2
        out = self.conv2(out)
        out = self.bn2(out)
        out = nn.functional.relu(out)

        # Conv3
        out = self.conv3(out)
        out = self.bn3(out)

        # Shortcut Connection
        out += self.shortcut(x)

        # ReLU
        out = nn.functional.relu(out)
        return out

class ResNet50(nn.Module):
    # CIFAR-10 용으로
    def __init__(self, input_shape=(3, 32, 32), num_classes=10):
        super(ResNet50, self).__init__()

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
        self.block1 = self.block(in_channels=64, filters=64, stride=1, num_blocks=3)

        # Conv3_x
        self.block2 = self.block(in_channels=256, filters=128, stride=2, num_blocks=4)

        # Conv4_x
        self.block3 = self.block(in_channels=512, filters=256, stride=2, num_blocks=6)

        # Conv5_x
        self.block4 = self.block(in_channels=1024, filters=512, stride=2, num_blocks=3)

        # Last Layer
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(2048, num_classes)

        # Loss Function
        self.criterion = nn.CrossEntropyLoss()
        # Optimizer
        self.optimizer = optim.SGD(self.parameters(),
                              lr=0.1,
                              momentum=0.9,
                              weight_decay=5e-4
                              )
        # Scheduler
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=[100,150],
            gamma=0.1
        )

    def block(self, in_channels, filters, stride, num_blocks):
        layers = []
        layers.append(BottleneckBlock(in_channels, filters, stride))

        for _ in range(1, num_blocks):
            layers.append(BottleneckBlock(filters*4, filters, stride=1))

        return nn.Sequential(*layers)

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
