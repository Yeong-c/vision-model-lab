import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, in_channels, filters, stride=1):
        super(BasicBlock, self).__init__()

        # 3 x 3 Conv
        self.conv1 = nn.Conv2d(in_channels, filters, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filters)

        # 3 x 3 Conv
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1, bias=False)
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

class ResNet(nn.Module):
    def __init__(self, Block=BasicBlock, num_blocks=[2,2,2,2]): # Default 18-layer
        super(ResNet, self).__init__()

        # 32x32 Start (32x32로 고정함)
        self.start = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.blocks = nn.ModuleList()

        # ResNet Block 붙이기
        # Basic Block
        # 18-layer: 2 2 2 2, 34-layer: 3 4 6 3
        if Block == BasicBlock:
            self.blocks.append(self.BasicBlocks(in_channels=64, filters=64, stride=1, num_blocks=num_blocks[0]))
            self.blocks.append(self.BasicBlocks(in_channels=64, filters=128, stride=2, num_blocks=num_blocks[1]))
            self.blocks.append(self.BasicBlocks(in_channels=128, filters=256, stride=2, num_blocks=num_blocks[2]))
            self.blocks.append(self.BasicBlocks(in_channels=256, filters=512, stride=2, num_blocks=num_blocks[3]))
            self.num_features = 512

        # Bottleneck Block
        # 50-layer: 3 4 6 3, 101-layer: 3 4 23 3, 152-layer: 3 8 36 3
        elif Block == BottleneckBlock:
            self.blocks.append(self.BottleneckBlocks(in_channels=64, filters=64, stride=1, num_blocks=num_blocks[0]))
            self.blocks.append(self.BottleneckBlocks(in_channels=256, filters=128, stride=2, num_blocks=num_blocks[1]))
            self.blocks.append(self.BottleneckBlocks(in_channels=512, filters=256, stride=2, num_blocks=num_blocks[2]))
            self.blocks.append(self.BottleneckBlocks(in_channels=1024, filters=512, stride=2, num_blocks=num_blocks[3]))
            self.num_features = 2048

        # Global Average Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    # ResNet18용 Block - 3x3 2번 (Not Bottleneck)
    def BasicBlocks(self, in_channels, filters, stride, num_blocks):
        layers = []
        layers.append(BasicBlock(in_channels, filters,  stride=stride))

        for _ in range(1, num_blocks):
            layers.append(BasicBlock(filters, filters, stride=1))

        return nn.Sequential(*layers)

    # ResNet50용 Block - 1x1 / 3x3 / 1x1 (Bottleneck)
    def BottleneckBlocks(self, in_channels, filters, stride, num_blocks):
        layers = []
        layers.append(BottleneckBlock(in_channels, filters, stride))

        for _ in range(1, num_blocks):
            layers.append(BottleneckBlock(filters*4, filters, stride=1))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        # 나가는 Output은 Method에서
        # Model은 기본 Backbone으로

        # conv1 start
        x = self.start(x)

        # conv2_x ~ conv5_x
        for b in self.blocks:
            # conv?_x (기존의 block1, block2...)
            x = b(x)

        # avg pooling
        x = self.avgpool(x)
        # flatten(vector)
        x = torch.flatten(x, 1)

        return x

def resnet18():
    return ResNet(Block=BasicBlock, num_blocks=[2, 2, 2, 2])

def resnet34():
    return ResNet(Block=BasicBlock, num_blocks=[3, 4, 6, 3])

def resnet50():
    return ResNet(Block=BottleneckBlock, num_blocks=[3, 4, 6, 3])

def resnet101():
    return ResNet(Block=BottleneckBlock, num_blocks=[3, 4, 23, 3])

def resnet152():
    return ResNet(Block=BottleneckBlock, num_blocks=[3, 8, 36, 3])