import torch
import torch.nn as nn

class NormalBlock(nn.Module):
    def __init__(self, in_channels, filters, stride=1):
        super(NormalBlock, self).__init__()

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
    def __init__(self, num_layers=50, input_shape=(3, 32, 32)):
        super(ResNet, self).__init__()

        # 입력 차원 관리
        # 1. 작으면 Channel만 64로 늘려서 Conv 후 Block으로
        # 2. 크면 7x7 Conv, 3x3 Pooling으로 사이즈 줄이면서 Channel 64로 늘리고 Block으로
        if input_shape[1] <= 32:
            self.start = nn.Sequential(
                nn.Conv2d(input_shape[0], 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )
        elif input_shape[1] <= 64:
            self.start = nn.Sequential(
                nn.Conv2d(input_shape[0], 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )
        else:
            self.start = nn.Sequential(
                nn.Conv2d(input_shape[0], 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )

        # num_layers: 18-layer, 50-layer
        # 18-LAYER
        if num_layers == 18:
            self.block1 = self.block18(in_channels=64, filters=64, stride=1)
            self.block2 = self.block18(in_channels=64, filters=128, stride=2)
            self.block3 = self.block18(in_channels=128, filters=256, stride=2)
            self.block4 = self.block18(in_channels=256, filters=512, stride=2)
            # 결과물 Shape
            self.num_features = 512
        # 50-LAYER
        elif num_layers == 50:
            self.block1 = self.block50(in_channels=64, filters=64, stride=1, num_blocks=3)
            self.block2 = self.block50(in_channels=256, filters=128, stride=2, num_blocks=4)
            self.block3 = self.block50(in_channels=512, filters=256, stride=2, num_blocks=6)
            self.block4 = self.block50(in_channels=1024, filters=512, stride=2, num_blocks=3)
            # 결과물 Shape
            self.num_features = 2048
        # DEFAULT: 50-LAYER
        else:
            self.block1 = self.block50(in_channels=64, filters=64, stride=1, num_blocks=3)
            self.block2 = self.block50(in_channels=256, filters=128, stride=2, num_blocks=4)
            self.block3 = self.block50(in_channels=512, filters=256, stride=2, num_blocks=6)
            self.block4 = self.block50(in_channels=1024, filters=512, stride=2, num_blocks=3)
            # 결과물 Shape
            self.num_features = 2048

        # Global Average Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    # ResNet18용 Block - 3x3 2번(Not Bottleneck)
    def block18(self, in_channels, filters, stride):
        return nn.Sequential(
            NormalBlock(in_channels, filters,  stride=stride),
            NormalBlock(filters, filters, stride=1)
        )

    # ResNet50용 Block - 1x1 / 3x3 / 1x1 (Bottleneck)
    def block50(self, in_channels, filters, stride, num_blocks):
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

        # conv2_x (output size: 56x56)
        x = self.block1(x)
        # conv3_x (output size: 28x28)
        x = self.block2(x)
        # conv4_x (output size: 14x14)
        x = self.block3(x)
        # conv5_x (output size: 7x7)
        x = self.block4(x)
        # avg pooling
        x = self.avgpool(x)
        # flatten(vector)
        x = torch.flatten(x, 1)

        return x
