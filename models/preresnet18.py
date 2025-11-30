import torch
import torch.nn as nn

# Pre-activation 블록 (BN -> ReLU -> Conv 순서)

class PreActBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super(PreActBlock, self).__init__()
        
        # 1. 1번째 (BN -> Conv)
        self.bn1 = nn.BatchNorm2d(in_c)
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False)
        
        # 2. 2 번째 (BN -> Conv)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)
        
        # 활성화 함수 미리 정의 
        self.relu = nn.ReLU(inplace=True)

        # 3. 숏컷 
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        # 구조: BN -> ReLU -> Conv
        
        # 첫 번째
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        
        # 두 번째
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        
        # 숏컷 더하기
        out += self.shortcut(x)
        
        return out


class PreResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(PreResNet18, self).__init__()
        
        # 초기 설정: CIFAR10은 3채널 이미지
        self.in_c = 64
        
        # 맨 처음 Conv (여기서는 BN/ReLU 안 함, 블록 안에서 하니까)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        #  nn.Sequential로 "같은 채널끼리" 묶어버림 (많이 이렇게 쓴다고 해서)
        
        # Layer 1: 64채널 유지 (2개 블록)
        self.layer1 = nn.Sequential(
            PreActBlock(64, 64, stride=1),
            PreActBlock(64, 64, stride=1)
        )
        
        # Layer 2: 128채널로 확장 (2개 블록)
        self.layer2 = nn.Sequential(
            PreActBlock(64, 128, stride=2),  # 여기서 크기 줄어듬
            PreActBlock(128, 128, stride=1)
        )
        
        # Layer 3: 256채널로 확장 (2개 블록)
        self.layer3 = nn.Sequential(
            PreActBlock(128, 256, stride=2),
            PreActBlock(256, 256, stride=1)
        )
        
        # Layer 4: 512채널로 확장 (2개 블록)
        self.layer4 = nn.Sequential(
            PreActBlock(256, 512, stride=2),
            PreActBlock(512, 512, stride=1)
        )
        
        # 마지막 정리
        self.bn_final = nn.BatchNorm2d(512)
        self.relu_final = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # 1. 입력
        out = self.conv1(x)
        
        # 2. 층별 통과 
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        # 3. 마무리 (BN -> ReLU -> Pool -> FC)
        out = self.bn_final(out)
        out = self.relu_final(out)
        
        out = self.avgpool(out)
        out = torch.flatten(out, 1) # 1자로 펴기
        out = self.fc(out)
        
        return out