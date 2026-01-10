import torch
from torch import nn

class DenseLayer(nn.Module):
    def __init__(self, inputs, outputs, bneck=4):
        super(DenseLayer, self).__init__()
        #1x1 채널 줄이기
        self.bn1 = nn.BatchNorm2d(inputs)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(inputs, bneck * outputs, 
                               kernel_size=1, stride=1, bias=False)
        #3x3 특징 추출
        self.bn2 = nn.BatchNorm2d(bneck * outputs)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bneck * outputs, outputs, 
                               kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(self.relu1(self.bn1(x)))
        out = self.conv2(self.relu2(self.bn2(out)))
        
        return torch.cat([x, out], 1) 
    
class DenseBlock(nn.Module):
    def __init__(self, num_Layers, inputs, outputs, bneck=4):
        super(DenseBlock, self).__init__()

        self.layers = nn.ModuleList()

        for i in range(num_Layers):
            #채널수 연산
            next_input = inputs + i * outputs 
            layer = DenseLayer(inputs = next_input, outputs= outputs, bneck=bneck)
            self.layers.append(layer)                                            

    def forward(self, x):
        features = x
        for layer in self.layers:
            features = layer(features)
        return features

class TransitionLayer(nn.Module):

    def __init__(self, inputs, outputs):
        super(TransitionLayer, self).__init__()
        
        self.bn = nn.BatchNorm2d(inputs)
        self.relu = nn.ReLU(inplace=True)
        #채널 수 조절
        self.conv = nn.Conv2d(inputs, outputs, kernel_size=1, stride=1, bias=False)
        #이미지 축소
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.pool(self.conv(self.relu(self.bn(x))))
        return out

class DenseNet(nn.Module):
    def __init__(self, num_blocks, growth_rate=32, comp = 0.5, bneck = 4):

        super(DenseNet, self).__init__()
        #그로우레이트 2배
        finput = 2*growth_rate 

        self.features = nn.Sequential(
            nn.Conv2d(3, finput, kernel_size=3, stride=1, padding=1, bias=False)
        )

        now_features = finput

        for i, num_Layers in enumerate(num_blocks):

            block = DenseBlock(
                num_Layers=num_Layers, 
                inputs=now_features,
                outputs=growth_rate, 
                bneck=bneck 
            )

            self.features.add_module(f'denseblock{i+1}', block)
            #채널 수 업데이트
            now_features = now_features + num_Layers * growth_rate
            #트랜지션 레이어 추가
            if i != len(num_blocks) - 1:
                
                trans_outputs = int(now_features * comp)
                trans = TransitionLayer(inputs=now_features,
                                        outputs=trans_outputs)
                self.features.add_module(f'transition{i+1}', trans)
                now_features = trans_outputs

        self.features.add_module('norm_final', nn.BatchNorm2d(now_features))
        self.features.add_module('relu_final', nn.ReLU(inplace=True))

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        #채널 개수 전달
        self.num_features = now_features

    def forward(self, x):
        features = self.features(x)
        out = self.avg_pool(features)
        out = torch.flatten(out, 1)
        return out


def densenet121():
    # k=32
    return DenseNet(num_blocks=[6,12,24,16])

def densenet169():
    # k=32
    return DenseNet(num_blocks=[6,12,32,32])

def densenet201():
    # k=32
    return DenseNet(num_blocks=[6,12,48,32])

def densenet264():
    # k=32
    return DenseNet(num_blocks=[6,12,64,48])
