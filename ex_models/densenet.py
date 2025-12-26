#DENSE_NET
from torch import nn

class DenseLayer(nn.Module):
    def __init__(self, inputs, outputs, bneck=4):
        super(DenseLayer, self).__init__()
        
        self.bn1 = nn.BatchNorm2d(inputs)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(inputs, bneck * outputs, 
                               kernel_size=1, stride=1, bias=False)

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
        self.conv = nn.Conv2d(inputs, outputs, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.pool(self.conv(self.relu(self.bn(x))))
        return out

class DenseNet(nn.Module):
    def __init__(self, k=12, dblock=(16,16,16), comp = 0.5, bneck = 4, classes=10):

        super(DenseNet, self).__init__()
        finput = 2*k 
        self.features = nn.Sequential(
            nn.Conv2d(3,finput,kernel_size=3,stride=1,padding=1,bias=False)
        )
        now_features = finput 
        for i, num_Layers in enumerate(dblock):

            block = DenseBlock(
                num_Layers=num_Layers, 
                inputs=now_features,
                outputs=k, 
                bneck=bneck 
            )

            self.features.add_module(f'denseblock{i+1}', block)
    
            now_features = now_features + num_Layers * k
            
            if i != len(dblock) - 1:
                
                trans_outputs = int(now_features * comp)
                trans = TransitionLayer(inputs=now_features,
                                        outputs=trans_outputs)
                self.features.add_module(f'transition{i+1}', trans)
                now_features = trans_outputs

        self.features.add_module('norm_final', nn.BatchNorm2d(now_features))
        self.features.add_module('relu_final', nn.ReLU(inplace=True))

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(now_features, classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = self.avg_pool(features)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out