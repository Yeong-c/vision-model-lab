import torch
import torch.nn as nn

# Residual(잔차) 연결을 돕는 블록
class Residual(nn.Module):
    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn

    def forward(self, x):
        # 입력값(x)을 결과에 다시 더해줌 (Skip Connection)
        return self.fn(x) + x

# ConvMixer의 핵심 블록
class ConvMixerBlock(nn.Module):
    def __init__(self, dim, kernel_size):
        super(ConvMixerBlock, self).__init__()

        # 1. Depthwise Convolution (공간 정보 섞기)
        # 채널마다 각각 Conv를 돌림 (groups=dim)
        self.depthwise = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=kernel_size, groups=dim, padding=kernel_size//2),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        )

        # 2. Pointwise Convolution (채널 정보 섞기)
        # 1x1 Conv로 픽셀별로 채널끼리 섞어줌
        self.pointwise = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        # Depthwise 부분에는 Residual 연결이 있다고 논문에 나옴
        out = self.depthwise(x)
        out = out + x  # 잔차 더하기
        
        # Pointwise 통과
        out = self.pointwise(out)
        return out

class ConvMixer(nn.Module):
    def __init__(self, dim, depth, kernel_size=9, patch_size=7, n_classes=10):
        super(ConvMixer, self).__init__()

        # 이 모델은 최종 feature 개수가 dim과 같음
        self.num_features = dim

        # 1. Patch Embedding (패치 단위로 쪼개서 넣기)
        # Conv2d로 패치 사이즈만큼 stride를 주면 패치 임베딩이랑 똑같다고 함
        self.patch_emb = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        )

        # 2. Main Blocks (블록 반복 쌓기)
        # 리스트에 블록을 담아서 nn.Sequential로 묶음
        blocks_list = []
        for _ in range(depth):
            block = ConvMixerBlock(dim=dim, kernel_size=kernel_size)
            blocks_list.append(block)
        
        self.blocks = nn.Sequential(*blocks_list)

        # 3. 마지막 Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 4. 가중치 초기화
        self.apply(self._init_weights)

    # 가중치 초기화 함수 
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 1. 패치 임베딩
        x = self.patch_emb(x)

        # 2. 블록들 통과
        x = self.blocks(x)

        # 3. 풀링 및 Flatten
        x = self.avgpool(x)
        x = torch.flatten(x, 1) # (Batch, dim) 형태로 펴주기

        return x


def convmixer_1024_20():
    return ConvMixer(dim=1024, depth=20, kernel_size=9, patch_size=2) 

def convmixer_512_16():
    return ConvMixer(dim=512, depth=16, kernel_size=8, patch_size=2) 

def convmixer_256_8():
    return ConvMixer(dim=256, depth=8, kernel_size=7, patch_size=2) 