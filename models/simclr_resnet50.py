import torch
import torch.nn as nn
import torch.optim as optim
from models.resnet50 import ResNet50

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5, batch_size=128):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.batch_size = batch_size
        self.similarity = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
    
    def forward(self, z, labels=None):
        # z [2*Batch_Size, Projection Dim]
        N = z.shape[0] // 2

        # Cos Sim을 위해 Normalize
        z = nn.functional.normalize(z, dim=1)

        # SimCLR 유명 구현체들 참고, 행렬곱으로 이중 for문 대체
        # 대각 성분 -10억으로(자기 자신)
        sim_matrix = torch.matmul(z, z.T) / self.temperature
        mask = torch.eye(2 * N, device=z.device).bool()
        sim_matrix.masked_fill_(mask, -1e9)

        # Labels 만들기
        labels_v1 = torch.arange(N, device=z.device) + N
        labels_v2 = torch.arange(N, device=z.device)
        target = torch.cat([labels_v1, labels_v2], dim=0)

        loss = self.criterion(sim_matrix, target)
        return loss / (2 * N)

# 기존에 만들어 둔 ResNet50 이용
class SimCLR_ResNet50(ResNet50):
    def __init__(self, input_shape=(3, 32, 32), projection_dim=128):
        super(SimCLR_ResNet50, self).__init__(input_shape, num_classes=10)

        # Projection Head
        # 마지막 fc들을 MLP로 교체
        self.fc = nn.Sequential(
            nn.Linear(2048, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, projection_dim, bias=False)
        )

        # Loss Function NT-Xent
        self.criterion = NTXentLoss(temperature=0.5)

        # Optimizer
        self.optimizer = optim.SGD(
            self.parameters(),
            lr=0.5,
            momentum=0.9,
            weight_decay=1e-4
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000
        )

    def forward(self, x):
        if x.dim() == 5:
            N, V, C, H, W = x.shape
            x = x.view(N * V, C, H, W)

        z = super(SimCLR_ResNet50, self).forward(x)

        return z