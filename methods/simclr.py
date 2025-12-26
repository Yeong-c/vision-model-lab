import torch
import torch.nn as nn

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

class SimCLR(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

        # Loss Function NT-Xent
        self.criterion = NTXentLoss(temperature=0.5)

        # Projection Head(+ MLP)
        self.head = nn.Sequential(
            nn.Linear(model.num_features, model.num_features),
            nn.BatchNorm1d(model.num_features),
            nn.ReLU(),
            nn.Linear(model.num_features, 128)
        )

    def forward(self, batch):
        x, _ = batch
        # x: [N, View, Channel, Height, Weight] / y는 라벨(필요가 없음)
        x = x.transpose(0, 1)
        x = x.reshape(-1, *x.shape[2:])

        # Representation
        h = self.model(x)

        # Projection
        z = self.head(h)

        # Loss Function (NT-Xent)
        loss = self.criterion(z)
        return loss
    
    def predict(self, x):
        features = self.model(x)
        output = self.head(x)
        return output