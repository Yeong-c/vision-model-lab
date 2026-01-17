import torch
import torch.nn as nn
import copy

class BYOL(nn.Module):
    def __init__(self, encoder, m=0.999):
        super(BYOL, self).__init__()

        self.m = m

        # Online Network
        self.online_encoder = encoder
        self.encoder = encoder

        self.projector_o = nn.Sequential(
            nn.Linear(encoder.num_features, encoder.num_features),
            nn.BatchNorm1d(encoder.num_features),
            nn.ReLU(inplace=True),
            nn.Linear(encoder.num_features, 128)
        )

        self.predictor = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128)
        )

        # Target Network
        self.target_encoder = copy.deepcopy(encoder)
        self.projector_t = nn.Sequential(
            nn.Linear(encoder.num_features, encoder.num_features),
            nn.BatchNorm1d(encoder.num_features),
            nn.ReLU(inplace=True),
            nn.Linear(encoder.num_features, 128)
        )

        # Target은 No grad
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        for param in self.projector_t.parameters():
            param.requires_grad = False

        self.criterion = nn.MSELoss()
        
    def update_ema(self, old, new):
        return old * self.m + (1 - self.m) * new
    
    def update_target_network(self):
        # Encoder Momentum Update
        for param_o, param_t in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_t.data = self.update_ema(param_t.data, param_o.data)
        # Projector Momentum Update
        for param_o, param_t in zip(self.projector_o.parameters(), self.projector_t.parameters()):
            param_t.data = self.update_ema(param_t.data, param_o.data)
    
    def online_network(self, x):
        # Representation
        online_rep = self.online_encoder(x)
        # Projection
        online_proj = self.projector_o(online_rep)
        # Predictor
        q = self.predictor(online_proj)
        # Normalize
        q = nn.functional.normalize(q, dim=1)

        return q
    
    def target_network(self, x):
        # Representation
        target_rep = self.target_encoder(x)
        # Projection
        target_proj = self.projector_t(target_rep)
        # Target Projector Detaching
        target_proj = target_proj.detach()
        # Normalize
        target_proj = nn.functional.normalize(target_proj, dim=1)

        return target_proj
    
    def forward(self, batch):
        x, _ = batch

        # Momentum Update
        with torch.no_grad():
            if self.training: self.update_target_network()

        # x: [N, View, Channel, Height, Weight] / y는 라벨(필요가 없음)
        # Concat x
        x = torch.cat([x[:, 0], x[:, 1]], dim=0)
        # Online Network 통과(v1, v2)
        q = self.online_network(x)
        q1, q2 = q.chunk(2)

        # Target Network 통과(v1, v2)
        z = self.target_network(x)
        z1, z2 = z.chunk(2)

        # Loss 계산 (Symmetric)
        loss = (self.criterion(q1, z2) + self.criterion(q2, z1)) / 2
        return loss